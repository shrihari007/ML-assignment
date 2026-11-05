from __future__ import annotations

import io
import re
from pathlib import Path

import streamlit as st

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	matthews_corrcoef,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_PATH = ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
TARGET = "Rent Category"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
	"BHK",
	"Size",
	"Bathroom",
	"Posted Year",
	"Posted Month",
	"Floor Level",
	"Total Floors",
]
CATEGORICAL_FEATURES = [
	"Area Type",
	"Area Locality",
	"City",
	"Furnishing Status",
	"Tenant Preferred",
	"Point of Contact",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_FACTORIES = {
	"Logistic Regression": lambda: LogisticRegression(
		max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
	),
	"Decision Tree": lambda: DecisionTreeClassifier(
		class_weight="balanced", min_samples_leaf=3, random_state=RANDOM_STATE
	),
	"KNN": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance"),
	"Naive Bayes": lambda: GaussianNB(),
	"Support Vector Machine": lambda: CalibratedClassifierCV(
		estimator=SVC(class_weight="balanced", random_state=RANDOM_STATE),
		cv=3,
		ensemble=False,
	),
}


def _parse_floor(value: str) -> tuple[float, float]:
	parts = re.split(r"\s+out of\s+", str(value).strip(), maxsplit=1, flags=re.I)

	def parse_level(text: str) -> float:
		match = re.search(r"-?\d+(?:\.\d+)?", text)
		if match:
			return float(match.group())
		if text.strip().lower() == "ground":
			return 0.0
		return float("nan")

	level = parse_level(parts[0])
	total = parse_level(parts[1]) if len(parts) == 2 else float("nan")
	return level, total


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
	required = {
		"Posted On",
		"BHK",
		"Rent",
		"Size",
		"Floor",
		"Area Type",
		"Area Locality",
		"City",
		"Furnishing Status",
		"Tenant Preferred",
		"Bathroom",
		"Point of Contact",
	}
	missing_columns = required.difference(raw.columns)
	if missing_columns:
		raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

	data = raw.copy()
	posted = pd.to_datetime(data["Posted On"], errors="coerce")
	data["Posted Year"] = posted.dt.year
	data["Posted Month"] = posted.dt.month
	floors = data["Floor"].map(_parse_floor)
	data["Floor Level"] = floors.map(lambda pair: pair[0])
	data["Total Floors"] = floors.map(lambda pair: pair[1])

	lower, upper = data["Rent"].quantile([1 / 3, 2 / 3]).tolist()
	data[TARGET] = pd.cut(
		data["Rent"],
		bins=[-float("inf"), lower, upper, float("inf")],
		labels=["Low", "Medium", "High"],
		include_lowest=True,
	).astype(str)
	return data[FEATURES + [TARGET]]


def make_preprocessor() -> ColumnTransformer:
	numeric = Pipeline(
		[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical = Pipeline(
		[
			("imputer", SimpleImputer(strategy="most_frequent")),
			(
				"onehot",
				OneHotEncoder(
					handle_unknown="ignore", min_frequency=5, sparse_output=False
				),
			),
		]
	)
	return ColumnTransformer(
		[("numeric", numeric, NUMERIC_FEATURES),
		 ("categorical", categorical, CATEGORICAL_FEATURES)],
		verbose_feature_names_out=False,
	)


def _calculate_auc(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> float:
	probabilities = model.predict_proba(features)
	classes = model.named_steps["classifier"].classes_
	try:
		if len(classes) == 2:
			return float(roc_auc_score(labels, probabilities[:, 1]))
		return float(
			roc_auc_score(
				labels,
				probabilities,
				labels=classes,
				average="weighted",
				multi_class="ovr",
			)
		)
	except ValueError:
		return float("nan")


def train_models(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	prepared = prepare_data(raw)
	features = prepared[FEATURES]
	labels = prepared[TARGET]
	x_train, x_test, y_train, y_test = train_test_split(
		features,
		labels,
		test_size=0.2,
		random_state=RANDOM_STATE,
		stratify=labels,
	)

	MODEL_DIR.mkdir(parents=True, exist_ok=True)
	rows = []
	for name, factory in MODEL_FACTORIES.items():
		model = Pipeline(
			[("preprocessor", make_preprocessor()), ("classifier", factory())]
		)
		model.fit(x_train, y_train)
		predictions = model.predict(x_test)
		rows.append(
			{
				"ML Model Name": name,
				"Accuracy": accuracy_score(y_test, predictions),
				"AUC": _calculate_auc(model, x_test, y_test),
				"Precision": precision_score(
					y_test, predictions, average="weighted", zero_division=0
				),
				"Recall": recall_score(
					y_test, predictions, average="weighted", zero_division=0
				),
				"F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
				"MCC": matthews_corrcoef(y_test, predictions),
			}
		)
		joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

	metrics = pd.DataFrame(rows)
	metrics.to_json(METRICS_PATH, orient="records", indent=2)
	test_data = x_test.copy()
	test_data[TARGET] = y_test
	test_data.to_csv(TEST_PATH, index=False)
	return metrics, test_data


def load_saved_models() -> dict[str, Pipeline]:
	models = {}
	for name in MODEL_FACTORIES:
		model_path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"
		if not model_path.exists():
			raise FileNotFoundError(
				f"Saved model not found: {model_path.name}. Train and include all five model files first."
			)
		models[name] = joblib.load(model_path)
	return models


def evaluate_uploaded_data(
	test_data: pd.DataFrame, models: dict[str, Pipeline]
) -> pd.DataFrame:
	required = set(FEATURES + [TARGET])
	missing_columns = required.difference(test_data.columns)
	if missing_columns:
		raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")
	if test_data.empty:
		raise ValueError("The uploaded CSV has no rows to evaluate.")
	if test_data[TARGET].isna().any():
		raise ValueError(f"The '{TARGET}' column contains missing labels.")

	features = test_data[FEATURES]
	labels = test_data[TARGET].astype(str)
	class_names = next(iter(models.values())).named_steps["classifier"].classes_
	unknown_labels = set(labels.unique()).difference(class_names)
	if unknown_labels:
		raise ValueError(
			f"Unknown target labels: {sorted(unknown_labels)}. Expected labels: {list(class_names)}."
		)

	rows = []
	for name, model in models.items():
		predictions = model.predict(features)
		rows.append(
			{
				"ML Model Name": name,
				"Accuracy": accuracy_score(labels, predictions),
				"AUC": _calculate_auc(model, features, labels),
				"Precision": precision_score(
					labels, predictions, average="weighted", zero_division=0
				),
				"Recall": recall_score(
					labels, predictions, average="weighted", zero_division=0
				),
				"F1": f1_score(labels, predictions, average="weighted", zero_division=0),
				"MCC": matthews_corrcoef(labels, predictions),
			}
		)
	return pd.DataFrame(rows)


def evaluate_model_details(
	test_data: pd.DataFrame, model: Pipeline
) -> tuple[pd.DataFrame, pd.DataFrame]:
	features = test_data[FEATURES]
	labels = test_data[TARGET].astype(str)
	class_names = model.named_steps["classifier"].classes_
	predictions = model.predict(features)
	report = pd.DataFrame(
		classification_report(
			labels,
			predictions,
			labels=class_names,
			output_dict=True,
			zero_division=0,
		)
	).transpose()
	matrix = pd.DataFrame(
		confusion_matrix(labels, predictions, labels=class_names),
		index=pd.Index(class_names, name="Actual"),
		columns=pd.Index(class_names, name="Predicted"),
	)
	return matrix, report


def run_app() -> None:

	st.set_page_config(page_title="House Rent Classifier Evaluation", layout="wide")
	st.title("House Rent Category Classification")
	st.markdown(
		"Dataset details: [House Rent Prediction Dataset on Kaggle]"
		"(https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset/data)"
	)
	st.caption(
		"Upload labeled test data to evaluate the saved classifiers. The app does not train models."
	)

	if TEST_PATH.exists():
		st.download_button(
			"Download sample test_data.csv",
			data=TEST_PATH.read_bytes(),
			file_name="test_data.csv",
			mime="text/csv",
		)

	upload = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
	if upload is None:
		st.info("Upload a CSV with all 13 model features and the 'Rent Category' target column.")
		st.stop()

	model_signature = tuple(
		(path.name, path.stat().st_mtime_ns, path.stat().st_size)
		for path in sorted(MODEL_DIR.glob("*.joblib"))
	)

	@st.cache_resource
	def get_cached_models(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Pipeline]:
		return load_saved_models()

	@st.cache_data(show_spinner=False)
	def get_cached_evaluation(
		csv_content: bytes, signature: tuple[tuple[str, int, int], ...]
	) -> tuple[pd.DataFrame, pd.DataFrame]:
		test_data = pd.read_csv(io.BytesIO(csv_content))
		results = evaluate_uploaded_data(
			test_data, get_cached_models(signature)
		)
		return test_data, results

	@st.cache_data(show_spinner=False)
	def get_cached_model_details(
		csv_content: bytes,
		model_name: str,
		signature: tuple[tuple[str, int, int], ...],
	) -> tuple[pd.DataFrame, pd.DataFrame]:
		test_data = pd.read_csv(io.BytesIO(csv_content))
		model = get_cached_models(signature)[model_name]
		return evaluate_model_details(test_data, model)

	try:
		with st.spinner("Evaluating test data with the saved models..."):
			test_data, results = get_cached_evaluation(
				upload.getvalue(), model_signature
			)
	except Exception as error:
		st.error(f"Could not evaluate uploaded data: {error}")
		st.stop()

	st.subheader(f"Results for uploaded test data ({len(test_data):,} rows)")
	st.dataframe(
		results.style.format(
			{
				metric: "{:.3f}"
				for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
			}
		),
		hide_index=True,
		use_container_width=True,
	)
	selected_model = st.selectbox("Model details", list(MODEL_FACTORIES))
	details_placeholder = st.empty()
	try:
		with details_placeholder.container():
			with st.spinner(f"Loading {selected_model} tables..."):
				matrix, report = get_cached_model_details(
					upload.getvalue(), selected_model, model_signature
				)
			col_matrix, col_report = st.columns(2)
			with col_matrix:
				st.subheader("Confusion Matrix")
				st.dataframe(matrix, use_container_width=True)
			with col_report:
				st.subheader("Classification Report")
				st.dataframe(report.style.format("{:.3f}"), use_container_width=True)
	except Exception as error:
		details_placeholder.error(f"Could not load {selected_model} tables: {error}")


if __name__ == "__main__":
	run_app()
