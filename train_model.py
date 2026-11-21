from __future__ import annotations

import argparse
import json
import pickle
import platform
from pathlib import Path
from statistics import fmean, pstdev

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["pclass", "sex", "age", "fare"]
NUMERIC_FEATURES = ["pclass", "age", "fare"]
CATEGORICAL_FEATURES = ["sex"]
ARTIFACT_DIR = Path("artifacts")


def load_data():
    X, y = fetch_openml(
        "titanic",
        version=1,
        as_frame=True,
        return_X_y=True,
        parser="liac-arff",  # Force the liac-arff parser to avoid the pandas parsing bug
    )
    return X[FEATURES], y


def preview_data() -> None:
    X, _ = load_data()
    print(f"Rows: {X.shape[0]}")
    print(f"Features: {', '.join(X.columns)}")
    X.info()


def show_raw_training_problem() -> None:
    X, y = load_data()
    raw_model = RandomForestClassifier(n_estimators=10, random_state=42)

    try:
        raw_model.fit(X, y)
    except ValueError as error:
        print("Expected raw-data failure:")
        print(error)
    else:
        print("Raw training unexpectedly succeeded; inspect the feature types before continuing.")


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
        ]
    )


def build_reference_stats(model: Pipeline, X_train) -> dict[str, dict[str, float]]:
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    rows = transformed.tolist()

    reference_stats = {}
    for index, feature_name in enumerate(feature_names):
        values = [float(row[index]) for row in rows]
        reference_stats[str(feature_name)] = {
            "mean": round(fmean(values), 6),
            "std": round(pstdev(values), 6),
        }
    return reference_stats


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def train_and_save() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    classes = [str(value) for value in model.named_steps["classifier"].classes_]
    matrix = confusion_matrix(y_test, predictions, labels=classes).tolist()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "model.pkl"
    metadata_path = ARTIFACT_DIR / "metadata.json"
    reference_path = ARTIFACT_DIR / "reference_stats.json"

    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file, protocol=5)

    metadata = {
        "dataset": {
            "name": "titanic",
            "openml_version": 1,
            "source": "https://www.openml.org/d/40945",
        },
        "features": FEATURES,
        "classes": classes,
        "test_size": 0.2,
        "random_state": 42,
        "accuracy": round(float(accuracy), 6),
        "confusion_matrix": matrix,
        "python_version": platform.python_version(),
        "pandas_version": __import__("pandas").__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
    }
    write_json(metadata_path, metadata)
    write_json(reference_path, build_reference_stats(model, X_train))

    print("Training complete")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Reference statistics: {reference_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible Titanic survival baseline."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the selected OpenML Titanic features.",
    )
    parser.add_argument(
        "--raw-check",
        action="store_true",
        help="Demonstrate why the raw mixed-type data needs preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview:
        preview_data()
    elif args.raw_check:
        show_raw_training_problem()
    else:
        train_and_save()


if __name__ == "__main__":
    main()