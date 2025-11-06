"""Decision tree classifier for the student career aspiration dataset."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from joblib import parallel_backend
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

DATA_PATH = Path(__file__).resolve().parent.parent / "student-scores.csv"
TARGET_COLUMN = "career_aspiration"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and drop rows without a known career aspiration."""
    data = pd.read_csv(path)
    return data[data[TARGET_COLUMN] != "Unknown"].copy()


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create hand-crafted features that capture aggregate performance."""
    subject_columns = [
        "math_score",
        "history_score",
        "physics_score",
        "chemistry_score",
        "biology_score",
        "english_score",
        "geography_score",
    ]

    engineered = data.copy()
    engineered["total_score"] = engineered[subject_columns].sum(axis=1)
    engineered["average_score"] = engineered["total_score"] / len(subject_columns)
    engineered["best_subject_score"] = engineered[subject_columns].max(axis=1)
    engineered["worst_subject_score"] = engineered[subject_columns].min(axis=1)
    engineered["study_efficiency"] = engineered["average_score"] / engineered[
        "weekly_self_study_hours"
    ].replace(0, 1)
    return engineered


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target columns."""
    cleaned = data.drop(columns=["id", "first_name", "last_name", "email"])
    X = cleaned.drop(columns=[TARGET_COLUMN])
    y = cleaned[TARGET_COLUMN]
    return X, y


def build_pipeline(
    feature_frame: pd.DataFrame,
    random_state: int = 42,
    max_depth: int | None = None,
    step_name: str = "model",
) -> Pipeline:
    """Construct the preprocessing and model pipeline."""
    numeric_features = feature_frame.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = feature_frame.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = DecisionTreeClassifier(
        random_state=random_state, max_depth=max_depth, class_weight="balanced"
    )
    return Pipeline(steps=[("preprocess", preprocessor), (step_name, model)])


def train_and_evaluate(
    random_state: int = 42, test_size: float = 0.2, max_depth: int | None = None
) -> dict[str, object]:
    """Train the decision tree classifier and return artifacts and metrics."""
    dataset = engineer_features(load_dataset())
    X, y = prepare_features(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline(
        X_train, random_state=random_state, max_depth=max_depth, step_name="clf"
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    param_grid = {
        "clf__criterion": ["gini", "entropy", "log_loss"],
        "clf__max_depth": [6, 10, 14, 18, None],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5, 10],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="f1_macro",
        verbose=0,
    )
    with parallel_backend("threading"):
        grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params = grid.best_params_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    recall = recall_score(y_test, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    return {
        "pipeline": best_model,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "balanced_accuracy": bal_acc,
        "report": report,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "best_params": best_params,
    }


def train_test_baseline(
    random_state: int = 42, test_size: float = 0.2
) -> dict[str, object]:
    """Train a most-frequent baseline classifier for comparison."""
    dataset = engineer_features(load_dataset())
    X, y = prepare_features(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)

    return {
        "baseline_accuracy": accuracy_score(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    baseline_results = train_test_baseline()
    print(f"Baseline (most frequent) accuracy: {baseline_results['baseline_accuracy']:.3f}")
    results = train_and_evaluate()
    print("Best params:", results["best_params"])
    print(f"Decision tree accuracy: {results['accuracy']:.3f}")
    print(f"Decision tree balanced accuracy: {results['balanced_accuracy']:.3f}")
    print(f"Decision tree precision (macro): {results['precision_macro']:.3f}")
    print(f"Decision tree recall (macro): {results['recall_macro']:.3f}")
    print(f"Decision tree F1 (macro): {results['f1_macro']:.3f}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "accuracy": results["accuracy"],
                "precision_macro": results["precision_macro"],
                "recall_macro": results["recall_macro"],
                "f1_macro": results["f1_macro"],
                "balanced_accuracy": results["balanced_accuracy"],
                "best_params": json.dumps(results["best_params"]) if results["best_params"] else "",
            }
        ]
    ).to_csv(
        ARTIFACTS_DIR / "decision_tree_test_metrics.csv",
        index=False,
    )
    print("Classification report:")
    print(results["report"])
