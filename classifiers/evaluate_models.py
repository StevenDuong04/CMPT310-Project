"""Evaluate multiple classifiers with consistent preprocessing and 10-fold CV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:  # pragma: no cover - allow running as module or script
    from .decisionTree import build_pipeline, engineer_features, load_dataset, prepare_features
except ImportError:  # pragma: no cover
    from decisionTree import build_pipeline, engineer_features, load_dataset, prepare_features


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_models() -> pd.DataFrame:
    """Run cross-validation for multiple classifiers and return aggregated metrics."""
    dataset = engineer_features(load_dataset())
    X, y = prepare_features(dataset)

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=5000, n_jobs=-1, solver="saga"),
        "KNN": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(random_state=42),
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }

    results: list[dict[str, object]] = []
    for name, model in models.items():
        pipeline = build_pipeline(X, random_state=42)
        pipeline.set_params(model=model)

        with parallel_backend("threading"):
            cv_scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

        for metric in scoring:
            scores = cv_scores[f"test_{metric}"]
            results.append(
                {
                    "model": name,
                    "metric": metric,
                    "mean": scores.mean(),
                    "std": scores.std(),
                }
            )

    df = pd.DataFrame(results)
    f1_order = (
        df[df["metric"] == "f1"].sort_values(by="mean", ascending=False)["model"].tolist()
    )
    if not f1_order:
        f1_order = list(models.keys())

    df["model"] = pd.Categorical(df["model"], categories=f1_order, ordered=True)
    df.sort_values(by=["model", "metric"], inplace=True)
    df = df.reset_index(drop=True)

    output_path = ARTIFACTS_DIR / "cv_10fold_metrics.csv"
    df.to_csv(output_path, index=False)

    return df


def format_results_table(df: pd.DataFrame) -> str:
    """Return a formatted string representation of the results DataFrame."""
    pivot = df.pivot(index="model", columns="metric", values=["mean", "std"])
    pivot = pivot.swaplevel(axis=1).sort_index(axis=1, level=0)
    pivot = pivot.loc[df["model"].cat.categories]
    formatted = pivot.apply(lambda col: col.map(lambda x: f"{x:.3f}"))
    return formatted.to_string()

if __name__ == "__main__":
    table = evaluate_models()
    # Print only the metrics in a tidy layout without altering saved CSV structure.
    print("10-fold cross-validation metrics (mean/std):")
    print(format_results_table(table))
