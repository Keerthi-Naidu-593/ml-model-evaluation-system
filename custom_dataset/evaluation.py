"""Leakage-safe model evaluation for the Custom Dataset page only.

This module deliberately has no dependency on either predefined case study.
It does not read or write the project's saved artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
CV_SPLITS = 5


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset cannot be evaluated safely."""


@dataclass(frozen=True)
class DatasetProfile:
    numeric_columns: List[str]
    categorical_columns: List[str]
    unsupported_columns: List[str]
    empty_feature_columns: List[str]
    class_values: List[Any]
    class_labels: List[str]
    class_counts: Dict[str, int]


def _display_label(value: Any) -> str:
    """Return a safe display value while preserving the original label internally."""
    return str(value)


def profile_dataset(dataframe: pd.DataFrame, target_column: str) -> DatasetProfile:
    """Inspect a selected dataset without altering it."""
    if dataframe.empty:
        raise DatasetValidationError("The uploaded dataset is empty.")
    if target_column not in dataframe.columns:
        raise DatasetValidationError("The selected target column does not exist in this dataset.")
    if dataframe[target_column].isna().any():
        raise DatasetValidationError(
            "The target column contains missing values. Please provide a target column with no missing labels."
        )

    class_values = list(pd.unique(dataframe[target_column]))
    if len(class_values) < 2:
        raise DatasetValidationError("The target column must contain at least two classes.")

    counts = dataframe[target_column].value_counts(dropna=False)
    class_labels = [_display_label(value) for value in class_values]
    class_counts = {
        _display_label(value): int(counts.loc[value]) for value in class_values
    }

    feature_frame = dataframe.drop(columns=[target_column])
    if feature_frame.shape[1] == 0:
        raise DatasetValidationError("The dataset needs at least one feature column in addition to the target.")

    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    unsupported_columns: List[str] = []
    empty_feature_columns: List[str] = []

    for column in feature_frame.columns:
        series = feature_frame[column]
        if series.isna().all():
            empty_feature_columns.append(column)
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
            categorical_columns.append(column)
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_complex_dtype(series):
            numeric_columns.append(column)
        else:
            unsupported_columns.append(column)

    return DatasetProfile(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        unsupported_columns=unsupported_columns,
        empty_feature_columns=empty_feature_columns,
        class_values=class_values,
        class_labels=class_labels,
        class_counts=class_counts,
    )


def validate_for_evaluation(dataframe: pd.DataFrame, target_column: str) -> DatasetProfile:
    """Validate the dataset before a split, fitting any preprocessing only later."""
    profile = profile_dataset(dataframe, target_column)
    if profile.unsupported_columns:
        columns = ", ".join(profile.unsupported_columns)
        raise DatasetValidationError(
            f"Unsupported feature type detected in: {columns}. Convert date/time or complex values before upload."
        )
    if profile.empty_feature_columns:
        columns = ", ".join(profile.empty_feature_columns)
        raise DatasetValidationError(
            f"These feature columns contain only missing values: {columns}. Please remove or populate them before evaluation."
        )
    if min(profile.class_counts.values()) < 2:
        raise DatasetValidationError(
            "Each target class needs at least two rows for an 80/20 stratified train/test split."
        )
    return profile


def _make_one_hot_encoder() -> OneHotEncoder:
    """Support both recent and older scikit-learn releases."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(profile: DatasetProfile) -> ColumnTransformer:
    """Build an unfitted preprocessor. Pipelines fit it within each training fold."""
    transformers = []
    if profile.numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                profile.numeric_columns,
            )
        )
    if profile.categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _make_one_hot_encoder()),
                    ]
                ),
                profile.categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipelines(profile: DatasetProfile) -> Dict[str, Pipeline]:
    """Return the four isolated custom-dataset model pipelines."""
    def pipeline(classifier: Any) -> Pipeline:
        return Pipeline(
            [("preprocessor", make_preprocessor(profile)), ("classifier", classifier)]
        )

    return {
        "Logistic Regression": pipeline(LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)),
        "k-Nearest Neighbors": pipeline(KNeighborsClassifier()),
        "Support Vector Machine": pipeline(SVC(random_state=RANDOM_STATE, probability=True)),
        "Random Forest": pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200)),
    }


def _probability_scores(estimator: Pipeline, X: pd.DataFrame, class_count: int) -> np.ndarray:
    """Return probability columns in encoded class order, with a safe fallback."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        classes = np.asarray(estimator.classes_, dtype=int)
        aligned = np.zeros((len(X), class_count), dtype=float)
        aligned[:, classes] = probabilities
        return aligned

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X))
        if scores.ndim == 1:
            probabilities = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack((1.0 - probabilities, probabilities))
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    predictions = np.asarray(estimator.predict(X), dtype=int)
    fallback = np.zeros((len(X), class_count), dtype=float)
    fallback[np.arange(len(X)), predictions] = 1.0
    return fallback


def _binary_roc_scorer(positive_class: int):
    def score(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray) -> float:
        probabilities = _probability_scores(estimator, X, class_count=2)
        return roc_auc_score((np.asarray(y) == positive_class).astype(int), probabilities[:, positive_class])

    return score


def _multiclass_roc_scorer(class_count: int):
    def score(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray) -> float:
        probabilities = _probability_scores(estimator, X, class_count=class_count)
        return roc_auc_score(
            np.asarray(y),
            probabilities,
            labels=np.arange(class_count),
            multi_class="ovr",
            average="macro",
        )

    return score


def _scoring(class_count: int, positive_class: Optional[int]) -> Dict[str, Any]:
    if class_count == 2:
        assert positive_class is not None
        return {
            "accuracy": "accuracy",
            "precision": lambda estimator, X, y: precision_score(
                y, estimator.predict(X), pos_label=positive_class, zero_division=0
            ),
            "recall": lambda estimator, X, y: recall_score(
                y, estimator.predict(X), pos_label=positive_class, zero_division=0
            ),
            "f1": lambda estimator, X, y: f1_score(
                y, estimator.predict(X), pos_label=positive_class, zero_division=0
            ),
            "roc_auc": _binary_roc_scorer(positive_class),
        }
    return {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
        "roc_auc": _multiclass_roc_scorer(class_count),
    }


def _metric_row(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
    positive_class: Optional[int],
) -> Dict[str, float]:
    values: Dict[str, float] = {"Accuracy": accuracy_score(y_true, y_predicted)}
    if class_count == 2:
        assert positive_class is not None
        values.update(
            {
                "Precision": precision_score(y_true, y_predicted, pos_label=positive_class, zero_division=0),
                "Recall": recall_score(y_true, y_predicted, pos_label=positive_class, zero_division=0),
                "F1-score": f1_score(y_true, y_predicted, pos_label=positive_class, zero_division=0),
                "ROC-AUC": roc_auc_score(
                    (y_true == positive_class).astype(int), probabilities[:, positive_class]
                ),
            }
        )
    else:
        values.update(
            {
                "Precision": precision_score(y_true, y_predicted, average="macro", zero_division=0),
                "Recall": recall_score(y_true, y_predicted, average="macro", zero_division=0),
                "F1-score": f1_score(y_true, y_predicted, average="macro", zero_division=0),
                "ROC-AUC": roc_auc_score(
                    y_true,
                    probabilities,
                    labels=np.arange(class_count),
                    multi_class="ovr",
                    average="macro",
                ),
            }
        )
    return values


def _encoded_target(dataframe: pd.DataFrame, target_column: str, profile: DatasetProfile) -> np.ndarray:
    mapping = {value: index for index, value in enumerate(profile.class_values)}
    return dataframe[target_column].map(mapping).to_numpy(dtype=int)


def _validate_cv_split(y_train: np.ndarray, class_labels: List[str]) -> None:
    train_counts = np.bincount(y_train, minlength=len(class_labels))
    too_small = [class_labels[index] for index, count in enumerate(train_counts) if count < CV_SPLITS]
    if too_small:
        raise DatasetValidationError(
            "The 80% training partition does not have five examples for every class, so 5-fold "
            f"stratified cross-validation cannot run. Add samples for: {', '.join(too_small)}."
        )


def evaluate_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    positive_class_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Fit and assess all four models without changing the uploaded dataframe."""
    profile = validate_for_evaluation(dataframe, target_column)
    class_count = len(profile.class_values)
    if class_count == 2:
        if positive_class_index not in (0, 1):
            raise DatasetValidationError("Select a positive class before running binary evaluation.")
    else:
        positive_class_index = None

    X = dataframe.drop(columns=[target_column]).copy()
    y = _encoded_target(dataframe, target_column, profile)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    _validate_cv_split(y_train, profile.class_labels)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = _scoring(class_count, positive_class_index)
    pipelines = build_model_pipelines(profile)
    comparison_rows: List[Dict[str, Any]] = []
    cv_rows: List[Dict[str, Any]] = []
    reports: Dict[str, pd.DataFrame] = {}
    matrices: Dict[str, np.ndarray] = {}
    probabilities_by_model: Dict[str, np.ndarray] = {}
    predictions_by_model: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, Pipeline] = {}
    failures: Dict[str, str] = {}

    for name, pipeline in pipelines.items():
        try:
            pipeline.fit(X_train, y_train)
            prediction = np.asarray(pipeline.predict(X_test), dtype=int)
            probabilities = _probability_scores(pipeline, X_test, class_count)
            metrics = _metric_row(
                y_test, prediction, probabilities, class_count, positive_class_index
            )
            comparison_rows.append({"Model": name, **metrics})
            report = pd.DataFrame(
                classification_report(
                    y_test,
                    prediction,
                    labels=np.arange(class_count),
                    target_names=profile.class_labels,
                    output_dict=True,
                    zero_division=0,
                )
            ).T.reset_index().rename(columns={"index": "Class"})
            reports[name] = report
            matrices[name] = confusion_matrix(y_test, prediction, labels=np.arange(class_count))
            probabilities_by_model[name] = probabilities
            predictions_by_model[name] = prediction
            fitted_models[name] = pipeline

            cv_scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                error_score="raise",
                n_jobs=1,
            )
            cv_rows.append(
                {
                    "Model": name,
                    "CV Accuracy": float(np.mean(cv_scores["test_accuracy"])),
                    "CV Precision": float(np.mean(cv_scores["test_precision"])),
                    "CV Recall": float(np.mean(cv_scores["test_recall"])),
                    "CV F1": float(np.mean(cv_scores["test_f1"])),
                    "CV ROC-AUC": float(np.mean(cv_scores["test_roc_auc"])),
                }
            )
        except Exception as exc:  # One model should not prevent the others from reporting.
            failures[name] = str(exc)

    if not comparison_rows:
        failure_text = "; ".join(f"{name}: {message}" for name, message in failures.items())
        raise DatasetValidationError(f"No custom model could be evaluated. {failure_text}")

    strategy = (
        f"Binary metrics use '{profile.class_labels[positive_class_index]}' as the positive class. "
        "ROC-AUC uses that class's probability."
        if class_count == 2
        else "Multiclass precision, recall and F1 are macro averages; ROC-AUC uses one-vs-rest macro averaging."
    )
    return {
        "profile": profile,
        "target_column": target_column,
        "positive_class_index": positive_class_index,
        "evaluation_strategy": strategy,
        "X_test": X_test,
        "y_test": y_test,
        "comparison": pd.DataFrame(comparison_rows),
        "cross_validation": pd.DataFrame(cv_rows),
        "classification_reports": reports,
        "confusion_matrices": matrices,
        "probabilities": probabilities_by_model,
        "predictions": predictions_by_model,
        "models": fitted_models,
        "failures": failures,
        "X_train": X_train,
        "y_train": y_train,
    }


def _parameter_grids(y_train: np.ndarray) -> Dict[str, Dict[str, List[Any]]]:
    smallest_fold_train_size = max(1, int(len(y_train) * (CV_SPLITS - 1) / CV_SPLITS))
    neighbor_counts = [k for k in (3, 5, 7, 11) if k <= smallest_fold_train_size]
    return {
        "Logistic Regression": {"classifier__C": [0.1, 1.0, 10.0]},
        "k-Nearest Neighbors": {
            "classifier__n_neighbors": neighbor_counts or [1],
            "classifier__weights": ["uniform", "distance"],
        },
        "Support Vector Machine": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__kernel": ["linear", "rbf"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10],
            "classifier__min_samples_split": [2, 5],
        },
    }


def tune_models(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Run optional GridSearchCV only against the held-out-training partition."""
    profile: DatasetProfile = evaluation["profile"]
    class_count = len(profile.class_values)
    positive_class = evaluation["positive_class_index"]
    X_train = evaluation["X_train"]
    y_train = evaluation["y_train"]
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    roc_scorer = (
        _binary_roc_scorer(positive_class)
        if class_count == 2
        else _multiclass_roc_scorer(class_count)
    )
    rows: List[Dict[str, Any]] = []
    models: Dict[str, Pipeline] = {}
    failures: Dict[str, str] = {}
    pipelines = build_model_pipelines(profile)

    for name, pipeline in pipelines.items():
        try:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=_parameter_grids(y_train)[name],
                scoring=roc_scorer,
                cv=cv,
                n_jobs=1,
                error_score="raise",
                refit=True,
            )
            search.fit(X_train, y_train)
            rows.append(
                {
                    "Model": name,
                    "Best Parameters": str(search.best_params_),
                    "Best CV ROC-AUC": float(search.best_score_),
                }
            )
            models[name] = search.best_estimator_
        except Exception as exc:
            failures[name] = str(exc)

    return {"results": pd.DataFrame(rows), "models": models, "failures": failures}


def recommended_model(evaluation: Dict[str, Any], tuning: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Choose the highest available mean CV ROC-AUC; this is descriptive, not universal."""
    source = "Initial 5-fold cross-validation"
    scores = evaluation["cross_validation"]
    if tuning is not None and not tuning["results"].empty:
        scores = tuning["results"]
        source = "Tuned 5-fold cross-validation"
    if scores.empty or "CV ROC-AUC" not in scores.columns and "Best CV ROC-AUC" not in scores.columns:
        return None
    score_column = "Best CV ROC-AUC" if "Best CV ROC-AUC" in scores.columns else "CV ROC-AUC"
    winner = scores.loc[scores[score_column].idxmax()]
    comparison = evaluation["comparison"]
    metric_row = comparison.loc[comparison["Model"] == winner["Model"]]
    return {
        "Model": winner["Model"],
        "CV ROC-AUC": float(winner[score_column]),
        "Source": source,
        "Test Metrics": metric_row.to_dict("records")[0] if not metric_row.empty else {},
    }
