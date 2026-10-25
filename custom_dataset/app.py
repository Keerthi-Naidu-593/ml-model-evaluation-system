"""Streamlit page for optional user-uploaded classification dataset evaluation."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import math
import re
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from custom_dataset.evaluation import (
    DatasetProfile,
    DatasetValidationError,
    evaluate_dataset,
    profile_dataset,
    recommended_model,
    tune_models,
    validate_for_evaluation,
)


STATE_PREFIX = "custom_dataset_"


def _key(name: str) -> str:
    return f"{STATE_PREFIX}{name}"


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            return int(match.group()) if match else default
        return int(value)
    except (ValueError, TypeError):
        return default


@st.cache_data(show_spinner=False)
def _read_csv(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(contents))


def _clear_results() -> None:
    for name in (
        "evaluation",
        "evaluation_signature",
        "tuning",
        "tuning_signature",
        "class_dist_page",
        "class_dist_page_select",
        "class_dist_target_tracker",
    ):
        st.session_state.pop(_key(name), None)


def _format_scores(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "Unavailable")
    return display


def _render_target_class_distribution(profile: DatasetProfile, target_column: str) -> None:
    st.markdown(f"**Selected Target:** `{target_column}`")
    items = list(profile.class_counts.items())
    total_classes = len(items)
    total_samples = sum(profile.class_counts.values())

    target_tracker_key = _key("class_dist_target_tracker")
    page_key = _key("class_dist_page")
    page_size_key = _key("class_dist_page_size")

    if st.session_state.get(target_tracker_key) != target_column:
        st.session_state[target_tracker_key] = target_column
        st.session_state[page_key] = 1

    page_size = _safe_int(st.session_state.get(page_size_key, 5), default=5)
    total_pages = max(1, math.ceil(total_classes / max(page_size, 1)))

    current_page = _safe_int(st.session_state.get(page_key, 1), default=1)
    if current_page < 1:
        current_page = 1
    elif current_page > total_pages:
        current_page = total_pages
    st.session_state[page_key] = current_page

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_classes)
    page_items = items[start_idx:end_idx]

    st.markdown(f"**Class Distribution** *(Total: {total_classes:,} classes, {total_samples:,} samples)*")

    table_data = [
        {
            "Class": str(label),
            "Count": f"{count:,}",
            "Share": f"{(count / total_samples * 100):.2f}%" if total_samples > 0 else "0.00%",
        }
        for label, count in page_items
    ]
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    if total_pages > 1:
        def _on_prev() -> None:
            p = _safe_int(st.session_state.get(page_key, 1), default=1)
            if p > 1:
                st.session_state[page_key] = p - 1

        def _on_next() -> None:
            p = _safe_int(st.session_state.get(page_key, 1), default=1)
            if p < total_pages:
                st.session_state[page_key] = p + 1

        def _on_size_change() -> None:
            st.session_state[page_key] = 1

        col_prev, col_page, col_next, col_size = st.columns([1, 2.5, 1, 1.2])
        with col_prev:
            st.button(
                "◀ Prev",
                on_click=_on_prev,
                disabled=current_page <= 1,
                key=_key("dist_btn_prev"),
                use_container_width=True,
            )
        with col_page:
            st.markdown(
                f"<div style='text-align: center; line-height: 2.3rem; font-weight: 600; font-size: 0.92rem; color: #334155;'>"
                f"Page {current_page} of {total_pages} &nbsp;<span style='color: #64748b; font-weight: 400;'>({start_idx + 1}–{end_idx} of {total_classes})</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_next:
            st.button(
                "Next ▶",
                on_click=_on_next,
                disabled=current_page >= total_pages,
                key=_key("dist_btn_next"),
                use_container_width=True,
            )
        with col_size:
            size_options = [5, 10, 20, 50]
            current_size_idx = size_options.index(page_size) if page_size in size_options else 0
            st.selectbox(
                "Rows per page",
                options=size_options,
                index=current_size_idx,
                key=page_size_key,
                on_change=_on_size_change,
                format_func=lambda s: f"{s} / page",
                label_visibility="collapsed",
            )


def _render_data_quality(dataframe: pd.DataFrame, target_column: str) -> None:
    st.subheader("Data Quality")
    missing = dataframe.isna().sum().rename("Missing Values").to_frame()
    missing["Missing %"] = (missing["Missing Values"] / max(len(dataframe), 1) * 100).round(2)
    st.dataframe(missing, use_container_width=True)
    try:
        profile = profile_dataset(dataframe, target_column)
        columns = st.columns(3)
        columns[0].metric("Numeric Features", len(profile.numeric_columns))
        columns[1].metric("Categorical Features", len(profile.categorical_columns))
        columns[2].metric("Unsupported Features", len(profile.unsupported_columns))
        if profile.unsupported_columns:
            st.warning("Unsupported feature columns: " + ", ".join(profile.unsupported_columns))
        if profile.empty_feature_columns:
            st.warning("Entirely missing feature columns: " + ", ".join(profile.empty_feature_columns))
        st.caption("Missing feature values are imputed inside each training fold. The uploaded source data is not modified.")
    except DatasetValidationError as exc:
        st.error(str(exc))


def _render_dataset_overview(dataframe: pd.DataFrame, filename: str, target_column: str) -> None:
    st.subheader("Dataset Overview")
    st.caption(f"Uploaded file: {filename}")
    metrics = st.columns(3)
    metrics[0].metric("Rows", f"{len(dataframe):,}")
    metrics[1].metric("Columns", len(dataframe.columns))
    metrics[2].metric("Selected Target", target_column)
    st.markdown("**Dataset Preview**")
    st.dataframe(dataframe.head(20), use_container_width=True)
    st.markdown("**Column Names and Data Types**")
    schema = pd.DataFrame({"Column": dataframe.columns, "Data Type": dataframe.dtypes.astype(str).values})
    st.dataframe(schema, use_container_width=True, hide_index=True)
    st.markdown("**Target Class Distribution**")
    distribution = dataframe[target_column].value_counts(dropna=False).rename_axis("Class").reset_index(name="Count")
    distribution["Share"] = (distribution["Count"] / max(len(dataframe), 1) * 100).round(2).astype(str) + "%"
    st.dataframe(distribution, use_container_width=True, hide_index=True)


def _render_comparison(evaluation: Dict[str, Any]) -> None:
    st.subheader("Model Comparison")
    st.caption(evaluation["evaluation_strategy"])
    st.dataframe(_format_scores(evaluation["comparison"]), use_container_width=True, hide_index=True)
    comparison = evaluation["comparison"].set_index("Model")
    fig, axis = plt.subplots(figsize=(10, 4.5))
    comparison[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]].plot(kind="bar", ax=axis)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_xlabel("")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=5)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_reports(evaluation: Dict[str, Any]) -> None:
    st.subheader("Classification Reports")
    for model_name, report in evaluation["classification_reports"].items():
        with st.expander(model_name, expanded=model_name == "Logistic Regression"):
            st.dataframe(_format_scores(report), use_container_width=True, hide_index=True)


def _render_confusion_matrices(evaluation: Dict[str, Any]) -> None:
    st.subheader("Confusion Matrices")
    labels = evaluation["profile"].class_labels
    matrices = evaluation["confusion_matrices"]
    columns = st.columns(2)
    for index, (model_name, matrix) in enumerate(matrices.items()):
        with columns[index % 2]:
            fig, axis = plt.subplots(figsize=(5, 4))
            image = axis.imshow(matrix, cmap="Blues")
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
            axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
            axis.set_yticks(np.arange(len(labels)), labels)
            axis.set_xlabel("Predicted Label")
            axis.set_ylabel("Actual Label")
            axis.set_title(model_name)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def _render_roc_curves(evaluation: Dict[str, Any]) -> None:
    st.subheader("ROC Curves")
    profile = evaluation["profile"]
    y_test = evaluation["y_test"]
    probabilities = evaluation["probabilities"]
    if len(profile.class_values) == 2:
        positive = evaluation["positive_class_index"]
        fig, axis = plt.subplots(figsize=(8, 5))
        for name, probability in probabilities.items():
            false_positive, true_positive, _ = roc_curve(
                (y_test == positive).astype(int), probability[:, positive]
            )
            axis.plot(false_positive, true_positive, label=f"{name} (AUC = {auc(false_positive, true_positive):.3f})")
        axis.plot([0, 1], [0, 1], "--", color="#64748b", label="No skill")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_title(f"ROC Curve — Positive Class: {profile.class_labels[positive]}")
        axis.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    st.caption("Multiclass ROC uses one-vs-rest curves. Select a model to inspect its class-level probabilities.")
    model_name = st.selectbox("Model for multiclass ROC", list(probabilities), key=_key("roc_model"))
    probability = probabilities[model_name]
    binarized = label_binarize(y_test, classes=np.arange(len(profile.class_values)))
    fig, axis = plt.subplots(figsize=(8, 5))
    plotted = 0
    for class_index, class_name in enumerate(profile.class_labels):
        if len(np.unique(binarized[:, class_index])) < 2:
            continue
        false_positive, true_positive, _ = roc_curve(binarized[:, class_index], probability[:, class_index])
        axis.plot(false_positive, true_positive, label=f"{class_name} (AUC = {auc(false_positive, true_positive):.3f})")
        plotted += 1
    axis.plot([0, 1], [0, 1], "--", color="#64748b", label="No skill")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"One-vs-Rest ROC Curves — {model_name}")
    if plotted:
        axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_cross_validation(evaluation: Dict[str, Any]) -> None:
    st.subheader("5-Fold Stratified Cross-Validation")
    st.caption("StratifiedKFold(n_splits=5, shuffle=True, random_state=42), run on the training partition only.")
    st.dataframe(_format_scores(evaluation["cross_validation"]), use_container_width=True, hide_index=True)


def _render_tuning(evaluation: Dict[str, Any], signature: str) -> Optional[Dict[str, Any]]:
    st.subheader("Hyperparameter Tuning")
    st.caption("GridSearchCV is optional and uses only the training partition with the same five-fold stratified CV.")
    if st.button("Run Hyperparameter Tuning", type="primary", key=_key("run_tuning")):
        with st.spinner("Running GridSearchCV for the four custom-dataset models..."):
            tuning = tune_models(evaluation)
        st.session_state[_key("tuning")] = tuning
        st.session_state[_key("tuning_signature")] = signature
    tuning = st.session_state.get(_key("tuning"))
    if st.session_state.get(_key("tuning_signature")) != signature:
        return None
    if tuning is not None:
        if not tuning["results"].empty:
            st.dataframe(_format_scores(tuning["results"]), use_container_width=True, hide_index=True)
        if tuning["failures"]:
            for model_name, error in tuning["failures"].items():
                st.warning(f"{model_name} tuning could not complete: {error}")
    return tuning


def _render_final_model(evaluation: Dict[str, Any], tuning: Optional[Dict[str, Any]]) -> None:
    st.subheader("Final Model")
    recommendation = recommended_model(evaluation, tuning)
    if recommendation is None:
        st.warning("A recommendation is not available because no valid cross-validation ROC-AUC score was produced.")
        return
    st.markdown(
        "<div class='final-model-box'><div><strong>Recommended Model</strong><div>"
        f"{recommendation['Model']}</div></div><div><strong>Selection Criterion</strong><div>"
        f"{recommendation['Source']}</div></div><div><strong>Best CV ROC-AUC</strong><div>"
        f"{recommendation['CV ROC-AUC']:.4f}</div></div></div>",
        unsafe_allow_html=True,
    )
    st.caption("Best-performing model for this dataset under the selected evaluation criterion; it is not a universal best model.")
    if recommendation["Test Metrics"]:
        metrics = pd.DataFrame([recommendation["Test Metrics"]])
        st.dataframe(_format_scores(metrics), use_container_width=True, hide_index=True)


def main(home_page=None) -> None:
    """Render the page without touching the existing case-study dashboards."""
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] { background: #f3f6fb; color: #111827; }
        .block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; background: #f3f6fb; }
        [data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #dfe7f1; }
        [data-testid="stHeader"] { background: #ffffff !important; }
        h1, h2, h3, h4, .stTabs [role="tab"] { color: #0f172a !important; letter-spacing: -.02em; }
        h1 { font-size: 2.2rem; margin-bottom: .2rem; }
        .stCaption { color: #475569; font-size: .97rem; }
        .stButton > button { border-radius: .5rem; border: 1px solid #dbe6f2; background: #ffffff; color: #1f2937; font-weight: 600; }
        .stButton > button:hover { border-color: #b7c8d9; background: #f8fbff; }
        button[kind="primary"] { background: #dfeaf8; color: #1f2937; border-color: #cfe1f7; }
        div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: .65rem; overflow: hidden; }
        .custom-note { background: #ffffff; border: 1px solid #dfe7ee; border-radius: .7rem; padding: .9rem 1rem; color: #334155; }
        .final-model-box { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; padding: 1rem 1.2rem; background: #ffffff; border: 1px solid #dfe7ee; border-radius: .75rem; margin: 1rem 0 1.25rem; }
        .final-model-box > div { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: .6rem; padding: .8rem .9rem; }
        .final-model-box strong { font-size: .72rem; letter-spacing: .07em; text-transform: uppercase; color: #64748b; }
        .final-model-box div div { margin-top: .45rem; font-size: 1.05rem; font-weight: 600; color: #0f172a; }
        @media (max-width: 900px) { .final-model-box { grid-template-columns: 1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Custom Dataset Evaluation")
    st.caption("Upload a CSV classification dataset for an isolated, leakage-safe evaluation.")
    if home_page is not None:
        st.page_link(home_page, label="Project Home", use_container_width=False)

    uploaded_file = st.file_uploader("Upload Classification Dataset", type=["csv"], key=_key("upload"))
    if uploaded_file is None:
        st.info("Upload a CSV file to select a target column and begin evaluation.")
        return

    contents = uploaded_file.getvalue()
    signature = sha256(contents).hexdigest()
    if st.session_state.get(_key("file_signature")) != signature:
        st.session_state[_key("file_signature")] = signature
        _clear_results()

    try:
        dataframe = _read_csv(contents)
    except Exception as exc:
        st.error(f"The uploaded file could not be read as CSV: {exc}")
        return
    if dataframe.empty or len(dataframe.columns) == 0:
        st.error("The uploaded CSV is empty or does not contain any columns.")
        return

    target_column = st.selectbox("Target Column", list(dataframe.columns), key=_key("target"))
    try:
        profile = profile_dataset(dataframe, target_column)
    except DatasetValidationError as exc:
        st.error(str(exc))
        return

    _render_target_class_distribution(profile, target_column)

    positive_index = None
    if len(profile.class_values) == 2:
        positive_index = st.selectbox(
            "Positive Class for Binary Metrics",
            options=[0, 1],
            format_func=lambda index: profile.class_labels[index],
            key=_key("positive_class"),
        )
    else:
        st.info("Multiclass dataset detected. Precision, recall and F1 use macro averaging; ROC-AUC uses one-vs-rest macro averaging.")

    tabs = st.tabs(
        [
            "Dataset Overview",
            "Data Quality",
            "Model Comparison",
            "Classification Reports",
            "Confusion Matrices",
            "ROC Curves",
            "Cross-Validation",
            "Hyperparameter Tuning",
            "Final Model",
        ]
    )
    with tabs[0]:
        _render_dataset_overview(dataframe, uploaded_file.name, target_column)
    with tabs[1]:
        _render_data_quality(dataframe, target_column)

    configuration_signature = f"{signature}:{target_column}:{positive_index}"
    if st.button("Run Custom Dataset Evaluation", type="primary", key=_key("run_evaluation")):
        try:
            validate_for_evaluation(dataframe, target_column)
            with st.spinner("Evaluating four models and running 5-fold stratified cross-validation..."):
                evaluation = evaluate_dataset(dataframe, target_column, positive_index)
            st.session_state[_key("evaluation")] = evaluation
            st.session_state[_key("evaluation_signature")] = configuration_signature
            st.session_state.pop(_key("tuning"), None)
            st.session_state.pop(_key("tuning_signature"), None)
        except DatasetValidationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Custom dataset evaluation could not complete: {exc}")

    evaluation = st.session_state.get(_key("evaluation"))
    if st.session_state.get(_key("evaluation_signature")) != configuration_signature:
        evaluation = None

    if evaluation is None:
        for tab in tabs[2:]:
            with tab:
                st.info("Run Custom Dataset Evaluation to populate this section.")
        return

    if evaluation["failures"]:
        for model_name, error in evaluation["failures"].items():
            st.warning(f"{model_name} could not be evaluated: {error}")
    with tabs[2]:
        _render_comparison(evaluation)
    with tabs[3]:
        _render_reports(evaluation)
    with tabs[4]:
        _render_confusion_matrices(evaluation)
    with tabs[5]:
        _render_roc_curves(evaluation)
    with tabs[6]:
        _render_cross_validation(evaluation)
    with tabs[7]:
        tuning = _render_tuning(evaluation, configuration_signature)
    tuning = st.session_state.get(_key("tuning")) if st.session_state.get(_key("tuning_signature")) == configuration_signature else None
    with tabs[8]:
        _render_final_model(evaluation, tuning)
