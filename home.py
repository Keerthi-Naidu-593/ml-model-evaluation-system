import os

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")


def resolve_data_path(filename):
    for base_path in (DATA_DIR, PROJECT_ROOT):
        candidate = os.path.join(base_path, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(DATA_DIR, filename)


st.set_page_config(
    page_title="Model Evaluation & Comparison System",
    page_icon=None,
    layout="wide",
)


def dataset_count(csv_path, fallback_value=None):
    resolved_path = resolve_data_path(csv_path)
    try:
        if os.path.exists(resolved_path):
            return int(pd.read_csv(resolved_path).shape[0])
    except Exception:
        pass
    return fallback_value


def render_home_page():
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] { background: #f4f7fb; color: #172033; }
        .block-container { max-width: 1280px; padding-top: 2.25rem; padding-bottom: 3rem; }
        [data-testid="stHeader"] { background: rgba(244,247,251,0.9); }
        h1 { color: #172033 !important; font-size: clamp(1.8rem, 4vw, 2.7rem) !important; letter-spacing: -0.04em; }
        h2, h3 { color: #172033 !important; }
        .stCaption { color: #526277 !important; font-size: 1rem; line-height: 1.6; max-width: 820px; }
        .stButton > button, [data-testid="stPageLink-NavLink"] {
            min-height: 2.65rem; border-radius: 0.55rem; font-weight: 600;
            border: 1px solid #cbd8e7; background: #ffffff; color: #1d4f7a;
        }
        .stButton > button:hover, [data-testid="stPageLink-NavLink"]:hover {
            background: #edf5fc; border-color: #8db5d9; color: #173f61;
        }
        @media (max-width: 720px) {
            .block-container { padding: 1.25rem 1rem 2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("MODEL EVALUATION & COMPARISON SYSTEM")
    st.markdown("### A Comparative Framework for Machine Learning Model Evaluation")
    st.caption(
        "A unified framework for evaluating and comparing machine learning\n"
        "models across different datasets using consistent preprocessing,\n"
        "validation, performance metrics, diagnostic analysis, and\n"
        "hyperparameter tuning."
    )

    st.markdown("---")

    st.subheader("Project Idea")
    st.markdown(
        """
        <div style="
            background: #f8fafc;
            border: 1px solid #dfe6ee;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            max-width: 980px;
        ">
            <p style="margin: 0; line-height: 1.7; color: #1f2937;">
                No single machine learning model is universally best.<br><br>
                Model performance depends on the dataset, problem characteristics,
                evaluation metrics, and validation strategy. Therefore, this system
                evaluates multiple machine learning models under a consistent
                evaluation framework instead of assuming that one model will perform
                best for every problem.<br><br>
                The objective is to understand model behaviour, compare performance,
                identify strengths and limitations, and select a suitable model for
                each dataset based on evidence from systematic evaluation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("Evaluation Framework")
    workflow = [
        "Dataset",
        "Data Preprocessing",
        "Train / Test Split",
        "Multiple ML Models",
        "Stratified 5-Fold Cross-Validation",
        "Performance Metrics",
        "Hyperparameter Tuning",
        "Model Comparison",
        "Dataset-Specific Model Selection",
    ]

    step_html = "".join(
        [
            f"<div style='display:inline-flex;align-items:center;gap:8px;'>"
            f"<span style='background:#eef4ff;border:1px solid #d9e7ff;color:#173a5a;padding:6px 12px;border-radius:999px;font-size:12px;font-weight:600;'>{step}</span>"
            + ("<span style='color:#8aa0b6;font-size:18px;'>→</span>" if idx < len(workflow) - 1 else "")
            + "</div>"
            for idx, step in enumerate(workflow)
        ]
    )
    st.markdown(
        f"<div style='overflow-x:auto; padding: 18px 0;'><div style='display:flex; flex-wrap:wrap; gap:8px; align-items:center;'>{step_html}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("Datasets Explored")
    breast_count = dataset_count("breast_cancer_dataset.csv", 569)
    card_count = dataset_count("credit_card_dataset.csv", 9993)

    breast_desc = "A classification problem focused on predicting the target class from diagnostic features."
    credit_desc = "A classification problem focused on predicting credit card default risk using customer financial attributes."

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="
                border: 1px solid #dfe6ee;
                border-radius: 14px;
                padding: 1.5rem 1.5rem 1.1rem 1.5rem;
                background: #ffffff;
                min-height: 260px;
                box-sizing: border-box;
            ">
                <div style="font-size: 12px; letter-spacing: 0.14em; color: #52677d; font-weight: 700; margin-bottom: 1rem;">BREAST CANCER</div>
                <div style="font-size: 15px; color: #1f2937; line-height: 1.8; margin-bottom: 0.5rem;">{count} instances</div>
                <div style="font-size: 15px; color: #1f2937; line-height: 1.8; margin-bottom: 1rem;">Binary Classification</div>
                <div style="font-size: 15px; color: #4b5563; line-height: 1.6; margin-top: 0.75rem; min-height: 70px;">{description}</div>
            </div>
            """.format(count=f"{breast_count:,}", description=breast_desc),
            unsafe_allow_html=True,
        )
        st.page_link(page=breast_page, label="Open Breast Cancer Dashboard", use_container_width=True)

    with col2:
        st.markdown(
            """
            <div style="
                border: 1px solid #dfe6ee;
                border-radius: 14px;
                padding: 1.5rem 1.5rem 1.1rem 1.5rem;
                background: #ffffff;
                min-height: 260px;
                box-sizing: border-box;
            ">
                <div style="font-size: 12px; letter-spacing: 0.14em; color: #52677d; font-weight: 700; margin-bottom: 1rem;">CREDIT CARD DEFAULT</div>
                <div style="font-size: 15px; color: #1f2937; line-height: 1.8; margin-bottom: 0.5rem;">{count} instances</div>
                <div style="font-size: 15px; color: #1f2937; line-height: 1.8; margin-bottom: 1rem;">Binary Classification</div>
                <div style="font-size: 15px; color: #4b5563; line-height: 1.6; margin-top: 0.75rem; min-height: 70px;">{description}</div>
            </div>
            """.format(count=f"{card_count:,}", description=credit_desc),
            unsafe_allow_html=True,
        )
        st.page_link(page=credit_page, label="Open Credit Card Dashboard", use_container_width=True)


def render_credit_card_dataset_page():
    from credit_card_app import main as credit_card_main

    credit_card_main(home_page=home_page)


def render_breast_cancer_dataset_page():
    from breast_cancer_app import main as breast_cancer_main

    breast_cancer_main(home_page=home_page)


def render_custom_dataset_page():
    from custom_dataset.app import main as custom_dataset_main

    custom_dataset_main(home_page=home_page)


if hasattr(st, "navigation"):
    home_page = st.Page(render_home_page, title="Project Home", default=True)
    credit_page = st.Page(render_credit_card_dataset_page, title="Credit Card Default")
    breast_page = st.Page(render_breast_cancer_dataset_page, title="Breast Cancer")
    custom_dataset_page = st.Page(render_custom_dataset_page, title="Custom Dataset")
    # Keep the project-level navigation visible so users can switch between
    # the two dataset dashboards at any time.
    navigation = st.navigation([home_page, credit_page, breast_page, custom_dataset_page])
    navigation.run()
else:
    render_home_page()
