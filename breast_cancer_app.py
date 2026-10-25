import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

try:
    import plotly.express as px
except ModuleNotFoundError:
    px = None

try:
    import altair as alt
except ModuleNotFoundError:
    alt = None

APP_PREFIX = "breast_cancer"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")


def resolve_data_path(filename):
    for base_path in (DATA_DIR, PROJECT_ROOT):
        candidate = os.path.join(base_path, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(DATA_DIR, filename)


def state_key(key):
    return f"{APP_PREFIX}_{key}"


def main(home_page=None):
    # home.py configures the routed app. Configure only when this file is
    # launched directly so navigating from the home page cannot raise an error.
    if home_page is None:
        st.set_page_config(
            page_title="Model Evaluation & Comparison",
            page_icon=None,
            layout="wide",
            initial_sidebar_state="expanded",
        )

    @st.cache_resource
    def load_model_and_metadata():
        model_path = 'breast_cancer_best_model.pkl'
        metadata_path = 'breast_cancer_metadata.json'
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            return None, None
        pipeline = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return pipeline, metadata

    def _file_mtime(path):
        return os.path.getmtime(path) if os.path.exists(path) else 0

    @st.cache_data
    def load_evaluation_data(mtime_test, mtime_cv, mtime_tuning):
        test_path = resolve_data_path('breast_cancer_test_results.csv')
        cv_path = resolve_data_path('breast_cancer_cv_results.csv')
        tuning_path = resolve_data_path('breast_cancer_tuning_results.csv')

        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
        cv_df = pd.read_csv(cv_path) if os.path.exists(cv_path) else None
        tuning_df = pd.read_csv(tuning_path) if os.path.exists(tuning_path) else None
        return test_df, cv_df, tuning_df

    pipeline, metadata = load_model_and_metadata()
    test_path = resolve_data_path('breast_cancer_test_results.csv')
    cv_path = resolve_data_path('breast_cancer_cv_results.csv')
    tuning_path = resolve_data_path('breast_cancer_tuning_results.csv')
    test_results_df, cv_results_df, tuning_results_df = load_evaluation_data(
        _file_mtime(test_path),
        _file_mtime(cv_path),
        _file_mtime(tuning_path),
    )

    if pipeline is None or metadata is None:
        st.error('Error: Required model (breast_cancer_best_model.pkl) or metadata (breast_cancer_metadata.json) not found.')
        st.stop()

    def predict_with_probability(model_pipeline, df_input):
        prediction = model_pipeline.predict(df_input)[0]
        try:
            prediction_proba = model_pipeline.predict_proba(df_input)[0]
        except Exception:
            try:
                from breast_cancer_analysis import _safe_predict_proba
                prediction_proba = _safe_predict_proba(model_pipeline, df_input)[0]
            except Exception:
                if hasattr(model_pipeline, 'decision_function'):
                    scores = model_pipeline.decision_function(df_input)
                    s = np.asarray(scores).ravel()
                    probs1 = 1.0 / (1.0 + np.exp(-s))
                    prediction_proba = [1 - probs1[0], probs1[0]]
                else:
                    p = int(prediction)
                    prediction_proba = [0.0, 0.0]
                    prediction_proba[p] = 1.0
        return prediction, prediction_proba

    def format_metric_table(df):
        if df is None:
            return df
        display = df.copy()
        for col in display.columns:
            if pd.api.types.is_numeric_dtype(display[col]):
                display[col] = display[col].map(lambda x: f'{float(x):.4f}' if pd.notna(x) else '')
            else:
                display[col] = display[col].map(lambda x: str(x) if pd.notna(x) else '')
        return display

    def render_plot(image_path, wide=False):
        """Keep charts readable on desktop without sacrificing mobile width."""
        if not os.path.exists(image_path):
            return
        layout = [1, 10, 1] if wide else [2, 8, 2]
        _, chart_column, _ = st.columns(layout)
        with chart_column:
            st.image(image_path, use_container_width=True)

    def render_sidebar():
        st.sidebar.markdown("<div class='sidebar-brand'>MODEL EVALUATION</div><div class='sidebar-subtitle'>Breast Cancer</div>", unsafe_allow_html=True)

        if state_key('current_page') not in st.session_state:
            st.session_state[state_key('current_page')] = 'Overview'

        if state_key('nav_override') in st.session_state:
            selected_page = st.session_state.pop(state_key('nav_override'))
            st.session_state[state_key('current_page')] = selected_page
            st.session_state[state_key('sidebar_page')] = selected_page

        if home_page is not None:
            st.sidebar.page_link(home_page, label='Project Home', use_container_width=True)

        st.sidebar.markdown("<div class='sidebar-group'>DASHBOARD</div>", unsafe_allow_html=True)
        pages = ['Overview', 'Model Comparison', 'Cross-Validation', 'Diagnostics', 'Hyperparameter Tuning', 'Default Prediction']
        active_page = st.session_state[state_key('current_page')]
        if active_page not in pages:
            active_page = 'Overview'
        selected_page = st.sidebar.radio(
            'Dashboard pages', pages, index=pages.index(active_page),
            key=state_key('sidebar_page'), label_visibility='collapsed',
        )
        st.session_state[state_key('current_page')] = selected_page

        selected_model = metadata.get('selected_model', 'Not available')
        test_metrics = metadata.get('test_metrics', {})
        auc_key = next((key for key in test_metrics if 'ROC-AUC' in key), None)
        score_label = auc_key or next(iter(test_metrics), None)
        score_value = test_metrics.get(score_label) if score_label else None
        score_html = (
            f"<div class='best-model-score'><span>{score_label}</span><strong>{float(score_value):.4f}</strong></div>"
            if score_value is not None else ""
        )
        st.sidebar.markdown(
            f"<div class='best-model-panel'><div class='best-model-label'>SELECTED MODEL</div>"
            f"<div class='best-model-name'>{selected_model}</div>{score_html}</div>",
            unsafe_allow_html=True,
        )

        return st.session_state[state_key('current_page')]

    def render_overview():
        st.title('Model Evaluation & Comparison')
        st.caption('Comparative evaluation of machine learning models for breast cancer diagnosis.')

        feature_count = metadata.get('num_features', 30)
        model_count = 4
        validation_text = 'Stratified 5-Fold CV'

        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.markdown("<div class='stat-block'><div class='stat-label'>Dataset</div><div class='stat-value'>Wisconsin Diagnostic Breast Cancer</div></div>", unsafe_allow_html=True)
        with summary_cols[1]:
            st.markdown(f"<div class='stat-block'><div class='stat-label'>Features</div><div class='stat-value'>{feature_count}</div></div>", unsafe_allow_html=True)
        with summary_cols[2]:
            st.markdown(f"<div class='stat-block'><div class='stat-label'>Models</div><div class='stat-value'>{model_count}</div></div>", unsafe_allow_html=True)
        with summary_cols[3]:
            st.markdown(f"<div class='stat-block'><div class='stat-label'>Validation</div><div class='stat-value'>{validation_text}</div></div>", unsafe_allow_html=True)

        st.subheader('Evaluation Summary')
        if test_results_df is not None:
            st.dataframe(format_metric_table(test_results_df), use_container_width=True, hide_index=True)
        else:
            st.warning('Test results file not found.')

        st.subheader('Model Performance')
        st.caption('Test-set performance comparison across evaluated models')
        performance_plot = os.path.join('plots', 'breast_cancer_model_comparison.png')
        if os.path.exists(performance_plot):
            render_plot(performance_plot)

        st.markdown('### Process Overview')
        steps = [
            'Dataset',
            'Preprocessing',
            'Train/Test Split',
            'Model Training',
            'Cross-Validation',
            'Hyperparameter Tuning',
            'Model Comparison',
        ]
        step_html = ''.join(
            [
                f"<div class='process-step'><span class='process-index'>{idx + 1}</span><span class='process-label'>{step}</span></div>"
                + ("<span class='process-arrow'>→</span>" if idx < len(steps) - 1 else "")
                for idx, step in enumerate(steps)
            ]
        )
        st.markdown(f"<div class='process-strip'>{step_html}</div>", unsafe_allow_html=True)

        st.subheader('Final Model')
        selected_model = metadata.get('selected_model', 'Logistic Regression')
        cv_roc = metadata.get('cv_metrics', {}).get('CV ROC-AUC', 0.0)
        st.markdown(
            f"<div class='final-model-box'><div><strong>Selected Model</strong><div>{selected_model}</div></div><div><strong>Selection Criterion</strong><div>Highest Mean 5-Fold CV ROC-AUC</div></div><div><strong>Mean CV ROC-AUC</strong><div>{float(cv_roc):.4f}</div></div></div>",
            unsafe_allow_html=True,
        )

        metric_cards = st.columns(6)
        test_metrics = metadata.get('test_metrics', {})
        metric_order = [
            ('Accuracy', 'Accuracy', 'Accuracy'),
            ('Malignant Precision', 'Malignant Precision', 'Malignant Precision'),
            ('Malignant Recall', 'Malignant Recall', 'Malignant Recall'),
            ('Malignant F1', 'Malignant F1', 'Malignant F1'),
            ('Malignant Test ROC-AUC', 'Malignant Test ROC-AUC', 'Malignant Test ROC-AUC'),
            ('CV ROC-AUC', 'CV ROC-AUC', 'CV ROC-AUC'),
        ]
        for idx, (label, fallback_key, target_key) in enumerate(metric_order):
            with metric_cards[idx]:
                value = test_metrics.get(target_key)
                if value is None and fallback_key in test_metrics:
                    value = test_metrics[fallback_key]
                if value is None and label == 'CV ROC-AUC':
                    value = cv_roc
                if value is None:
                    value = 0.0
                st.metric(label, f'{float(value):.4f}')

    def render_model_comparison():
        st.title('Model Comparison')
        st.caption('Comparison of classification performance across four machine-learning models.')

        if test_results_df is not None:
            st.dataframe(test_results_df, use_container_width=True, hide_index=True)
            dashboard_palette = ['#4e79a7', '#f4b942', '#59b0a9', '#e76f51']

            st.subheader('Metric Comparison')
            metric_aliases = {
                'Accuracy': ['accuracy'],
                'Precision': ['precision'],
                'Recall': ['recall'],
                'F1-Score': ['f1score', 'f1'],
            }
            metric_columns = ['Model']
            for wanted, aliases in metric_aliases.items():
                match = None
                for column in test_results_df.columns:
                    normalized = column.lower().replace('-', '').replace(' ', '')
                    if any(alias in normalized for alias in aliases):
                        match = column
                        break
                if match is not None:
                    metric_columns.append(match)

            roc_match = None
            for column in test_results_df.columns:
                normalized = column.lower().replace('-', '').replace(' ', '')
                if normalized.endswith('rocauc') or 'rocauc' in normalized:
                    roc_match = column
                    break
            if roc_match is not None and roc_match not in metric_columns:
                metric_columns.append(roc_match)

            if len(metric_columns) > 1:
                metric_df = test_results_df[metric_columns].copy()
                metric_long = metric_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
                metric_long['Percentage'] = metric_long['Score'] * 100
                if px is not None:
                    metric_chart = px.bar(
                        metric_long,
                        x='Metric',
                        y='Score',
                        color='Model',
                        barmode='group',
                        hover_name='Model',
                        hover_data={
                            'Metric': False,
                            'Score': ':.4f',
                            'Percentage': ':.2f',
                        },
                        title='Accuracy, Precision, Recall, F1-Score, and ROC-AUC by Model',
                    )
                    metric_chart.update_traces(
                        hovertemplate='<b>%{customdata[0]}</b><br>Metric: %{x}<br>Score: %{y:.4f}<br>Percentage: %{customdata[1]:.2f}%<extra></extra>',
                        customdata=metric_long[['Model', 'Percentage']].values,
                    )
                    metric_chart.update_layout(xaxis_title='Metric', yaxis_title='Score', legend_title_text='Model')
                    st.plotly_chart(metric_chart, use_container_width=True)
                elif alt is not None:
                    metric_chart = alt.Chart(metric_long).mark_bar().encode(
                        x='Metric:N',
                        y='Score:Q',
                        color='Model:N',
                        tooltip=['Model:N', 'Metric:N', alt.Tooltip('Score:Q', format='.4f'), alt.Tooltip('Percentage:Q', format='.2f')],
                    ).properties(title='Accuracy, Precision, Recall, F1-Score, and ROC-AUC by Model')
                    st.altair_chart(metric_chart, use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(11, 5))
                    sns.barplot(data=metric_long, x='Metric', y='Score', hue='Model', palette=dashboard_palette, ax=ax)
                    ax.set_title('Accuracy, Precision, Recall, F1-Score, and ROC-AUC by Model')
                    ax.set_xlabel('Metric')
                    ax.set_ylabel('Score')
                    ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc='upper left')
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.4f', padding=3)
                    st.pyplot(fig)
            else:
                st.info('No metric columns were found for the performance chart.')

            st.subheader('ROC-AUC Comparison')
            roc_match = None
            for column in test_results_df.columns:
                normalized = column.lower().replace('-', '').replace(' ', '')
                if normalized.endswith('rocauc') or 'rocauc' in normalized:
                    roc_match = column
                    break
            if roc_match is not None:
                roc_df = test_results_df[['Model', roc_match]].sort_values(by=roc_match, ascending=False).copy()
                roc_df = roc_df.rename(columns={roc_match: 'ROC-AUC'})
                roc_df['Percentage'] = roc_df['ROC-AUC'] * 100
                if px is not None:
                    roc_chart = px.bar(
                        roc_df,
                        x='ROC-AUC',
                        y='Model',
                        orientation='h',
                        color='Model',
                        hover_name='Model',
                        hover_data={'ROC-AUC': ':.4f', 'Percentage': ':.2f'},
                        title='Test-set ROC-AUC by Model',
                    )
                    roc_chart.update_traces(
                        hovertemplate='<b>%{y}</b><br>ROC-AUC: %{x:.4f}<br>Percentage: %{customdata:.2f}%<extra></extra>',
                        customdata=roc_df['Percentage'].values,
                    )
                    roc_chart.update_layout(xaxis_title='ROC-AUC', yaxis_title='Model', legend_title_text='Model')
                    st.plotly_chart(roc_chart, use_container_width=True)
                elif alt is not None:
                    roc_chart = alt.Chart(roc_df).mark_bar().encode(
                        x='ROC-AUC:Q',
                        y='Model:N',
                        color='Model:N',
                        tooltip=['Model:N', alt.Tooltip('ROC-AUC:Q', format='.4f'), alt.Tooltip('Percentage:Q', format='.2f')],
                    ).properties(title='Test-set ROC-AUC by Model')
                    st.altair_chart(roc_chart, use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(10, 3))
                    sns.barplot(data=roc_df, x='ROC-AUC', y='Model', palette=dashboard_palette, ax=ax, orient='h')
                    ax.set_title('Test-set ROC-AUC by Model')
                    ax.set_xlabel('ROC-AUC')
                    ax.set_ylabel('Model')
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.4f', padding=3)
                    st.pyplot(fig)
            else:
                st.info('No ROC-AUC column was found for the comparison chart.')
        else:
            st.warning('Test results file not found.')

    def render_cross_validation():
        st.title('Cross-Validation')
        st.caption('Stratified 5-Fold Cross-Validation')
        st.markdown('> Stratified 5-Fold Cross-Validation evaluates each model across five folds while maintaining the class distribution in each fold.')

        if cv_results_df is not None:
            st.dataframe(format_metric_table(cv_results_df), use_container_width=True, hide_index=True)
        else:
            st.warning('Cross-validation results file not found.')

        with st.expander('Why 5-Fold Cross-Validation?'):
            st.markdown(
                'This validation strategy reduces variance in performance estimates by averaging model behavior across five class-balanced train/test splits. It is especially useful for imbalanced classification problems where a single split can misrepresent minority-class performance.'
            )

    def render_diagnostics():
        st.title('Model Diagnostics')
        tabs = st.tabs(['ROC Curves', 'Confusion Matrices'])

        with tabs[0]:
            roc_path = os.path.join('plots', 'breast_cancer_roc_curves.png')
            if os.path.exists(roc_path):
                render_plot(roc_path)
            st.caption('ROC curves illustrate the trade-off between true-positive and false-positive rates across classification thresholds.')

        with tabs[1]:
            cm_tabs = st.tabs(['Logistic Regression', 'kNN', 'SVM', 'Random Forest', 'Combined View'])
            with cm_tabs[0]:
                p = os.path.join('plots', 'breast_cancer_cm_logistic_regression.png')
                if os.path.exists(p):
                    render_plot(p)
            with cm_tabs[1]:
                p = os.path.join('plots', 'breast_cancer_cm_knn.png')
                if os.path.exists(p):
                    render_plot(p)
            with cm_tabs[2]:
                p = os.path.join('plots', 'breast_cancer_cm_svm.png')
                if os.path.exists(p):
                    render_plot(p)
            with cm_tabs[3]:
                p = os.path.join('plots', 'breast_cancer_cm_random_forest.png')
                if os.path.exists(p):
                    render_plot(p)
            with cm_tabs[4]:
                p = os.path.join('plots', 'breast_cancer_confusion_matrices.png')
                if os.path.exists(p):
                    render_plot(p, wide=True)

    def render_tuning():
        st.title('Hyperparameter Tuning')
        st.caption('GridSearchCV-based model parameter evaluation')
        st.markdown('> GridSearchCV evaluates predefined parameter combinations using cross-validation to identify configurations that perform well under the selected evaluation criterion.')

        if tuning_results_df is not None:
            st.dataframe(format_metric_table(tuning_results_df), use_container_width=True, hide_index=True)
        else:
            st.warning('Tuning results file not found.')

    def render_feature_importance(model_pipeline, model_metadata):
        st.subheader('Feature Importance')
        classifier = model_pipeline.named_steps['classifier']
        feature_names = model_metadata.get('feature_names', [])

        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values(by='Importance', ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=imp_df, color='#2a5d9f', ax=ax, dodge=False)
            ax.set_title('Top Predictive Features')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.grid(axis='x', linestyle='--', alpha=0.25)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            st.pyplot(fig)
        elif hasattr(classifier, 'coef_'):
            coefficients = classifier.coef_[0]
            coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
            coef_df['Absolute_Coefficient'] = coef_df['Coefficient'].abs()
            coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Coefficient', y='Feature', data=coef_df, color='#2a5d9f', ax=ax, dodge=False)
            ax.set_title('Feature Coefficients')
            ax.set_xlabel('Coefficient')
            ax.set_ylabel('Feature')
            ax.grid(axis='x', linestyle='--', alpha=0.25)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            st.pyplot(fig)
        else:
            st.info('Feature importance is not directly available for this model type.')

    def render_prediction():
        st.title('Breast Cancer Diagnosis')
        st.caption('Predict whether a diagnostic sample is malignant or benign using the selected model.')

        feature_names = metadata.get('feature_names', [])

        DEMO_MALIGNANT = {
            'mean radius': 17.99,
            'mean texture': 10.38,
            'mean perimeter': 122.8,
            'mean area': 1001.0,
            'mean smoothness': 0.1184,
            'mean compactness': 0.2776,
            'mean concavity': 0.3001,
            'mean concave points': 0.1471,
            'mean symmetry': 0.2419,
            'mean fractal dimension': 0.07871,
            'radius error': 1.095,
            'texture error': 0.9053,
            'perimeter error': 8.589,
            'area error': 153.4,
            'smoothness error': 0.006399,
            'compactness error': 0.04904,
            'concavity error': 0.05373,
            'concave points error': 0.01587,
            'symmetry error': 0.03003,
            'fractal dimension error': 0.006193,
            'worst radius': 25.38,
            'worst texture': 17.33,
            'worst perimeter': 184.6,
            'worst area': 2019.0,
            'worst smoothness': 0.1622,
            'worst compactness': 0.6656,
            'worst concavity': 0.7119,
            'worst concave points': 0.2654,
            'worst symmetry': 0.4601,
            'worst fractal dimension': 0.1189,
        }

        DEMO_BENIGN = {
            'mean radius': 13.54,
            'mean texture': 14.36,
            'mean perimeter': 87.46,
            'mean area': 566.3,
            'mean smoothness': 0.09779,
            'mean compactness': 0.08129,
            'mean concavity': 0.06664,
            'mean concave points': 0.04781,
            'mean symmetry': 0.1885,
            'mean fractal dimension': 0.05766,
            'radius error': 0.2699,
            'texture error': 0.7886,
            'perimeter error': 2.058,
            'area error': 23.56,
            'smoothness error': 0.008462,
            'compactness error': 0.0146,
            'concavity error': 0.02387,
            'concave points error': 0.01315,
            'symmetry error': 0.0198,
            'fractal dimension error': 0.0023,
            'worst radius': 15.11,
            'worst texture': 19.26,
            'worst perimeter': 99.7,
            'worst area': 711.2,
            'worst smoothness': 0.144,
            'worst compactness': 0.1773,
            'worst concavity': 0.239,
            'worst concave points': 0.1288,
            'worst symmetry': 0.2977,
            'worst fractal dimension': 0.07259,
        }

        dataset_path = 'breast_cancer_dataset.csv'
        feature_max_values = {}
        if os.path.exists(dataset_path):
            dataset_df = pd.read_csv(dataset_path)
            for feature in feature_names:
                if feature in dataset_df.columns:
                    values = pd.to_numeric(dataset_df[feature], errors='coerce').dropna()
                    if not values.empty:
                        feature_max_values[feature] = float(values.max())

        for feature in feature_names:
            key = f'bc_{feature.replace(" ", "_")}'
            if key not in st.session_state:
                st.session_state[key] = 0.0

        st.subheader('Example Profiles')
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            if st.button('Malignant Case', use_container_width=True):
                for feature_name, value in DEMO_MALIGNANT.items():
                    st.session_state[f'bc_{feature_name.replace(" ", "_")}'] = float(value)
                st.session_state[state_key('active_demo')] = 'Malignant'
                st.rerun()
        with demo_col2:
            if st.button('Benign Case', use_container_width=True):
                for feature_name, value in DEMO_BENIGN.items():
                    st.session_state[f'bc_{feature_name.replace(" ", "_")}'] = float(value)
                st.session_state[state_key('active_demo')] = 'Benign'
                st.rerun()
        if state_key('active_demo') in st.session_state and st.session_state[state_key('active_demo')]:
            st.info(f'Loaded preset: Demo {st.session_state[state_key("active_demo")]} case.')

        input_columns = st.columns(2)
        for index, feature in enumerate(feature_names):
            key = f'bc_{feature.replace(" ", "_")}'
            max_value = feature_max_values.get(feature, 100.0)
            max_value = max(float(max_value) * 1.2, 100.0)
            current_value = float(st.session_state.get(key, max_value * 0.5))
            if current_value > max_value:
                st.session_state[key] = max_value * 0.5
            with input_columns[index % 2]:
                st.number_input(
                    feature,
                    min_value=0.0,
                    max_value=max_value,
                    value=float(st.session_state.get(key, max_value * 0.5)),
                    step=0.01,
                    key=key,
                )

        st.markdown('---')
        if st.button('Predict Diagnosis', type='primary', use_container_width=True):
            ordered_data = {feature: float(st.session_state[f'bc_{feature.replace(" ", "_")}']) for feature in feature_names}
            df_input = pd.DataFrame([ordered_data])
            prediction, prediction_proba = predict_with_probability(pipeline, df_input)

            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown('<div class="result-title">Prediction Result</div>', unsafe_allow_html=True)
            if prediction == 0:
                st.markdown("<div class='result-tag danger'>Malignant</div><div class='result-value danger'>MALIGNANT</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-tag safe'>Benign</div><div class='result-value safe'>BENIGN</div>", unsafe_allow_html=True)
            col_default, col_no_default = st.columns(2)
            with col_default:
                st.metric('Probability of Malignant', f'{prediction_proba[0]:.2%}')
            with col_no_default:
                st.metric('Probability of Benign', f'{prediction_proba[1]:.2%}')
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('## Prediction Context')
            worst_radius = ordered_data.get('worst radius', 0.0)
            worst_area = ordered_data.get('worst area', 0.0)
            worst_concavity = ordered_data.get('worst concavity', 0.0)
            worst_concavity_points = ordered_data.get('worst concave points', 0.0)
            mean_radius = ordered_data.get('mean radius', 0.0)
            if prediction == 0:
                st.markdown(f"""
                - Largest radius: {worst_radius:.2f}
                - Largest area: {worst_area:.2f}
                - Worst concavity: {worst_concavity:.4f}
                - Worst concave points: {worst_concavity_points:.4f}
                - Mean radius: {mean_radius:.2f}
                """)
            else:
                st.markdown(f"""
                - Largest radius: {worst_radius:.2f}
                - Largest area: {worst_area:.2f}
                - Mean radius: {mean_radius:.2f}
                - The model suggests these shape and texture measurements stay within the benign range.
                """)

            render_feature_importance(pipeline, metadata)

    st.markdown(
        '''
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: #f3f6fb;
            color: #111827;
        }
        @keyframes pageFadeIn {
            from {
                opacity: 0;
                transform: translateY(6px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
            background: #f3f6fb;
            animation: pageFadeIn 0.22s ease;
        }
        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #dfe7f1;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
        .sidebar-brand {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            color: #111827;
            margin-top: 0.2rem;
        }
        .sidebar-subtitle {
            font-size: 0.92rem;
            color: #374151;
            margin-bottom: 1rem;
        }
        .sidebar-group {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #475569;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .best-model-panel {
            margin-top: 1rem;
            padding: 1rem;
            background: #ffffff;
            border: 1px solid #dbe5ef;
            border-radius: 0.75rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] {
            display: grid;
            gap: 0.35rem;
        }
        [data-testid="stSidebar"] div[role="radio"] {
            padding: 0.45rem 0.5rem;
            border-radius: 0.45rem;
        }
        [data-testid="stSidebar"] div[role="radio"]:hover {
            background: #edf5fc;
        }
        .best-model-label {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: #60738a;
        }
        .best-model-name {
            margin-top: 0.4rem;
            color: #172033;
            font-size: 1.05rem;
            font-weight: 650;
            line-height: 1.35;
        }
        .best-model-score {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            margin-top: 0.8rem;
            padding-top: 0.75rem;
            border-top: 1px solid #e5edf5;
            color: #60738a;
            font-size: 0.76rem;
        }
        .best-model-score strong { color: #172033; font-size: 0.82rem; }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] strong,
        [data-testid="stSidebar"] .sidebar-brand,
        [data-testid="stSidebar"] .sidebar-subtitle,
        [data-testid="stSidebar"] .sidebar-group {
            color: #111827 !important;
        }
        [data-testid="stHeader"] {
            background: #ffffff !important;
        }
        [data-testid="stHeader"] * {
            color: #111827 !important;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        [data-testid="stMetric"] * {
            color: #111827 !important;
        }
        .stButton > button,
        button[kind="primary"],
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            color: #111827 !important;
        }
        h1, h2, h3, h4,
        .stSubheader,
        .stMarkdown h1,
        .stMarkdown h2,
        .stMarkdown h3,
        .stMarkdown h4,
        .stTabs [role="tab"],
        .stSelectbox label,
        .stNumberInput label,
        .stSlider label,
        .stRadio label,
        .stTextInput label,
        .stDateInput label,
        .stTimeInput label,
        .stCheckbox label,
        .stMultiSelect label,
        .stExpander summary {
            color: #0f172a !important;
            letter-spacing: -0.02em;
        }
        h1 {
            font-size: 2.2rem;
            margin-bottom: 0.2rem;
        }
        h2 {
            font-size: 1.4rem;
            margin-top: 1.8rem;
            margin-bottom: 0.7rem;
        }
        .stCaption {
            color: #475569;
            font-size: 0.97rem;
        }
        [data-testid="stMain"] * {
            color: inherit;
        }
        .stat-block {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 0.6rem;
            padding: 0.7rem 0.9rem;
            min-height: 82px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
        }
        .stat-label {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin-bottom: 0.35rem;
        }
        .stat-value {
            font-size: 1rem;
            font-weight: 600;
            color: #0f172a;
            line-height: 1.35;
        }
        .process-strip {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
            margin-top: 0.6rem;
            margin-bottom: 1rem;
        }
        .process-step {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            background: #ffffff;
            border: 1px solid #dfe7ee;
            border-radius: 0.5rem;
            padding: 0.5rem 0.7rem;
            min-width: 125px;
            max-width: 185px;
        }
        .process-index {
            width: 1.5rem;
            height: 1.5rem;
            border-radius: 50%;
            background: #e2e8f0;
            color: #0f172a;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
            font-weight: 700;
            flex-shrink: 0;
        }
        .process-label {
            font-size: 0.75rem;
            color: #0f172a;
            font-weight: 600;
            line-height: 1.3;
        }
        .process-arrow {
            font-size: 0.9rem;
            color: #94a3b8;
            margin: 0 0.1rem;
        }
        .final-model-box {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            padding: 1rem 1.2rem;
            background: #ffffff;
            border: 1px solid #dfe7ee;
            border-radius: 0.75rem;
            margin: 1rem 0 1.25rem 0;
        }
        .final-model-box > div {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 0.6rem;
            padding: 0.8rem 0.9rem;
        }
        .final-model-box strong {
            font-size: 0.72rem;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #64748b;
        }
        .final-model-box div div {
            margin-top: 0.45rem;
            font-size: 1.05rem;
            font-weight: 600;
            color: #0f172a;
        }
        .result-panel {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 0.7rem;
            padding: 1.25rem 1.4rem;
            margin: 1rem 0 1.5rem 0;
        }
        .result-title {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.8rem;
        }
        .result-tag {
            display: inline-block;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
            padding: 0.35rem 0.5rem;
            border-radius: 999px;
            font-weight: 600;
        }
        .result-tag.danger {
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        .result-tag.safe {
            background: #ecfdf5;
            color: #166534;
            border: 1px solid #bbf7d0;
        }
        .result-value {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            margin-bottom: 0.7rem;
        }
        .result-value.danger { color: #991b1b; }
        .result-value.safe { color: #166534; }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 0.65rem;
            overflow: hidden;
        }
        table, th, td {
            border-color: #e2e8f0 !important;
        }
        @media (max-width: 900px) {
            .block-container {
                padding: 1.25rem 1rem 2rem;
            }
            .final-model-box {
                grid-template-columns: 1fr;
            }
            h1 { font-size: 1.75rem; }
            .stat-block { min-height: 72px; }
        }
        </style>
        ''', unsafe_allow_html=True)

    current_page = render_sidebar()

    if current_page == 'Overview':
        render_overview()
    elif current_page == 'Model Comparison':
        render_model_comparison()
    elif current_page == 'Cross-Validation':
        render_cross_validation()
    elif current_page == 'Diagnostics':
        render_diagnostics()
    elif current_page == 'Hyperparameter Tuning':
        render_tuning()
    elif current_page == 'Default Prediction':
        render_prediction()
    else:
        render_overview()


if __name__ == '__main__':
    main()
