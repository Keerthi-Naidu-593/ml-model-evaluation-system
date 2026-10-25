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

APP_PREFIX = "credit_card"
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
    # The router in home.py owns the page configuration. Calling this again
    # after navigation causes Streamlit's "set_page_config must be first"
    # exception. Keep this configuration only for direct execution.
    if home_page is None:
        st.set_page_config(
            page_title="Model Evaluation & Comparison",
            page_icon=None,
            layout="wide",
            initial_sidebar_state="expanded",
        )

    @st.cache_resource
    def load_model_and_metadata():
        model_path = 'credit_card_best_model.pkl'
        metadata_path = 'credit_card_metadata.json'
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
        test_path = resolve_data_path('credit_card_test_results.csv')
        cv_path = resolve_data_path('credit_card_cv_results.csv')
        tuning_path = resolve_data_path('credit_card_tuning_results.csv')

        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
        cv_df = pd.read_csv(cv_path) if os.path.exists(cv_path) else None
        tuning_df = pd.read_csv(tuning_path) if os.path.exists(tuning_path) else None
        return test_df, cv_df, tuning_df


    pipeline, metadata = load_model_and_metadata()
    test_path = resolve_data_path('credit_card_test_results.csv')
    cv_path = resolve_data_path('credit_card_cv_results.csv')
    tuning_path = resolve_data_path('credit_card_tuning_results.csv')
    test_results_df, cv_results_df, tuning_results_df = load_evaluation_data(
        _file_mtime(test_path),
        _file_mtime(cv_path),
        _file_mtime(tuning_path),
    )

    if pipeline is None or metadata is None:
        st.error('Error: Required model (credit_card_best_model.pkl) or metadata (credit_card_metadata.json) not found.')
        st.stop()


    def predict_with_probability(model_pipeline, df_input):
        prediction = model_pipeline.predict(df_input)[0]
        try:
            prediction_proba = model_pipeline.predict_proba(df_input)[0]
        except Exception:
            try:
                from credit_card_analysis import _safe_predict_proba
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
        st.sidebar.markdown("<div class='sidebar-brand'>MODEL EVALUATION</div><div class='sidebar-subtitle'>Credit Card Default</div>", unsafe_allow_html=True)

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
        st.caption('Comparative evaluation of machine learning models for credit card default prediction.')

        feature_count = metadata.get('num_features', 23)
        model_count = 4
        validation_text = 'Stratified 5-Fold CV'

        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.markdown("<div class='stat-block'><div class='stat-label'>Dataset</div><div class='stat-value'>UCI Credit Card Default</div></div>", unsafe_allow_html=True)
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
        performance_plot = os.path.join('plots', 'credit_card_model_comparison.png')
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
        selected_model = metadata.get('selected_model', 'Random Forest')
        cv_roc = metadata.get('cv_metrics', {}).get('roc_auc', 0.0)
        st.markdown(
            f"<div class='final-model-box'><div><strong>Selected Model</strong><div>{selected_model}</div></div><div><strong>Selection Criterion</strong><div>Highest Mean 5-Fold CV ROC-AUC</div></div><div><strong>Mean CV ROC-AUC</strong><div>{float(cv_roc):.4f}</div></div></div>",
            unsafe_allow_html=True,
        )

        metric_cards = st.columns(6)
        test_metrics = metadata.get('test_metrics', {})
        metric_order = [
            ('Accuracy', 'Test Accuracy', 'Accuracy'),
            ('Precision', 'Test Precision', 'Precision'),
            ('Recall', 'Test Recall', 'Recall'),
            ('F1-Score', 'Test F1', 'F1'),
            ('Test ROC-AUC', 'Test ROC-AUC', 'ROC-AUC'),
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

            st.subheader('Metric Comparison')
            metric_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
            if 'ROC-AUC' in test_results_df.columns:
                metric_cols.append('ROC-AUC')
            metric_df = test_results_df[metric_cols].copy()
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
                sns.barplot(data=metric_long, x='Metric', y='Score', hue='Model', palette=['#4e79a7', '#f4b942', '#59b0a9', '#e76f51'], ax=ax)
                ax.set_title('Accuracy, Precision, Recall, F1-Score, and ROC-AUC by Model')
                ax.set_xlabel('Metric')
                ax.set_ylabel('Score')
                ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc='upper left')
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.4f', padding=3)
                st.pyplot(fig)

            st.subheader('ROC-AUC Comparison')
            roc_df = test_results_df[['Model', 'ROC-AUC']].sort_values('ROC-AUC', ascending=False).copy()
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
                sns.barplot(data=roc_df, x='ROC-AUC', y='Model', palette=['#4e79a7', '#f4b942', '#59b0a9', '#e76f51'], ax=ax, orient='h')
                ax.set_title('Test-set ROC-AUC by Model')
                ax.set_xlabel('ROC-AUC')
                ax.set_ylabel('Model')
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.4f', padding=3)
                st.pyplot(fig)
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
            roc_path = os.path.join('plots', 'credit_card_roc_curves.png')
            if os.path.exists(roc_path):
                render_plot(roc_path)
            st.caption('ROC curves illustrate the trade-off between true-positive and false-positive rates across classification thresholds.')

        with tabs[1]:
            cm_tabs = st.tabs(['Logistic Regression', 'kNN', 'SVM', 'Random Forest', 'Combined View'])
            with cm_tabs[0]:
                p = os.path.join('plots', 'credit_card_cm_logistic_regression.png')
                if os.path.exists(p):
                    render_plot(p)
            with cm_tabs[1]:
                p = os.path.join('plots', 'credit_card_cm_knn.png')
                if os.path.exists(p):
                    render_plot(p)
            with cm_tabs[2]:
                p = os.path.join('plots', 'credit_card_cm_svm.png')
                if os.path.exists(p):
                    render_plot(p)
            with cm_tabs[3]:
                p = os.path.join('plots', 'credit_card_cm_random_forest.png')
                if os.path.exists(p):
                    render_plot(p)
            with cm_tabs[4]:
                p = os.path.join('plots', 'credit_card_confusion_matrices.png')
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
        feature_names = model_metadata['feature_names']

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
        st.title('Credit Card Default Prediction')
        st.caption('Estimate the probability of next-month payment default using the selected machine-learning model.')

        DEMO_DEFAULT = {
            'LIMIT_BAL': 20000,
            'SEX': '2 - Female',
            'EDUCATION': '2 - University',
            'MARRIAGE': '1 - Married',
            'AGE': 24,
            'PAY_0': '2 - Payment delay 2 months',
            'PAY_2': '2 - Payment delay 2 months',
            'PAY_3': '-1 - Paid in full',
            'PAY_4': '-1 - Paid in full',
            'PAY_5': '-2 - No consumption',
            'PAY_6': '-2 - No consumption',
            'BILL_AMT1': 3913,
            'BILL_AMT2': 3102,
            'BILL_AMT3': 689,
            'BILL_AMT4': 0,
            'BILL_AMT5': 0,
            'BILL_AMT6': 0,
            'PAY_AMT1': 0,
            'PAY_AMT2': 689,
            'PAY_AMT3': 0,
            'PAY_AMT4': 0,
            'PAY_AMT5': 0,
            'PAY_AMT6': 0,
        }
        DEMO_NO_DEFAULT = {
            'LIMIT_BAL': 90000,
            'SEX': '2 - Female',
            'EDUCATION': '2 - University',
            'MARRIAGE': '2 - Single',
            'AGE': 34,
            'PAY_0': '0 - Revolving credit',
            'PAY_2': '0 - Revolving credit',
            'PAY_3': '0 - Revolving credit',
            'PAY_4': '0 - Revolving credit',
            'PAY_5': '0 - Revolving credit',
            'PAY_6': '0 - Revolving credit',
            'BILL_AMT1': 29239,
            'BILL_AMT2': 14027,
            'BILL_AMT3': 13559,
            'BILL_AMT4': 14331,
            'BILL_AMT5': 14948,
            'BILL_AMT6': 15549,
            'PAY_AMT1': 1518,
            'PAY_AMT2': 1500,
            'PAY_AMT3': 1000,
            'PAY_AMT4': 1000,
            'PAY_AMT5': 1000,
            'PAY_AMT6': 5000,
        }
        DEFAULT_STATE = {
            'LIMIT_BAL': 200000,
            'SEX': '1 - Male',
            'EDUCATION': '2 - University',
            'MARRIAGE': '1 - Married',
            'AGE': 35,
            'PAY_0': '0 - Revolving credit',
            'PAY_2': '0 - Revolving credit',
            'PAY_3': '0 - Revolving credit',
            'PAY_4': '0 - Revolving credit',
            'PAY_5': '0 - Revolving credit',
            'PAY_6': '0 - Revolving credit',
            'BILL_AMT1': 0,
            'BILL_AMT2': 0,
            'BILL_AMT3': 0,
            'BILL_AMT4': 0,
            'BILL_AMT5': 0,
            'BILL_AMT6': 0,
            'PAY_AMT1': 0,
            'PAY_AMT2': 0,
            'PAY_AMT3': 0,
            'PAY_AMT4': 0,
            'PAY_AMT5': 0,
            'PAY_AMT6': 0,
        }

        for key, value in DEFAULT_STATE.items():
            session_key = f'cc_{key}'
            if session_key not in st.session_state:
                st.session_state[session_key] = value

        st.subheader('Example Profiles')
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            if st.button('Default Case', use_container_width=True):
                for key, val in DEMO_DEFAULT.items():
                    st.session_state[f'cc_{key}'] = val
                st.session_state['cc_active_demo'] = 'Default Demo'
                st.session_state[state_key('current_page')] = 'Default Prediction'
                st.session_state[state_key('nav_override')] = 'Default Prediction'
                st.rerun()
        with demo_col2:
            if st.button('No Default Case', use_container_width=True):
                for key, val in DEMO_NO_DEFAULT.items():
                    st.session_state[f'cc_{key}'] = val
                st.session_state['cc_active_demo'] = 'No-Default Demo'
                st.session_state[state_key('current_page')] = 'Default Prediction'
                st.session_state[state_key('nav_override')] = 'Default Prediction'
                st.rerun()
        if 'cc_active_demo' in st.session_state and st.session_state['cc_active_demo']:
            st.info('Example profile loaded. You can modify any input before prediction.')

        sex_options = ['1 - Male', '2 - Female']
        education_options = ['1 - Graduate School', '2 - University', '3 - High School', '4 - Others']
        marriage_options = ['1 - Married', '2 - Single', '3 - Others']
        pay_options = ['-2 - No consumption', '-1 - Paid in full', '0 - Revolving credit', '1 - Payment delay 1 month', '2 - Payment delay 2 months', '3 - Payment delay 3 months', '4 - Payment delay 4 months', '5 - Payment delay 5 months', '6 - Payment delay 6 months', '7 - Payment delay 7 months', '8 - Payment delay 8 months']

        st.subheader('Client Profile')
        personal_col, repayment_col = st.columns(2)
        with personal_col:
            st.markdown('### Personal Information')
            limit_bal = st.number_input('Credit Limit', min_value=10000, max_value=1000000, step=10000, key='cc_LIMIT_BAL')
            age = st.slider('Age', min_value=20, max_value=80, key='cc_AGE')
            sex_input = st.selectbox('Sex', sex_options, key='cc_SEX')
            education_input = st.selectbox('Education', education_options, key='cc_EDUCATION')
            marriage_input = st.selectbox('Marital Status', marriage_options, key='cc_MARRIAGE')
        with repayment_col:
            st.markdown('### Repayment History')
            pay_0_input = st.selectbox('PAY_0', pay_options, key='cc_PAY_0')
            pay_2_input = st.selectbox('PAY_2', pay_options, key='cc_PAY_2')
            pay_3_input = st.selectbox('PAY_3', pay_options, key='cc_PAY_3')
            pay_4_input = st.selectbox('PAY_4', pay_options, key='cc_PAY_4')
            pay_5_input = st.selectbox('PAY_5', pay_options, key='cc_PAY_5')
            pay_6_input = st.selectbox('PAY_6', pay_options, key='cc_PAY_6')

        st.subheader('Financial History')
        bill_col, payment_col = st.columns(2)
        with bill_col:
            st.markdown('### Bill Amounts')
            bill_amt1 = st.number_input('BILL_AMT1', min_value=-500000, max_value=1000000, step=1000, key='cc_BILL_AMT1')
            bill_amt2 = st.number_input('BILL_AMT2', min_value=-500000, max_value=1000000, step=1000, key='cc_BILL_AMT2')
            bill_amt3 = st.number_input('BILL_AMT3', min_value=-500000, max_value=1000000, step=1000, key='cc_BILL_AMT3')
            bill_amt4 = st.number_input('BILL_AMT4', min_value=-500000, max_value=1000000, step=1000, key='cc_BILL_AMT4')
            bill_amt5 = st.number_input('BILL_AMT5', min_value=-500000, max_value=1000000, step=1000, key='cc_BILL_AMT5')
            bill_amt6 = st.number_input('BILL_AMT6', min_value=-500000, max_value=1000000, step=1000, key='cc_BILL_AMT6')
        with payment_col:
            st.markdown('### Previous Payments')
            pay_amt1 = st.number_input('PAY_AMT1', min_value=0, max_value=1000000, step=1000, key='cc_PAY_AMT1')
            pay_amt2 = st.number_input('PAY_AMT2', min_value=0, max_value=1000000, step=1000, key='cc_PAY_AMT2')
            pay_amt3 = st.number_input('PAY_AMT3', min_value=0, max_value=1000000, step=1000, key='cc_PAY_AMT3')
            pay_amt4 = st.number_input('PAY_AMT4', min_value=0, max_value=1000000, step=1000, key='cc_PAY_AMT4')
            pay_amt5 = st.number_input('PAY_AMT5', min_value=0, max_value=1000000, step=1000, key='cc_PAY_AMT5')
            pay_amt6 = st.number_input('PAY_AMT6', min_value=0, max_value=1000000, step=1000, key='cc_PAY_AMT6')

        sex = int(sex_input.split(' - ')[0])
        education = int(education_input.split(' - ')[0])
        marriage = int(marriage_input.split(' - ')[0])
        pay_0 = int(pay_0_input.split(' - ')[0])
        pay_2 = int(pay_2_input.split(' - ')[0])
        pay_3 = int(pay_3_input.split(' - ')[0])
        pay_4 = int(pay_4_input.split(' - ')[0])
        pay_5 = int(pay_5_input.split(' - ')[0])
        pay_6 = int(pay_6_input.split(' - ')[0])

        st.markdown('---')
        if st.button('Run Prediction', type='primary', use_container_width=True):
            st.session_state[state_key('current_page')] = 'Default Prediction'
            st.session_state[state_key('nav_override')] = 'Default Prediction'
            input_data = {
                'LIMIT_BAL': float(limit_bal),
                'SEX': int(sex),
                'EDUCATION': int(education),
                'MARRIAGE': int(marriage),
                'AGE': int(age),
                'PAY_0': int(pay_0),
                'PAY_2': int(pay_2),
                'PAY_3': int(pay_3),
                'PAY_4': int(pay_4),
                'PAY_5': int(pay_5),
                'PAY_6': int(pay_6),
                'BILL_AMT1': float(bill_amt1),
                'BILL_AMT2': float(bill_amt2),
                'BILL_AMT3': float(bill_amt3),
                'BILL_AMT4': float(bill_amt4),
                'BILL_AMT5': float(bill_amt5),
                'BILL_AMT6': float(bill_amt6),
                'PAY_AMT1': float(pay_amt1),
                'PAY_AMT2': float(pay_amt2),
                'PAY_AMT3': float(pay_amt3),
                'PAY_AMT4': float(pay_amt4),
                'PAY_AMT5': float(pay_amt5),
                'PAY_AMT6': float(pay_amt6),
            }

            expected = list(metadata.get('feature_names', []))
            if not expected:
                st.error('Metadata missing feature names. Cannot proceed.')
                st.stop()

            missing = [f for f in expected if f not in input_data]
            if missing:
                st.error(f'Input data missing features required by the model: {missing}')
                st.stop()

            ordered_data = {feature: input_data[feature] for feature in expected}
            df_input = pd.DataFrame([ordered_data])
            prediction, prediction_proba = predict_with_probability(pipeline, df_input)

            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown('<div class="result-title">Prediction Result</div>', unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("<div class='result-tag danger'>Risk Classification</div><div class='result-value danger'>HIGH RISK</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-tag safe'>Risk Classification</div><div class='result-value safe'>LOW RISK</div>", unsafe_allow_html=True)
            col_default, col_no_default = st.columns(2)
            with col_default:
                st.metric('Probability of Default', f'{prediction_proba[1]:.2%}')
            with col_no_default:
                st.metric('Probability of No Default', f'{prediction_proba[0]:.2%}')
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('## Prediction Context')
            latest_status = pay_0_input
            max_delay = max(pay_0, pay_2, pay_3, pay_4, pay_5, pay_6)
            st.markdown(f"""
            - Latest repayment status: {latest_status}
            - Maximum recorded repayment delay: {max_delay} month(s)
            - Latest bill amount: ${bill_amt1:,.0f}
            - Latest payment amount: ${pay_amt1:,.0f}
            - Credit limit: ${limit_bal:,.0f}
            """)

            st.markdown('---')
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
        .stButton > button {
            border-radius: 0.5rem;
            border: 1px solid #dbe6f2;
            background: #ffffff;
            color: #1f2937;
            font-weight: 600;
            transition: background-color 0.12s ease, border-color 0.12s ease, transform 0.12s ease;
        }
        .stButton > button:hover {
            border-color: #b7c8d9;
            background: #f8fbff;
        }
        button[kind="primary"] {
            background: #dfeaf8;
            color: #1f2937;
            border-color: #cfe1f7;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.1rem;
            font-weight: 600;
        }
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
