import os
import sys
import json
import warnings
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
# PIL cropping was used previously; replaced with direct per-model saves below

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets')
os.makedirs(DATA_DIR, exist_ok=True)


def data_path(filename):
    for base_path in (DATA_DIR, PROJECT_ROOT):
        candidate = os.path.join(base_path, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(DATA_DIR, filename)


def _safe_predict_proba(estimator, X):
    """Return probability estimates for the positive class for binary classification.
    Falls back to decision_function and a sigmoid transform when predict_proba is unavailable.
    Returns an array of shape (n_samples, n_classes) matching sklearn's predict_proba output.
    """
    # Prefer predict_proba if available
    if hasattr(estimator, 'predict_proba'):
        try:
            return estimator.predict_proba(X)
        except Exception:
            pass

    # Fall back to decision_function if available
    if hasattr(estimator, 'decision_function'):
        try:
            scores = estimator.decision_function(X)
            scores = np.asarray(scores)
            if scores.ndim == 1:
                probs1 = 1.0 / (1.0 + np.exp(-scores))
                probs = np.vstack([1 - probs1, probs1]).T
                return probs
            else:
                exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs = exps / np.sum(exps, axis=1, keepdims=True)
                return probs
        except Exception:
            pass

    # Last resort: use predict and convert to one-hot-like probabilities
    preds = estimator.predict(X)
    preds = np.asarray(preds)
    unique = np.unique(preds)
    n_classes = int(unique.max()) + 1 if preds.size else 2
    probs = np.zeros((len(preds), n_classes))
    for i, p in enumerate(preds):
        probs[i, int(p)] = 1.0
    return probs

def main():
    print("================ STEP 1: Initialization ================")
    os.makedirs('plots', exist_ok=True)
    print("Output directory 'plots' created or already exists.\n")

    print("================ STEP 2: Load Dataset ================")
    file_path = data_path('UCI_Credit_Card.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Dataset not found at {file_path}. Please ensure it is in the datasets directory.")

    df = pd.read_csv(file_path)
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn Names:")
    print(df.columns.tolist())
    print()

    print("================ STEP 3: Dataset Information ================")
    print("Data Types:")
    print(df.dtypes)
    print("\nDataFrame Info:")
    df.info()
    print("\nSummary Statistics:")
    print(df.describe())
    print()

    print("================ STEP 4: Verify Target ================")
    target_column = 'default.payment.next.month'
    if target_column not in df.columns:
        raise ValueError(f"Error: Target column '{target_column}' not found in the dataset.")
    
    class_counts = df[target_column].value_counts()
    class_percentages = df[target_column].value_counts(normalize=True) * 100
    
    print("Target Column:", target_column)
    print(f"Class Counts:\n{class_counts}")
    print(f"\nClass Percentages:\n{class_percentages}")
    print("\nNOTE: This dataset is IMBALANCED (~22% default).")
    print()

    print("================ STEP 4.1: Stratified Reduction to 10,000 Rows ================")
    if len(df) > 10000:
        print(f"Original dataset rows: {len(df)}. Performing stratified reduction to 10,000 rows...")
        df, _ = train_test_split(df, train_size=10000, random_state=42, stratify=df[target_column])
        df = df.reset_index(drop=True)
        print(f"Reduced dataset shape: {df.shape}")
        print(f"Stratified class distribution:\n{df[target_column].value_counts()}")
        print(f"Class percentages:\n{df[target_column].value_counts(normalize=True) * 100}\n")

    print("================ STEP 5: Export Before Preprocessing ================")
    df.to_excel(os.path.join(DATA_DIR, 'credit_card_before_preprocessing.xlsx'), index=False, engine='openpyxl')
    df.to_csv(os.path.join(DATA_DIR, 'credit_card_before_preprocessing.csv'), index=False)
    df.to_csv(os.path.join(DATA_DIR, 'credit_card_10k_subset.csv'), index=False)
    print("Saved original data to 'credit_card_before_preprocessing.xlsx', 'credit_card_before_preprocessing.csv', and 'credit_card_10k_subset.csv'.\n")

    print("================ STEP 6: Data Cleaning ================")
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        print("Dropped column: ID (identifier, not a feature)")
    print("Current Data Types Check:")
    print(df.dtypes.value_counts())
    print()

    print("================ STEP 7: Missing Values ================")
    missing_counts = df.isnull().sum()
    print("Missing value counts per column:")
    print(missing_counts[missing_counts > 0])
    if missing_counts.sum() > 0:
        print("Imputing missing values: numeric -> median, categorical -> mode...")
        # Numeric median
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        # Categorical mode (object / category types)
        obj_cols = df.select_dtypes(include=['object', 'category']).columns
        for c in obj_cols:
            if df[c].isnull().any():
                try:
                    df[c] = df[c].fillna(df[c].mode().iloc[0])
                except Exception:
                    df[c] = df[c].fillna('missing')
    else:
        print("No missing values found.")
    print()

    print("================ STEP 8: Duplicate Records ================")
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate count: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"Removed duplicates. New shape: {df.shape}")
    print()

    print("================ STEP 9: EDA ================")
    # 1. Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df, palette='Set2')
    plt.title('Credit Card Default Class Distribution')
    plt.xlabel('Default Payment Next Month (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('plots/credit_card_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Correlation heatmap (numeric columns only)
    plt.figure(figsize=(14, 10))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/credit_card_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Feature distributions
    dist_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(dist_features, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('plots/credit_card_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Boxplots by Class
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(dist_features, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(y=df[feature], x=df[target_column], palette='viridis')
        plt.title(f'Boxplot of {feature} by Class')
    plt.tight_layout()
    plt.savefig('plots/credit_card_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("EDA plots saved to 'plots/' directory.\n")

    print("================ STEP 10: Separate X and y ================")
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    feature_names = X.columns.tolist()
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature Names: {feature_names}\n")

    print("================ STEP 11: Train/Test Split ================")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Training shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing shape: X={X_test.shape}, y={y_test.shape}\n")

    print("================ STEPS 12-13: Build and Train Pipelines ================")
    pipelines = {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=42, max_iter=1000))]),
        'kNN': Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier())]),
        'SVM': Pipeline([('scaler', StandardScaler()), ('classifier', SVC(random_state=42, probability=True))]),
        'Random Forest': Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(random_state=42))])
    }

    print("Training 4 ML models on training set...")
    fitted_models = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        print(f"Trained {name} successfully.")
    print()

    print("================ STEP 14: Test Set Metrics ================")
    test_results = []
    for name, pipeline in fitted_models.items():
        y_pred = pipeline.predict(X_test)
        y_proba = _safe_predict_proba(pipeline, X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_proba)
        
        test_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc_score
        })
    
    test_results_df = pd.DataFrame(test_results).set_index('Model')
    print("Test Set Metrics:")
    print(test_results_df.to_string())
    pd.DataFrame(test_results).to_csv(os.path.join(DATA_DIR, 'credit_card_test_results.csv'), index=False)
    print("Saved test set results to 'credit_card_test_results.csv'.\n")

    print("================ STEP 15: Classification Reports ================")
    for name, pipeline in fitted_models.items():
        y_pred = pipeline.predict(X_test)
        print(f"--- Classification Report for {name} ---")
        print(classification_report(y_test, y_pred, zero_division=0))
    print()

    print("================ STEP 16: 5-Fold Cross-Validation ================")
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_results_list = []
    print("Running Stratified 5-Fold Cross-Validation across all 4 models...")
    for name, pipeline in pipelines.items():
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv_kfold, scoring=scoring, n_jobs=-1)
        cv_results_list.append({
            'Model': name,
            'CV Accuracy': cv_scores['test_accuracy'].mean(),
            'CV Precision': cv_scores['test_precision'].mean(),
            'CV Recall': cv_scores['test_recall'].mean(),
            'CV F1': cv_scores['test_f1'].mean(),
            'CV ROC-AUC': cv_scores['test_roc_auc'].mean()
        })
        print(f"5-Fold CV completed for {name}")
    
    cv_results_df = pd.DataFrame(cv_results_list)
    cv_results_df.to_csv(os.path.join(DATA_DIR, 'credit_card_cv_results.csv'), index=False)
    print("\nCross-Validation Results:")
    print(cv_results_df.to_string())
    print()

    print("================ STEP 17: Confusion Matrices ================")
    # Save combined 2x2 grid for overview
    plt.figure(figsize=(10, 8))
    for i, (name, pipeline) in enumerate(fitted_models.items(), 1):
        plt.subplot(2, 2, i)
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('plots/credit_card_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved combined confusion matrices plot.")

    # Also save individual confusion matrices (more robust than image-cropping)
    for name, pipeline in fitted_models.items():
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        key = name.lower().replace(' ', '_').replace('-', '_')
        fig.savefig(f'plots/credit_card_cm_{key}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    print("Saved individual confusion matrix images to 'plots/'.\n")

    print("================ STEP 18: ROC Curves ================")
    plt.figure(figsize=(8, 6))
    for name, pipeline in fitted_models.items():
        y_proba = _safe_predict_proba(pipeline, X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/credit_card_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved ROC curves plot.\n")

    print("================ STEP 19: Model Comparison Graph ================")
    df_melted = test_results_df.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=df_melted)
    plt.title('Model Performance Comparison (Test Set)')
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/credit_card_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved model comparison graph.\n")

    print("================ STEPS 20-22: Hyperparameter Tuning ================")
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.01, 0.1, 1, 10], 
            'classifier__solver': ['liblinear', 'lbfgs'],
            'classifier__class_weight': [None, 'balanced']
        },
        'kNN': {
            'classifier__n_neighbors': [3, 5, 7], 
            'classifier__weights': ['uniform', 'distance']
        },
        'SVM': {
            'classifier__C': [0.1, 1], 
            'classifier__kernel': ['rbf'], 
            'classifier__gamma': ['scale']
        },
        'Random Forest': {
            'classifier__n_estimators': [100, 200], 
            'classifier__max_depth': [10, 20], 
            'classifier__min_samples_split': [2, 5],
            'classifier__class_weight': [None, 'balanced']
        }
    }

    tuned_models = {}
    best_params_dict = {}
    grid_objects = {}
    for name in pipelines.keys():
        import time
        start_time = time.time()
        print(f"Tuning {name} with GridSearchCV (5-Fold CV)...")
        
        grid_search = GridSearchCV(
            pipelines[name], 
            param_grids[name], 
            cv=cv_kfold, 
            scoring='roc_auc', 
            n_jobs=-1
        )
        grid_objects[name] = grid_search
        grid_search.fit(X_train, y_train)
        # Save cv_results_ for transparency
        try:
            cvres = pd.DataFrame(grid_search.cv_results_)
            cvres.to_csv(os.path.join(DATA_DIR, f'credit_card_gridsearch_{name.replace(" ","_")}_cv_results.csv'), index=False)
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        tuned_models[name] = grid_search.best_estimator_
        best_params_dict[name] = grid_search.best_params_
        
        print(f"Best params for {name}: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"Time taken: {elapsed:.2f} seconds\n")

    print("================ STEP 23: Evaluate Tuned Models ================")
    tuned_test_results = []
    tuned_cv_roc = {}

    for name, pipeline in tuned_models.items():
        y_pred = pipeline.predict(X_test)
        y_proba = _safe_predict_proba(pipeline, X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_proba)
        
        mean_cv_roc = grid_objects[name].best_score_
        tuned_cv_roc[name] = mean_cv_roc
        
        tuned_test_results.append({
            'Model': name,
            'Test Accuracy': acc,
            'Test Precision': prec,
            'Test Recall': rec,
            'Test F1': f1,
            'Test ROC-AUC': auc_score,
            'CV ROC-AUC': mean_cv_roc
        })

    tuned_results_df = pd.DataFrame(tuned_test_results).set_index('Model')
    print("Tuned Models Evaluation:")
    print(tuned_results_df.to_string())
    print()

    # Save hyperparameter tuning results summary table
    tuning_summary = []
    for name in pipelines.keys():
        params_str = ", ".join([f"{k.split('__')[-1]}={v}" for k, v in best_params_dict[name].items()])
        tuning_summary.append({
            'Model': name,
            'Best Parameters': params_str,
            'Mean CV ROC-AUC': float(tuned_cv_roc[name])
        })
    pd.DataFrame(tuning_summary).to_csv(os.path.join(DATA_DIR, 'credit_card_tuning_results.csv'), index=False)
    print("Saved tuning summary to 'credit_card_tuning_results.csv'.\n")

    print("================ STEP 24: Select Final Model ================")
    # Select based on highest GridSearchCV best CV ROC-AUC
    best_model_name = max(
        grid_objects,
        key=lambda name: grid_objects[name].best_score_
    )
    best_pipeline = tuned_models[best_model_name]
    best_metrics = tuned_results_df.loc[best_model_name]
    
    print(f"Final selected model name: {best_model_name}")
    print(f"Metrics for {best_model_name}:")
    print(best_metrics)
    print("\nExplanation: ROC-AUC was used as the primary selection criterion because it evaluates ranking performance across all classification thresholds, making it more robust than accuracy alone.")
    print("This is especially important for the Credit Card dataset where class imbalance (~22% default) makes accuracy a misleading metric.")
    print("Disclaimer: This model achieved the highest score under this evaluation setup. Results may vary with different data or configurations.\n")

    print("================ STEP 25: Save Model ================")
    joblib.dump(best_pipeline, 'credit_card_best_model.pkl')
    print("Saved best model to 'credit_card_best_model.pkl'.\n")

    print("================ STEP 26: Save Metadata ================")
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        return obj

    metadata = {
        'dataset_name': 'UCI Credit Card Default Dataset',
        'dataset_size': len(df),
        'target_column': target_column,
        'target_labels': {'0': 'No Default', '1': 'Default'},
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'selected_model': best_model_name,
        'test_metrics': make_serializable(best_metrics.to_dict()),
        'cv_metrics': {'roc_auc': float(tuned_cv_roc[best_model_name])},
        'best_hyperparameters': make_serializable(best_params_dict[best_model_name]),
        'class_imbalance_note': 'This is especially important for the Credit Card dataset where class imbalance (~78% vs ~22%) makes accuracy a misleading metric.',
        'training_date': datetime.datetime.now().isoformat(),
        'sampling_note': 'Stratified sample of 10,000 records used for full model evaluation, 5-fold cross-validation, and GridSearchCV hyperparameter tuning.'
    }
    
    with open('credit_card_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Saved metadata to 'credit_card_metadata.json'.\n")

    print("================ STEP 27: After-Preprocessing Export ================")
    df.to_excel(os.path.join(DATA_DIR, 'credit_card_after_preprocessing.xlsx'), index=False, engine='openpyxl')
    df.to_csv(os.path.join(DATA_DIR, 'credit_card_after_preprocessing.csv'), index=False)
    df.to_csv(os.path.join(DATA_DIR, 'credit_card_dataset.csv'), index=False)
    print("Saved cleaned data to 'credit_card_after_preprocessing.xlsx', 'credit_card_after_preprocessing.csv', and 'credit_card_dataset.csv'.\n")


if __name__ == "__main__":
    main()
