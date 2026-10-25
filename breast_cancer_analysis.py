import os
import warnings
import datetime
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

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
    if hasattr(estimator, 'predict_proba'):
        try:
            return estimator.predict_proba(X)
        except Exception:
            pass

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

    preds = estimator.predict(X)
    preds = np.asarray(preds)
    unique = np.unique(preds)
    n_classes = int(unique.max()) + 1 if preds.size else 2
    probs = np.zeros((len(preds), n_classes))
    for i, p in enumerate(preds):
        probs[i, int(p)] = 1.0
    return probs

print("STEP 2 — Load Dataset")
try:
    dataset_path = data_path('breast_cancer_dataset.csv')
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    raise FileNotFoundError("Dataset file 'breast_cancer_dataset.csv' not found.")

print("Shape:", df.shape)
print("First 5 rows:\n", df.head())
print("Column names:", df.columns.tolist())

print("\nSTEP 3 — Dataset Information")
df.info()
print("Data Types:\n", df.dtypes)
print("Describe:\n", df.describe())

print("\nSTEP 4 — Verify Target")
if 'target' not in df.columns:
    raise ValueError("Target column 'target' not found in the dataset.")
print("Unique values:", df['target'].unique())
print("Class counts:\n", df['target'].value_counts())
print("Class percentages:\n", df['target'].value_counts(normalize=True) * 100)

print("\nSTEP 5 — Export Before Preprocessing")
df.to_excel(os.path.join(DATA_DIR, 'breast_cancer_before_preprocessing.xlsx'), index=False, engine='openpyxl')

print("\nSTEP 6 — Data Cleaning")
print("Data Types Summary:\n", df.dtypes.value_counts())
# No ID column to drop for this dataset
# No unnecessary modifications

print("\nSTEP 7 — Missing Values")
missing_counts = df.isnull().sum()
print("Missing values per column:\n", missing_counts)
if missing_counts.sum() > 0:
    print("Imputing missing values: numeric -> median, categorical -> mode...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    obj_cols = df.select_dtypes(include=['object', 'category']).columns
    for c in obj_cols:
        if df[c].isnull().any():
            try:
                df[c] = df[c].fillna(df[c].mode().iloc[0])
            except Exception:
                df[c] = df[c].fillna('missing')
else:
    print('No missing values found')

print("\nSTEP 8 — Duplicate Records")
duplicates = df.duplicated().sum()
print("Duplicate count:", duplicates)
if duplicates > 0:
    df = df.drop_duplicates()
    print("New shape:", df.shape)

print("\nSTEP 9 — EDA")
# 1. Class distribution bar chart
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title('Class Distribution')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'breast_cancer_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# 2. Feature distribution histograms
key_features = df.drop('target', axis=1).columns[:10]
fig, axes = plt.subplots(5, 2, figsize=(10, 15))
axes = axes.flatten()
for i, col in enumerate(key_features):
    sns.histplot(df[col], ax=axes[i], kde=True)
    axes[i].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'breast_cancer_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# 3. Correlation heatmap (numeric columns only)
plt.figure(figsize=(16, 12))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'breast_cancer_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4. Boxplots for key features
fig, axes = plt.subplots(5, 2, figsize=(10, 15))
axes = axes.flatten()
for i, col in enumerate(key_features):
    sns.boxplot(x='target', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'breast_cancer_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nSTEP 10 — Separate X and y")
X = df.drop('target', axis=1)
y = df['target']
feature_names = X.columns.tolist()
print("Feature count:", len(feature_names))

print("\nSTEP 11 — Train/Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print("Training shape:", X_train.shape, y_train.shape)
print("Testing shape:", X_test.shape, y_test.shape)

print("\nSTEPS 12-13 — Build and Train Pipelines")
pipelines = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=42, max_iter=1000))]),
    'kNN': Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier())]),
    'SVM': Pipeline([('scaler', StandardScaler()), ('classifier', SVC(random_state=42, probability=True))]),
    'Random Forest': Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(random_state=42))])
}

for name, p in pipelines.items():
    p.fit(X_train, y_train)
    print(f"Trained {name} successfully.")

print("\nSTEP 14 — Test Set Metrics")
metrics_list = []
for name, p in pipelines.items():
    y_pred = p.predict(X_test)
    proba = _safe_predict_proba(p, X_test)
    # Use class-0 (Malignant) probability column — robust to class ordering
    class_index = list(p.classes_).index(0)
    y_proba_malignant = proba[:, class_index]
    y_malignant = (np.asarray(y_test) == 0).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec  = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1   = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    roc  = roc_auc_score(y_malignant, y_proba_malignant)
    metrics_list.append({
        'Model': name,
        'Accuracy': acc,
        'Malignant Precision': prec,
        'Malignant Recall': rec,
        'Malignant F1': f1,
        'Malignant ROC-AUC': roc
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)
metrics_df.to_csv(os.path.join(DATA_DIR, 'breast_cancer_test_results.csv'), index=False)
print("Saved test results to 'breast_cancer_test_results.csv'.")

print("\nSTEP 15 — Classification Reports")
for name, p in pipelines.items():
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, p.predict(X_test), zero_division=0))

print("\nSTEP 16 — 5-Fold Cross-Validation")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Custom Malignant-positive scorers (class 0 = Malignant)
malignant_precision_scorer = make_scorer(precision_score, pos_label=0, zero_division=0)
malignant_recall_scorer    = make_scorer(recall_score,    pos_label=0, zero_division=0)
malignant_f1_scorer        = make_scorer(f1_score,        pos_label=0, zero_division=0)

def malignant_roc_auc(estimator, X, y):
    y_malignant = (np.asarray(y) == 0).astype(int)
    proba = _safe_predict_proba(estimator, X)
    class_index = list(estimator.classes_).index(0)
    return roc_auc_score(y_malignant, proba[:, class_index])

scoring = {
    'accuracy':             'accuracy',
    'malignant_precision':  malignant_precision_scorer,
    'malignant_recall':     malignant_recall_scorer,
    'malignant_f1':         malignant_f1_scorer,
    'malignant_roc_auc':    malignant_roc_auc
}

cv_results_list = []
for name, p in pipelines.items():
    scores = cross_validate(p, X_train, y_train, cv=cv, scoring=scoring)
    cv_results_list.append({
        'Model':                   name,
        'CV Accuracy':             scores['test_accuracy'].mean(),
        'CV Malignant Precision':  scores['test_malignant_precision'].mean(),
        'CV Malignant Recall':     scores['test_malignant_recall'].mean(),
        'CV Malignant F1':         scores['test_malignant_f1'].mean(),
        'CV Malignant ROC-AUC':    scores['test_malignant_roc_auc'].mean()
    })

cv_results = pd.DataFrame(cv_results_list)
cv_results.to_csv(os.path.join(DATA_DIR, 'breast_cancer_cv_results.csv'), index=False)
print(cv_results)

print("\nSTEP 17 — Confusion Matrices")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
cm_keys = ['logistic_regression', 'knn', 'svm', 'random_forest']
for i, (name, p) in enumerate(pipelines.items()):
    cm = confusion_matrix(y_test, p.predict(X_test), labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    axes[i].set_title(f'{name} Confusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'breast_cancer_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()

# Save individual per-model confusion matrix images for Streamlit tabs
# Save individual confusion matrices directly (avoid image-cropping)
for i, (name, p) in enumerate(pipelines.items()):
    cm = confusion_matrix(y_test, p.predict(X_test), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    key = name.lower().replace(' ', '_').replace('-', '_')
    fig.savefig(os.path.join('plots', f'breast_cancer_cm_{key}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
print("Saved individual confusion matrix images.")

print("\nSTEP 18 — ROC Curves (Malignant-positive)")
y_malignant_test = (np.asarray(y_test) == 0).astype(int)
plt.figure(figsize=(8, 6))
for name, p in pipelines.items():
    proba = _safe_predict_proba(p, X_test)
    class_index = list(p.classes_).index(0)
    y_proba_malignant = proba[:, class_index]
    fpr, tpr, _ = roc_curve(y_malignant_test, y_proba_malignant)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate (Benign predicted as Malignant)')
plt.ylabel('True Positive Rate (Malignant Recall)')
plt.title('ROC Curves — Malignant as Positive Class')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'breast_cancer_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nSTEP 19 — Model Comparison Graph")
metrics_df_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df_melted)
plt.title('Model Comparison')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join('plots', 'breast_cancer_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nSTEPS 20-22 — Hyperparameter Tuning")
param_grids = {
    'Logistic Regression': {'classifier__C': [0.01, 0.1, 1, 10], 'classifier__solver': ['liblinear', 'lbfgs']},
    'kNN': {'classifier__n_neighbors': [3, 5, 7, 9], 'classifier__weights': ['uniform', 'distance'], 'classifier__p': [1, 2]},
    'SVM': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf'], 'classifier__gamma': ['scale', 'auto']},
    'Random Forest': {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20], 'classifier__min_samples_split': [2, 5]}
}

best_estimators = {}
grid_objects = {}
tuning_summary = []
for name in pipelines.keys():
    grid = GridSearchCV(pipelines[name], param_grids[name], cv=cv, scoring=malignant_roc_auc, n_jobs=-1)
    grid.fit(X_train, y_train)
    # store the GridSearchCV object for later reference
    grid_objects[name] = grid
    # Save cv_results_ for transparency
    try:
        cvres = pd.DataFrame(grid.cv_results_)
        cvres.to_csv(os.path.join(DATA_DIR, f'breast_cancer_gridsearch_{name.replace(" ","_")}_cv_results.csv'), index=False)
    except Exception:
        pass
    print(f"\n{name} Best Params: {grid.best_params_}")
    print(f"{name} Best Malignant ROC-AUC: {grid.best_score_:.4f}")
    best_estimators[name] = grid.best_estimator_
    params_str = ", ".join([f"{k.split('__')[-1]}={v}" for k, v in grid.best_params_.items()])
    tuning_summary.append({'Model': name, 'Best Parameters': params_str, 'Mean CV Malignant ROC-AUC': float(grid.best_score_)})

pd.DataFrame(tuning_summary).to_csv(os.path.join(DATA_DIR, 'breast_cancer_tuning_results.csv'), index=False)
print("Saved tuning results to 'breast_cancer_tuning_results.csv'.")

print("\nSTEP 23 — Evaluate Tuned Models")
tuned_metrics_list = []
for name, p in best_estimators.items():
    y_pred = p.predict(X_test)
    proba = _safe_predict_proba(p, X_test)
    class_index = list(p.classes_).index(0)
    y_proba_malignant = proba[:, class_index]
    y_malignant = (np.asarray(y_test) == 0).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec  = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1   = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    roc  = roc_auc_score(y_malignant, y_proba_malignant)

    # Use the GridSearchCV best_score_ as the CV ROC-AUC for the tuned estimator
    cv_roc = grid_objects[name].best_score_

    tuned_metrics_list.append({
        'Model':                      name,
        'Accuracy':                   acc,
        'Malignant Precision':        prec,
        'Malignant Recall':           rec,
        'Malignant F1':               f1,
        'Malignant Test ROC-AUC':     roc,
        'CV Malignant ROC-AUC':       cv_roc
    })

tuned_metrics_df = pd.DataFrame(tuned_metrics_list)
print(tuned_metrics_df)

print("\nSTEP 24 — Select Final Model")
final_model_name = max(
    grid_objects,
    key=lambda name: grid_objects[name].best_score_
)

best_model_row = tuned_metrics_df[
    tuned_metrics_df['Model'] == final_model_name
].iloc[0]
print(f"Final selected model name: {final_model_name}")
print("\nMetrics:")
print(best_model_row)
print("\nExplanation: 'ROC-AUC was used as the primary selection criterion because it evaluates ranking performance across all classification thresholds, making it more robust than accuracy alone.'")
print("Disclaimer: 'This model achieved the highest score under this evaluation setup. Results may vary with different data or configurations.'")

print("\nSTEP 25 — Save Model")
best_pipeline = best_estimators[final_model_name]
joblib.dump(best_pipeline, os.path.join(PROJECT_ROOT, 'breast_cancer_best_model.pkl'))

print("\nSTEP 26 — Save Metadata")
# Helper to convert numpy types for JSON serialization
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

best_params_clean = make_serializable(grid_objects[final_model_name].best_params_)
test_metrics_clean = make_serializable(best_model_row.drop('Model').to_dict())

metadata = {
    'dataset_name': 'Breast Cancer Dataset',
    'target_column': 'target',
    'target_labels': {'0': 'Malignant', '1': 'Benign'},
    'positive_class': 0,
    'positive_class_name': 'Malignant',
    'feature_names': feature_names,
    'num_features': len(feature_names),
    'selected_model': final_model_name,
    'test_metrics': test_metrics_clean,
    'cv_metrics': {'roc_auc': float(best_model_row['CV Malignant ROC-AUC'])},
    'best_hyperparameters': best_params_clean,
    'training_date': datetime.datetime.now().isoformat()
}
with open(os.path.join(PROJECT_ROOT, 'breast_cancer_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print("\nSTEP 27 — After-Preprocessing Export")
df.to_excel(os.path.join(DATA_DIR, 'breast_cancer_after_preprocessing.xlsx'), index=False, engine='openpyxl')


