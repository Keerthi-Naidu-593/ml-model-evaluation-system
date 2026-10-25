# Dataset Sources and Attribution

This project uses publicly available datasets for educational purposes, machine learning experimentation, model evaluation, and comparison.

## 1. Breast Cancer Wisconsin (Diagnostic) Dataset

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Access method:** `sklearn.datasets.load_breast_cancer()`
- **Source:** scikit-learn dataset utilities
- **Original source:** UCI Machine Learning Repository
- **Dataset documentation:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- **Original dataset reference:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

**Purpose:** Used for binary classification experiments to distinguish malignant and benign breast cancer cases.

The dataset is accessed through scikit-learn. The original dataset source is acknowledged above. Refer to the dataset documentation and original source for applicable terms and licensing information.

## 2. Credit Card Default Dataset

- **Dataset:** Default of Credit Card Clients
- **Downloaded from:** Kaggle
- **Kaggle dataset URL:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/data
- **Original source:** UCI Machine Learning Repository
- **Original dataset reference:** https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- **Kaggle-listed license:** CC0: Public Domain
- **License reference:** https://creativecommons.org/publicdomain/zero/1.0/

**Purpose:** Used to train and evaluate machine learning classifiers for predicting whether a credit card client defaults on payment.

**Data handling:** A stratified subset of approximately 10,000 records is used for experimentation. The `ID` column is excluded from model training.

**Acknowledgement:**

Lichman, M. (2013). *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science. http://archive.ics.uci.edu/ml

The Kaggle dataset page lists the dataset under the CC0: Public Domain license and acknowledges the UCI Machine Learning Repository as the original source.

## Disclaimer

This project is developed for academic and educational purposes. The model outputs are experimental and are not intended to be used as the sole basis for real-world medical diagnoses or financial and credit decisions.