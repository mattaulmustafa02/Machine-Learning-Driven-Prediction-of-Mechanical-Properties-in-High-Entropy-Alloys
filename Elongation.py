# ============================================
# Prediction of Elongation in High Entropy Alloys
# Data Loading, Preprocessing, SVR Model Training & Evaluation
# Includes Benchmarking with All Scikit-learn Regressors
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sklearn.metrics as sm
from sklearn.utils import all_estimators

# ---------------------------
# 1. Load Dataset
# ---------------------------
DATA_PATH = "ElongationFeature.xlsx"   # Update path if needed
df = pd.read_excel(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------
# 2. Feature & Target Selection
# ---------------------------
feature_cols = [
    'Electronegativity local mismatch',
    'MagpieData avg_dev GSvolume_pa',
    'MagpieData range MendeleevNumber',
    'MagpieData mean NsUnfilled',
    'MagpieData maximum GSvolume_pa',
    'MagpieData maximum NUnfilled',
    'MagpieData avg_dev MeltingT',
    'MagpieData avg_dev MendeleevNumber',
    'MagpieData minimum SpaceGroupNumber',
    'MagpieData mode SpaceGroupNumber',
    'MagpieData mean NValence',
    'Lambda entropy'
]

X = df[feature_cols]
y = df['Elongation']

# ---------------------------
# 3. Exploratory Data Analysis
# ---------------------------
plt.figure(figsize=(6,4))
plt.hist(y, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Elongation')
plt.ylabel('Frequency')
plt.title('Distribution of Elongation Values')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ---------------------------
# 4. Feature Scaling
# ---------------------------
scaler = StandardScaler()
transformer = make_column_transformer((scaler, feature_cols))
X_processed = transformer.fit_transform(X)

# ---------------------------
# 5. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=0
)

# ---------------------------
# 6. Support Vector Regression (SVR)
# ---------------------------
model = SVR(kernel='rbf', C=110, gamma=0.07)
model.fit(X_train, y_train)

# ---------------------------
# 7. Predictions
# ---------------------------
y_pred = model.predict(X_test)
print("Sample Predictions:", y_pred[:5])

# ---------------------------
# 8. SVR Model Evaluation
# ---------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nSVR Model Evaluation Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# ---------------------------
# 9. Benchmark: All Regressors in scikit-learn
# ---------------------------
model_name = []
model_r2 = []

estimators = all_estimators(type_filter='regressor')
for name, get_model in estimators:
    try:
        model = get_model()
        model.fit(X_train, y_train)
        pred_y = model.predict(X_test)
        model_r2.append(sm.r2_score(y_test, pred_y))
        model_name.append(name)
    except Exception as e:
        print(f'Unable to run {name}:', e)

results = pd.DataFrame({"Model Name": model_name, "R² Score": model_r2})
results = results.sort_values(by="R² Score", ascending=False).reset_index(drop=True)

print("\nModel Benchmark Results (Top 10):")
print(results.head(10))

# ============================================
# Cross-Validation (LOOCV) for Multiple Models
# ============================================

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

# ---------------------------
# Utility function for LOOCV
# ---------------------------
def loocv_r2(model, X, y):
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred[0])

    return r2_score(y_true_all, y_pred_all)

# ---------------------------
# 1. Support Vector Regressor
# ---------------------------
svr_model = SVR(kernel='rbf', C=110, gamma=0.07)
svr_r2 = loocv_r2(svr_model, X_processed, y.values)
print(f"SVR LOOCV R² Score: {svr_r2:.4f}")

# ---------------------------
# 2. AdaBoost Regressor
# ---------------------------
base_estimator = DecisionTreeRegressor(
    max_depth=3,
    min_samples_split=5,
    random_state=42
)

ada_model = AdaBoostRegressor(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    loss='linear',
    random_state=42
)

ada_r2 = loocv_r2(ada_model, X_processed, y.values)
print(f"AdaBoost LOOCV R² Score: {ada_r2:.4f}")

# ---------------------------
# 3. Extra Trees Regressor
# ---------------------------
et_model = ExtraTreesRegressor(
    n_estimators=100,
    criterion='squared_error',
    random_state=42
)

et_r2 = loocv_r2(et_model, X_processed, y.values)
print(f"ExtraTrees LOOCV R² Score: {et_r2:.4f}")


# ---------------------------
# Utility function for K-fold
# ---------------------------

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# SVR
model = SVR(kernel='rbf', C=110, gamma=0.07)
r2_scores = []

for train_idx, test_idx in kf.split(X, y):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_test_fold)  
    r2_scores.append(r2_score(y_test_fold, y_pred))

print("SVR Mean R²:", np.mean(r2_scores))


# ExtraTrees 
et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
r2_scores = []

for train_idx, test_idx in kf.split(X, y):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    et_model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_test_fold)   
    r2_scores.append(r2_score(y_test_fold, y_pred))

print("ExtraTrees Mean R²:", np.mean(r2_scores))


# AdaBoost 
base_estimator = DecisionTreeRegressor(max_depth=3, min_samples_split=5, random_state=42)
ada_model = AdaBoostRegressor(estimator=base_estimator, n_estimators=50,
                              learning_rate=1.0, loss='linear', random_state=42)
r2_scores = []

for train_idx, test_idx in kf.split(X, y):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    ada_model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_test_fold)   
    r2_scores.append(r2_score(y_test_fold, y_pred))

print("AdaBoost Mean R²:", np.mean(r2_scores))


from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
import numpy as np

# Repeated K-Fold cross-validator
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

# ---------------- SVR ----------------
svr_model = SVR(kernel='rbf', C=110, gamma=0.07)
r2_scores = []

for train_index, test_index in rkf.split(X):
    X_train_fold, X_test_fold = processed_x[train_index], processed_x[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    svr_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = svr_model.predict(X_test_fold)
    r2_scores.append(r2_score(y_test_fold, y_pred_fold))

print(f"✅ SVR Repeated K-Fold R²: Mean = {np.mean(r2_scores):.4f}, Std = {np.std(r2_scores):.4f}")


# ---------------- AdaBoost ----------------
base_estimator = DecisionTreeRegressor(max_depth=3, min_samples_split=5, random_state=42)

ada_model = AdaBoostRegressor(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    loss='linear',
    random_state=42
)

r2_scores = []

for train_index, test_index in rkf.split(X):
    X_train_fold, X_test_fold = processed_x[train_index], processed_x[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    ada_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = ada_model.predict(X_test_fold)
    r2_scores.append(r2_score(y_test_fold, y_pred_fold))

print(f"✅ AdaBoost Repeated K-Fold R²: Mean = {np.mean(r2_scores):.4f}, Std = {np.std(r2_scores):.4f}")


# ---------------- ExtraTrees ----------------
et_model = ExtraTreesRegressor(
    n_estimators=100,
    criterion='squared_error',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

r2_scores = []

for train_index, test_index in rkf.split(X):
    X_train_fold, X_test_fold = processed_x[train_index], processed_x[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    et_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = et_model.predict(X_test_fold)

    r2_scores.append(r2_score(y_test_fold, y_pred_fold))

print(f"✅ ExtraTrees Repeated K-Fold R²: Mean = {np.mean(r2_scores):.4f}, Std = {np.std(r2_scores):.4f}")
