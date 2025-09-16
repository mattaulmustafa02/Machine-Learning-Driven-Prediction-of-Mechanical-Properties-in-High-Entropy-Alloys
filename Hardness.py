# ============================================================
# Machine Learning Pipeline for Hardness (HV) Prediction in HEAs
# ============================================================

# --- Import Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import all_estimators
import sklearn.metrics as sm
import lightgbm as lgb

# --- Load Dataset ---
df = pd.read_excel("HV_Feature.xlsx")

# --- Define Features and Target ---
features = [
    'MagpieData avg_dev CovalentRadius',
    'MagpieData avg_dev Electronegativity',
    'MagpieData mean Electronegativity',
    'MagpieData avg_dev SpaceGroupNumber',
    'Mean cohesive energy',
    'MagpieData mean NsUnfilled',
    'MagpieData mean NUnfilled',
    'Shear modulus local mismatch'
]
X = df[features]
y = df['HV']

# --- Optional: Short Feature Names (useful for figures/tables) ---
short_names = {
    'MagpieData avg_dev CovalentRadius': 'Radius',
    'MagpieData mean Electronegativity': 'Mean_X',
    'MagpieData avg_dev SpaceGroupNumber': 'Space_Group',
    'Mean cohesive energy': 'Cohesive',
    'MagpieData mean NsUnfilled': 'NsUnfilled',
    'MagpieData mean NUnfilled': 'NUnfilled',
    'Shear modulus local mismatch': 'Modulus',
    'MagpieData avg_dev Electronegativity':'dev_X',
    'HV': 'HV'
}

# --- Visualize Hardness Distribution ---
plt.figure(figsize=(6,4))
plt.hist(y, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Hardness (HV)')
plt.ylabel('Frequency')
plt.title('Distribution of Hardness Values (HV)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Data Preprocessing (Standardization) ---
scaler = StandardScaler()
transform_x = make_column_transformer((scaler, features))
X_processed = transform_x.fit_transform(X)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=0
)

# ============================================================
# 1. LightGBM Model Training
# ============================================================
params = {
    'n_estimators': 690,
    'max_depth': 12,
    'boosting_type': 'gbdt',
    'num_leaves': 15,
    'min_data_in_leaf': 20,
    'bagging_fraction': 0.8,    # corrected spelling & typical value
    'bagging_freq': 0,
    'learning_rate': 0.1,
    'feature_fraction': 1.0
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000)

# --- Predictions and Evaluation ---
y_pred = model.predict(X_test)

print("LightGBM Performance:")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# ============================================================
# 2. Benchmarking Against All Scikit-learn Regressors
# ============================================================
model_name, model_r2 = [], []

estimators = all_estimators(type_filter='regressor')
for name, get_model in estimators:
    try:
        # Train each model
        model = get_model()
        model.fit(X_train, y_train)
        pred_y = model.predict(X_test)

        # Store results
        model_r2.append(sm.r2_score(y_test, pred_y))
        model_name.append(name)

    except Exception as e:
        # Some models may not work with given dataset
        print(f"Unable to use {name}: {e}")

# --- Store and Sort Results ---
results = pd.DataFrame({"Model Name": model_name, "R² Score": model_r2})
results = results.sort_values(by="R² Score", ascending=False)

print("\nTop Performing Models:")
print(results.head())

# ============================================================
# 3. Cross-Validation with Different Models
# ============================================================
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import lightgbm as lgb

# --- Define Reusable Evaluation Function ---
def evaluate_model_cv(model, X, y, k=10):
    """
    Perform K-Fold cross-validation and return R² and MSE scores.
    
    Parameters:
        model : sklearn estimator
            The regression model to evaluate.
        X : array-like
            Features.
        y : array-like
            Target values.
        k : int, default=10
            Number of folds for cross-validation.
            
    Returns:
        r2_scores, mse_scores : list
            Lists of R² and MSE scores for each fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores, mse_scores = [], []

    for fold_num, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train_fold = X[train_index] if not hasattr(X, 'iloc') else X.iloc[train_index]
        X_test_fold  = X[test_index] if not hasattr(X, 'iloc') else X.iloc[test_index]
        y_train_fold = y[train_index] if not hasattr(y, 'iloc') else y.iloc[train_index]
        y_test_fold  = y[test_index] if not hasattr(y, 'iloc') else y.iloc[test_index]

        # Train and predict
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        # Evaluate
        r2 = r2_score(y_test_fold, y_pred)
        mse = mean_squared_error(y_test_fold, y_pred)

        r2_scores.append(r2)
        mse_scores.append(mse)

        print(f"Fold {fold_num+1}: R² = {r2:.4f}, MSE = {mse:.4f}")

    print(f"\n✅ Average R² Score: {np.mean(r2_scores):.4f}")
    print(f"✅ Average MSE: {np.mean(mse_scores):.4f}\n")

    return r2_scores, mse_scores


# --- LightGBM Model ---
lgb_params = {
    'n_estimators': 690,
    'max_depth': 12,
    'boosting_type': 'gbdt',
    'num_leaves': 15,
    'min_data_in_leaf': 20,
    'bagging_fraction': 0.8,
    'bagging_freq': 0,
    'learning_rate': 0.1,
    'feature_fraction': 1.0,
    'random_state': 42
}
print("🔹 LightGBM Cross-Validation Results")
evaluate_model_cv(lgb.LGBMRegressor(**lgb_params), X_train, y_train)


# --- Extra Trees Model ---
print("🔹 Extra Trees Cross-Validation Results")
evaluate_model_cv(ExtraTreesRegressor(
    n_estimators=100,
    criterion='squared_error',
    random_state=42
), X_train, y_train)


# --- Random Forest Model ---
print("🔹 Random Forest Cross-Validation Results")
evaluate_model_cv(RandomForestRegressor(
    n_estimators=100,
    random_state=42
), X_train, y_train)

# ============================================================
# 4. Leave-One-Out Cross-Validation (LOOCV)
# ============================================================
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate_model_loocv(model, X, y):
    """
    Perform Leave-One-Out Cross-Validation (LOOCV) for a given model.
    
    Parameters:
        model : sklearn estimator
            Regression model to evaluate.
        X : array-like
            Features.
        y : array-like
            Target values.
            
    Returns:
        mean_r2 : float
            Average R² score across LOOCV.
        mean_rmse : float
            Average RMSE across LOOCV.
    """
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])

    # Calculate metrics
    mean_r2 = r2_score(y_true_all, y_pred_all)
    mean_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

    print(f"✅ LOOCV R² Score: {mean_r2:.4f}")
    print(f"✅ LOOCV RMSE: {mean_rmse:.4f}\n")

    return mean_r2, mean_rmse


# --- Random Forest ---
print("🔹 Random Forest LOOCV Results")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_model_loocv(rf_model, np.array(processed_x), np.array(y))

# --- Extra Trees ---
print("🔹 Extra Trees LOOCV Results")
et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
evaluate_model_loocv(et_model, np.array(processed_x), np.array(y))

# --- LightGBM ---
print("🔹 LightGBM LOOCV Results")
lgb_params = {
    'n_estimators': 690,
    'max_depth': 12,
    'boosting_type': 'gbdt',
    'num_leaves': 15,
    'min_data_in_leaf': 20,
    'bagging_fraction': 0.8,
    'bagging_freq': 0,
    'learning_rate': 0.1,
    'feature_fraction': 1.0,
    'random_state': 42,
    'verbose': -1
}
lgb_model = lgb.LGBMRegressor(**lgb_params)
evaluate_model_loocv(lgb_model, np.array(processed_x), np.array(y))
