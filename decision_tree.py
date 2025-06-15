import pandas as pd
import numpy as np
import pprint
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def load_data(filepath, target_col='Balance'):
    df = pd.read_csv(filepath)
    
    df['Own'] = df['Own'].map({'No': 0, 'Yes': 1}).astype(float)
    df['Student'] = df['Student'].map({'No': 0, 'Yes': 1}).astype(float)
    df['Married'] = df['Married'].map({'No': 0, 'Yes': 1}).astype(float)
    df = pd.get_dummies(df, columns=['Region'], drop_first=True, dtype=np.float64)
    
    features = df.drop(columns=[target_col])
    features_standardized = (features - features.mean(axis=0)) / features.std(axis=0)
    
    df_processed = pd.concat([features_standardized, df[target_col]], axis=1)
    return df_processed, target_col

def rss_calculator(region1, region2, target_col):
    """Calculate RSS using dynamic target column"""
    rss_r1 = np.sum((region1[target_col] - region1[target_col].mean())**2)
    rss_r2 = np.sum((region2[target_col] - region2[target_col].mean())**2)
    return rss_r1 + rss_r2

def build_tree(df, target_col, min_samples=3, depth=0, max_depth=3):
    """Build tree with dynamic target column"""
    if len(df) < min_samples or depth == max_depth:
        return {
            'leaf': True,
            'prediction': df[target_col].mean(),
            'samples': len(df),
            'depth': depth
        }

    min_rss = float('inf')
    best_cutpoint = None
    best_feature = None
    best_left = None
    best_right = None

    for feature in df.columns.drop(target_col):
        X = df[feature].values
        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]
        cutpoints = [(X_sorted[i] + X_sorted[i+1])/2 for i in range(len(X_sorted)-1)]

        for cutpoint in cutpoints:
            left = df[df[feature] < cutpoint]
            right = df[df[feature] >= cutpoint]

            if len(left) < min_samples or len(right) < min_samples:
                continue

            rss = rss_calculator(left, right, target_col)

            if rss < min_rss:
                min_rss = rss
                best_cutpoint = cutpoint
                best_feature = feature
                best_left = left
                best_right = right

    if best_feature is None:
        return {
            'leaf': True,
            'prediction': df[target_col].mean(),
            'samples': len(df),
            'depth': depth
        }

    left_tree = build_tree(best_left, target_col, min_samples, depth+1, max_depth)
    right_tree = build_tree(best_right, target_col, min_samples, depth+1, max_depth)

    return {
        'leaf': False,
        'split_feature': best_feature,
        'split_value': best_cutpoint,
        'left': left_tree,
        'right': right_tree,
        'samples': len(df),
        'depth': depth
    }

def predict(tree, X):
    if tree['leaf']:
        return tree['prediction']
    if X[tree['split_feature']] < tree['split_value']:
        return predict(tree['left'], X)
    else:
        return predict(tree['right'], X)

def predict_batch(tree, df):
    return np.array([predict(tree, row) for _, row in df.iterrows()])


if __name__ == "__main__":

    df_processed, target_col = load_data('Credit.csv', target_col='Balance')
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    custom_tree = build_tree(pd.concat([X_train, y_train], axis=1), 
                          target_col, min_samples=3, max_depth=3)
    
    y_pred_custom = predict_batch(custom_tree, X_test)
    
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    rmse_custom = np.sqrt(mse_custom)
    r2_custom = r2_score(y_test, y_pred_custom)
    
    # --- scikit-learn Implementation ---
    sklearn_tree = DecisionTreeRegressor(
        max_depth=3,
        min_samples_split=3,
        random_state=42
    )
    sklearn_tree.fit(X_train, y_train)
    y_pred_sklearn = sklearn_tree.predict(X_test)
    
    # Calculate metrics
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    rmse_sklearn = np.sqrt(mse_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    
    print("\nModel Comparison Results:")
    print(f"{'Metric':<20} {'Custom Tree':<15} {'sklearn Tree':<15}")
    print("-"*50)
    print(f"{'MSE':<20} {mse_custom:<15.2f} {mse_sklearn:<15.2f}")
    print(f"{'RMSE':<20} {rmse_custom:<15.2f} {rmse_sklearn:<15.2f}")
    print(f"{'R-squared':<20} {r2_custom:<15.2f} {r2_sklearn:<15.2f}")

    print("\nFirst 5 Test Samples Comparison:")
    print(f"{'Sample':<10} {'Actual':<10} {'Custom Pred':<12} {'sklearn Pred':<12}")
    print("-"*50)
    for i in range(5):
        print(f"{i:<10} {y_test.iloc[i]:<10.2f} {y_pred_custom[i]:<12.2f} {y_pred_sklearn[i]:<12.2f}")