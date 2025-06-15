import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def load_data(filepath, target_col='Student'):
    df = pd.read_csv(filepath)

    df['Own'] = df['Own'].map({'No': 0, 'Yes': 1}).astype(float)
    df['Married'] = df['Married'].map({'No': 0, 'Yes': 1}).astype(float)
    df['Student'] = df['Student'].map({'No': 0, 'Yes': 1}).astype(int)
    df = pd.get_dummies(df, columns=['Region'], drop_first=True, dtype=np.float64)

    features = df.drop(columns=[target_col])
    df_processed = pd.concat([features, df[target_col]], axis=1)
    return df_processed, target_col

def gini_impurity(groups, target_col):
    total_samples = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        class_counts = group[target_col].value_counts(normalize=True)
        score = np.sum(class_counts ** 2)
        gini += (1.0 - score) * (len(group) / total_samples)
    return gini

def build_tree_classification(df, target_col, min_samples=3, depth=0, max_depth=3):
    if len(df) < min_samples or depth == max_depth or df[target_col].nunique() == 1:
        return {
            'leaf': True,
            'prediction': df[target_col].mode()[0],
            'samples': len(df),
            'depth': depth
        }

    min_gini = float('inf')
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

            gini = gini_impurity([left, right], target_col)

            if gini < min_gini:
                min_gini = gini
                best_cutpoint = cutpoint
                best_feature = feature
                best_left = left
                best_right = right

    if best_feature is None:
        return {
            'leaf': True,
            'prediction': df[target_col].mode()[0],
            'samples': len(df),
            'depth': depth
        }

    left_tree = build_tree_classification(best_left, target_col, min_samples, depth+1, max_depth)
    right_tree = build_tree_classification(best_right, target_col, min_samples, depth+1, max_depth)

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
    df_processed, target_col = load_data('Data/Credit.csv', target_col='Student')
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    custom_tree = build_tree_classification(pd.concat([X_train, y_train], axis=1),
                                            target_col, min_samples=3, max_depth=3)

    y_pred_custom = predict_batch(custom_tree, X_test)

    # Custom Tree Evaluation
    print("\nCustom Classification Tree Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_custom))
    print(classification_report(y_test, y_pred_custom))

    # Scikit-learn Tree for Comparison
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=3, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_sklearn = clf.predict(X_test)

    print("\nScikit-learn Tree Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_sklearn))
    print(classification_report(y_test, y_pred_sklearn))

    print("\nFirst 5 Test Samples Comparison:")
    print(f"{'Sample':<10} {'Actual':<10} {'Custom Pred':<12} {'sklearn Pred':<12}")
    print("-"*50)
    for i in range(5):
        print(f"{i:<10} {y_test.iloc[i]:<10} {y_pred_custom[i]:<12} {y_pred_sklearn[i]:<12}")
