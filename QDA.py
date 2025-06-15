import numpy as np
import pandas as pd
from numpy.linalg import inv, det
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def compute_class_means(X, Y):
    class_labels = Y.idxmax(axis=1)
    class_means = X.groupby(class_labels).mean().round(4)
    means = {cls: class_means.loc[cls] for cls in class_means.index}
    return means

def compute_class_covariances(X, Y, means):
    """Modified for QDA: Compute class-specific covariance matrices"""
    class_labels = Y.idxmax(axis=1).unique()
    covs = {}
    for label in class_labels:
        class_data = X[Y.idxmax(axis=1) == label]
        diff = class_data - means[label]
        covs[label] = (diff.T @ diff) / (len(class_data) - 1)  # Class-specific covariance
    return covs

def log_prior_calculation(X, Y):
    class_labels = Y.idxmax(axis=1)
    log_prior = np.log(X.groupby(class_labels).size() / X.shape[0])
    return log_prior

def compute_qda_discriminant(x, cov, mean, log_prior):
    """QDA Discriminant Function"""
    diff = x - mean
    cov_inv = inv(cov)
    term1 = -0.5 * np.log(det(cov))  # Log determinant term
    term2 = -0.5 * diff.T @ cov_inv @ diff  # Quadratic term
    return term1 + term2 + log_prior

def predict_qda(X_new, model):
    """QDA Prediction Function"""
    predictions = []
    for _, x in X_new.iterrows():
        discriminants = []
        for cls in model['classes']:
            disc = compute_qda_discriminant(
                x.values,
                model['class_covs'][cls],  # Class-specific covariance
                model['class_means'][cls].values,
                model['log_prior'][cls]
            )
            discriminants.append(disc)
        predicted_class = model['classes'][np.argmax(discriminants)]
        predictions.append(predicted_class)
    return predictions

def predict_proba(X_new, model):
    probabilities = []
    for _, x in X_new.iterrows():
        discriminants = []
        for cls in model['classes']:
            disc = compute_qda_discriminant(
                x.values, 
                model['class_covs'][cls], 
                model['class_means'][cls].values, 
                model['log_prior'][cls]
            )
            discriminants.append(disc)
        # Numerical stability trick: subtract max
        max_disc = np.max(discriminants)
        exp_discs = np.exp(np.array(discriminants) - max_disc)
        probs = exp_discs / np.sum(exp_discs)
        probabilities.append(dict(zip(model['classes'], probs)))
    # Convert to DataFrame for consistency
    return pd.DataFrame(probabilities, index=X_new.index)

# Data loading and preprocessing (unchanged)
df = pd.read_csv('Diabetes.csv')
X = df.iloc[:, 2:-1]
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
X = X.astype(float)
X = (X - X.mean())/X.std()
Y = df.iloc[:, -1].str.strip()
Y = pd.get_dummies(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# QDA Training
class_means = compute_class_means(X_train, Y_train)
class_covs = compute_class_covariances(X_train, Y_train, class_means)  # Class-specific covs
log_prior = log_prior_calculation(X_train, Y_train)

qda_model = {
    'class_means': class_means,
    'class_covs': class_covs,  # Stores all class-specific cov matrices
    'log_prior': log_prior,
    'classes': Y_train.columns.tolist()
}

classes = Y.columns.tolist()
# QDA Prediction
y_pred = predict_qda(X_test, qda_model)
y_true = Y_test.idxmax(axis=1)

# Evaluation
accuracy = np.mean(y_pred == y_true)
print(f"QDA Model accuracy: {accuracy:.4f}")

# Confusion Matrix (using sklearn)
cm = confusion_matrix(y_true, y_pred, labels=classes)
print("\nConfusion Matrix (sklearn):")
print(pd.DataFrame(cm, index=classes, columns=classes))

# Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))
