import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def compute_class_means(X, Y):
    class_labels = Y.idxmax(axis=1)
    class_means = X.groupby(class_labels).mean().round(4)
    means = {cls: class_means.loc[cls] for cls in class_means.index}
    return means

def covariance_matrix(X, Y, means):
    class_labels = Y.idxmax(axis=1).unique()
    cov = np.zeros((X.shape[1], X.shape[1]))
    for label in class_labels:
        diff = X[Y.idxmax(axis=1) == label] - means[label]
        cov += diff.T @ diff
    
    N = len(Y)
    K = len(class_labels)
    cov = cov / (N - K) 
    return cov

def log_prior_calculation(X, Y):
    class_labels = Y.idxmax(axis=1)
    log_prior = np.log(X.groupby(class_labels).size() / X.shape[0])
    return log_prior

def compute_discriminant(x, cov, mean, log_prior):
    term1 = x.T @ inv(cov) @ mean
    term2 = 0.5 * mean.T @ inv(cov) @ mean
    return term1 - term2 + log_prior

def predict(X_new, model):
    predictions = []
    for _, x in X_new.iterrows():
        discriminants = []
        for cls in model['classes']:
            disc = compute_discriminant(
                x.values, 
                model['cov_matrix'], 
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
            disc = compute_discriminant(
                x.values, 
                model['cov_matrix'], 
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

# Main execution
df = pd.read_csv('Diabetes.csv')
X = df.iloc[:, 2:-1]
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
X = X.astype(float)
X = (X - X.mean())/X.std()
Y = df.iloc[:, -1].str.strip()
Y = pd.get_dummies(Y) 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

class_means = compute_class_means(X_train, Y_train)
cov_matrix = covariance_matrix(X_train, Y_train, class_means)
log_prior = log_prior_calculation(X_train, Y_train)

trained_model = {
    'class_means': class_means,
    'cov_matrix': cov_matrix,
    'log_prior': log_prior,
    'classes': Y_train.columns.tolist()
}

classes = Y.columns.tolist()

# Get true labels (convert from one-hot encoding)
y_true = Y_test.idxmax(axis=1)  

# Generate predictions
y_pred = predict(X_test, trained_model)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Confusion Matrix (using sklearn)
cm = confusion_matrix(y_true, y_pred, labels=classes)
print("\nConfusion Matrix (sklearn):")
print(pd.DataFrame(cm, index=classes, columns=classes))

# Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))
