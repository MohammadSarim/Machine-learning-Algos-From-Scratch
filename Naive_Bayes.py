import numpy as np
import pandas as pd
from math import pi
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def prior_probability(X, Y):
    class_labels = Y.idxmax(axis=1)
    prior = Y.mean()  # Equivalent to class counts divided by total samples
    return prior

def calculate_stats(X, Y):
    class_labels = Y.idxmax(axis=1)
    class_stats = {}
    
    for cls in Y.columns:
        class_data = X[class_labels == cls]
        stats = {}
        
        # Handle numerical features (Gaussian)
        numerical_features = class_data.select_dtypes(include=['number'])
        stats['mean'] = numerical_features.mean()
        stats['std'] = numerical_features.std()
        
        # Handle binary features (Bernoulli)
        binary_features = class_data.select_dtypes(include=['bool', 'uint8'])
        stats['prob'] = binary_features.mean()  # Probability of feature being 1
        
        class_stats[cls] = stats
    
    return class_stats

def gaussian_likelihood(x, mean, std):
    # Handle case where std is 0 (add small epsilon to avoid division by zero)
    std = np.where(std == 0, 1e-9, std)
    exponent = -0.5 * ((x - mean) / std)**2
    return -np.log(std * np.sqrt(2 * pi)) + exponent

def bernoulli_likelihood(x, prob):
    # log(P(x|prob)) = x*log(prob) + (1-x)*log(1-prob)
    prob = np.clip(prob, 1e-9, 1 - 1e-9)  # Avoid log(0)
    return x * np.log(prob) + (1 - x) * np.log(1 - prob)

def calculate_log_likelihood(x, class_stats):
    log_likelihoods = {}
    
    for cls, stats in class_stats.items():
        # Gaussian likelihood for numerical features
        numerical_mask = x.index.isin(stats['mean'].index)
        if numerical_mask.any():
            gaussian_ll = gaussian_likelihood(
                x[numerical_mask], 
                stats['mean'], 
                stats['std']
            ).sum()
        else:
            gaussian_ll = 0
        
        # Bernoulli likelihood for binary features
        binary_mask = x.index.isin(stats.get('prob', pd.Series()).index)
        if binary_mask.any():
            bernoulli_ll = bernoulli_likelihood(
                x[binary_mask],
                stats['prob']
            ).sum()
        else:
            bernoulli_ll = 0
        
        log_likelihoods[cls] = gaussian_ll + bernoulli_ll
    
    return log_likelihoods

def predict(X, prior, class_stats):
    predictions = []
    for _, x in X.iterrows():
        log_likelihoods = calculate_log_likelihood(x, class_stats)
        # Add log prior to log likelihood
        class_scores = {cls: log_likelihoods[cls] + np.log(prior[cls]) 
                       for cls in prior.index}
        predicted_class = max(class_scores.items(), key=lambda x: x[1])[0]
        predictions.append(predicted_class)
    return np.array(predictions)

# Load and preprocess data
df = pd.read_csv('Data/Diabetes.csv')
X = df.iloc[:, 2:-1]
Y = df.iloc[:, -1].str.strip()

# Convert categorical features (Gender) to binary
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

# Convert Y to one-hot encoding
Y = pd.get_dummies(Y)

# Normalize numerical features (only numerical columns)
numerical_cols = X.select_dtypes(include=['number']).columns
X[numerical_cols] = (X[numerical_cols] - X[numerical_cols].mean()) / X[numerical_cols].std()

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
prior = prior_probability(X_train, Y_train)
class_stats = calculate_stats(X_train, Y_train)

# Predictions
y_pred = predict(X_test, prior, class_stats)
y_true = Y_test.idxmax(axis=1)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred))