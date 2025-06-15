import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class GradientDescent:
    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.coef_ = None

    def fit(self, X, Y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.zeros(X.shape[1])
        n = len(X)
        for i in range(self.epochs):
            y_pred = np.dot(X, self.coef_)
            
            error = Y - y_pred

            gradients = -(1/n) * np.dot(X.T, error)

            self.coef_ = self.coef_ - (self.lr*gradients)

            if i % 1000 == 0:
                cost = np.sum(error ** 2) / (2 * n)
                print(f"Epoch {i}, Cost: {cost}")
                print(f"The intercept is {self.coef_[0]} and the coefficients are {self.coef_[1:]}")
    
    def predict(self, X_test):
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
        return X_test @ self.coef_

df = pd.read_csv('Data/Advertising.csv')

X = df[['TV', 'radio']].values
Y = df['sales'].values

# ======================
# 1. Train-Test Split
# ======================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize (using train stats)
scaler_mean = X_train.mean(axis=0)
scaler_std = X_train.std(axis=0)
X_train_scaled = (X_train - scaler_mean) / scaler_std
X_test_scaled = (X_test - scaler_mean) / scaler_std

# Train models
gd = GradientDescent(learning_rate=0.001, epochs=10000)
gd.fit(X_train_scaled, Y_train)

lr = LinearRegression()
lr.fit(X_train_scaled, Y_train)

# Evaluate
gd_pred = gd.predict(X_test_scaled)
lr_pred = lr.predict(X_test_scaled)

print("=== Train-Test Split Results ===")
print(f"GradientDescent RMSE: {np.sqrt(mean_squared_error(Y_test, gd_pred)):.4f}")
print(f"LinearRegression RMSE: {np.sqrt(mean_squared_error(Y_test, lr_pred)):.4f}")

# ======================
# 2. K-Fold Cross Validation
# ======================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
gd_rmse_scores, lr_rmse_scores = [], []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    
    # Standardize per fold
    fold_mean = X_train.mean(axis=0)
    fold_std = X_train.std(axis=0)
    X_train_scaled = (X_train - fold_mean) / fold_std
    X_test_scaled = (X_test - fold_mean) / fold_std
    
    # Train and evaluate GD
    gd = GradientDescent(0.01, 15000)
    gd.fit(X_train_scaled, Y_train)
    gd_rmse_scores.append(np.sqrt(mean_squared_error(Y_test, gd.predict(X_test_scaled))))
    
    # Train and evaluate LR
    lr = LinearRegression()
    lr.fit(X_train_scaled, Y_train)
    lr_rmse_scores.append(np.sqrt(mean_squared_error(Y_test, lr.predict(X_test_scaled))))

print("\n=== K-Fold CV Results ===")
print("GradientDescent:")
print(f"  Mean RMSE: {np.mean(gd_rmse_scores):.4f}")
print(f"  Std RMSE: {np.std(gd_rmse_scores):.4f}")

print("\nLinearRegression:")
print(f"  Mean RMSE: {np.mean(lr_rmse_scores):.4f}")
print(f"  Std RMSE: {np.std(lr_rmse_scores):.4f}")