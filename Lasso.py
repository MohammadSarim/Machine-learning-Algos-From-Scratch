import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class Lasso_Regression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = 0.0
    
    @staticmethod
    def soft_thresholding(rho, lmbd):
        """Soft thresholding operator for L1 regularization"""
        if rho < -lmbd:
            return rho + lmbd
        elif rho > lmbd:
            return rho - lmbd
        else:
            return 0.0

    def fit(self, X, y, lbd):
        n, m = X.shape
        self.coef_ = np.zeros(m)
        self.intercept_ = 0

        for _ in range(self.epochs):
            y_pred = self.predict(X)
            error = y - y_pred

            grad_b = -np.sum(error) / n
            self.intercept_ -= self.lr * grad_b

            for j in range(m):
                grad_j = -(X[:, j] @ error) / n
                rho = self.coef_[j] - self.lr * grad_j
                self.coef_[j] = self.soft_thresholding(rho, self.lr * lbd)

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

df = pd.read_csv('Credit.csv')
df['Own'] = df['Own'].map({'No': 0, 'Yes': 1}).astype(float)
df['Student'] = df['Student'].map({'No': 0, 'Yes': 1}).astype(float)
df['Married'] = df['Married'].map({'No': 0, 'Yes': 1}).astype(float)
df = pd.get_dummies(df, columns=['Region'], drop_first=True, dtype=np.float64)

X = df.drop(columns=['Balance'])
y = df['Balance'].astype(float)

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Custom Lasso Regression with Different λ Values:")
for alpha in alphas:
    print(f"\nλ = {alpha}")
    mse_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Lasso_Regression(learning_rate=0.01, epochs=10000)
        model.fit(X_train_scaled, y_train, lbd=alpha)

        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        print(f"  Fold {fold+1} MSE: {mse:.2f}")

    print(f"  Average MSE: {np.mean(mse_scores):.2f}")
    print(f"  Std Dev: {np.std(mse_scores):.2f}")

print("\nScikit-learn Lasso Regression with Different α Values:")
for alpha in alphas:
    print(f"\nα = {alpha}")
    mse_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SklearnLasso(alpha=alpha, max_iter=10000)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        print(f"  Fold {fold+1} MSE: {mse:.2f}")

    print(f"  Average MSE: {np.mean(mse_scores):.2f}")
    print(f"  Std Dev: {np.std(mse_scores):.2f}")
