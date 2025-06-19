import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

class SimplePLS:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.beta_pls = None
        self.y_mean = None
        self.y_std = None

    def fit(self, X, y):
        self.y_mean = y.mean()
        self.y_std = y.std()

        y = (y - self.y_mean) / self.y_std  

        n, p = X.shape
        X_current = X.copy()
        y_current = y.copy()
        self.beta_pls = np.zeros(p)

        for _ in range(self.n_components):
            phi_m = X_current.T @ y_current
            w_m = phi_m / np.linalg.norm(phi_m)   
            z_m = X_current @ w_m
            theta_m = (z_m @ y_current) / (z_m @ z_m)
            self.beta_pls += theta_m * w_m

            # ✅ Deflate X
            proj = (z_m @ X_current) / (z_m @ z_m)
            X_current -= np.outer(z_m, proj)

            # ✅ Deflate y
            y_current = y_current - theta_m * z_m

    def predict(self, X):
        y_pred_scaled = X @ self.beta_pls
        return y_pred_scaled * self.y_std + self.y_mean  

# Load and preprocess data
df = pd.read_csv('Data/Credit.csv')
df['Own'] = df['Own'].map({'No': 0, 'Yes': 1}).astype(float)
df['Student'] = df['Student'].map({'No': 0, 'Yes': 1}).astype(float)
df['Married'] = df['Married'].map({'No': 0, 'Yes': 1}).astype(float)
df = pd.get_dummies(df, columns=['Region'], drop_first=True, dtype=np.float64)

X = df.drop(columns=['Balance']).values
y = df['Balance'].values.astype(float)

# Standardize X
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Your custom PLS
obj1 = SimplePLS(n_components=2)
obj1.fit(X, y)
y_pred_custom = obj1.predict(X)

# Scikit-learn PLS model
pls_sklearn = PLSRegression(n_components=2)
pls_sklearn.fit(X, y)
y_pred_sklearn = pls_sklearn.predict(X).ravel()  # ravel() to flatten predictions

# Print results and accuracy scores
print("Custom SimplePLS predictions:")
print(y_pred_custom[:5])
print("\nScikit-learn PLSRegression predictions:")
print(y_pred_sklearn[:5])

print("\nR² score for Custom SimplePLS:", r2_score(y, y_pred_custom))
print("R² score for scikit-learn PLSRegression:", r2_score(y, y_pred_sklearn))
