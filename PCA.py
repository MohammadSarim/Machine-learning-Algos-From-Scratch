import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eigh

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X):
        cov_matrix = np.cov(X, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        
        # Select top n_components
        if self.n_components is not None:
            self.eigenvectors = self.eigenvectors[:, :self.n_components]

        X_pca = np.dot(X, self.eigenvectors)
        return X_pca, self.eigenvalues, self.eigenvectors

# Load and preprocess data
df = pd.read_csv('Data/Credit.csv')
df['Own'] = df['Own'].map({'No': 0, 'Yes': 1}).astype(float)
df['Student'] = df['Student'].map({'No': 0, 'Yes': 1}).astype(float)
df['Married'] = df['Married'].map({'No': 0, 'Yes': 1}).astype(float)
df = pd.get_dummies(df, columns=['Region'], drop_first=True, dtype=np.float64)

X = df.drop(columns=['Balance'])
y = df['Balance'].astype(float)    

components_to_test = range(1, X.shape[1]+1)
k = components_to_test[1]
kf = KFold(n_splits=k, shuffle=True, random_state=42)

results = []

for n_components in components_to_test:
    fold_mses = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Apply StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # Use same scaler as training
        
        pca = PCA(n_components=n_components)
        X_train_pca, _, _ = pca.fit(X_train_scaled)
        X_test_pca = np.dot(X_test_scaled, pca.eigenvectors)
        
        lr = LinearRegression()
        lr.fit(X_train_pca, y_train)
        
        y_pred = lr.predict(X_test_pca)
        fold_mses.append(mean_squared_error(y_test, y_pred))
    
    avg_mse = np.mean(fold_mses)
    results.append((n_components, avg_mse))
    print(f"Components: {n_components}, Avg MSE: {avg_mse:.2f}")

best_n_components, best_mse = min(results, key=lambda x: x[1])
print(f"\nOptimal number of components: {best_n_components} with MSE: {best_mse:.2f}")

import matplotlib.pyplot as plt
plt.plot([r[0] for r in results], [r[1] for r in results])
plt.xlabel('Number of PCA Components')
plt.ylabel('Average MSE')
plt.title('PCA Component Optimization')
plt.show()


