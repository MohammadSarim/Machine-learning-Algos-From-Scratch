import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class Ridge_Regression:
    def __init__(self, learning_rate=0.1, epochs=10000):
        self.lr = learning_rate
        self.epochs = epochs
        self.coef_ = None

    def fit(self, X, Y, lbd): 
        self.coef_ = np.zeros(X.shape[1])  
        n = len(X)
        self.bias = 0
        for _ in range(self.epochs):
            y_pred = self.predict(X)
            error = Y - y_pred
            
            db =  (-np.sum(error)) / n
            dW = (-(X.T @ error) + (lbd * self.coef_)) / n            

            self.bias -= self.lr * db
            self.coef_ -= self.lr * dW
            
            # if _ % 1000 == 0:
            #     cost = (1/(2*n)) * (np.sum(error**2) + lbd * np.sum(self.coef_[1:]**2))
            #     print(f'Cost: {cost:.2f}')     

    def predict(self, X):
        return X @ self.coef_ + self.bias


df = pd.read_csv('Credit.csv')
dict1 = {'No': 0, 'Yes': 1}

for col in ['Own', 'Student', 'Married']:
    df[col] = df[col].map(dict1).astype(np.float64)

df = pd.get_dummies(df, columns=['Region'], drop_first=True, dtype=np.float64)

X = df.drop(columns=['Balance'])
y = df['Balance'].astype(np.float64)


alphas = [0.001, 0.01, 0.1, 1, 10, 100]


k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

print("Custom Ridge Regression with Different λ Values:")
for alpha in alphas:
    print(f"\nλ = {alpha}:")
    mse_scores = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = Ridge_Regression(learning_rate = 0.1, epochs=10000)
        model.fit(X_train_scaled, y_train, lbd = alpha)
        
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        
        print(f"  Fold {fold+1} MSE: {mse:.2f}")
    
    print(f"  Average MSE: {np.mean(mse_scores):.2f}")
    print(f"  Std Dev: {np.std(mse_scores):.2f}")

print("\nScikit-learn Ridge Regression with Different α Values:")
for alpha in alphas:
    print(f"\nα = {alpha}:")
    sklearn_mse = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        sklearn_mse.append(mse)
        
        print(f"  Fold {fold+1} MSE: {mse:.2f}")
    
    print(f"  Average MSE: {np.mean(sklearn_mse):.2f}")
    print(f"  Std Dev: {np.std(sklearn_mse):.2f}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.values  
final_model = Ridge(alpha=0.1)
final_model.fit(X_scaled, y)

custom_model = Ridge_Regression(learning_rate=0.1, epochs=10000)
custom_model.fit(X_scaled, y, lbd = 0.1)
print("Final Coefficients (Custom Ridge):", custom_model.coef_)
print("Final Coefficients (Scikit learn):", final_model.coef_)