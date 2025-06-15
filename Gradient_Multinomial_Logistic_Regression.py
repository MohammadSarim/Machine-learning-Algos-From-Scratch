import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

class GradientDescent:
    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.coef_ = None

    def _sigmoid(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
        return  exp_z / np.sum(exp_z, axis=1, keepdims=True)     

    def fit(self, X, Y):
        if isinstance(Y, pd.DataFrame):
            self.class_dict_ = {i: label for i, label in enumerate(Y.columns)}
        else:
            self.class_dict_ = {i: label for i, label in enumerate(np.unique(Y))}
            
        Y = Y.values if isinstance(Y, pd.DataFrame) else Y
        X = X.values if isinstance(X, pd.DataFrame) else X

        X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.zeros((Y.shape[1], X.shape[1]))
        n = len(X)

        for i in range(self.epochs):
            z = X @ self.coef_.T
            y_pred = self._sigmoid(z)    
            error = y_pred - Y
            gradients = error.T @ X / X.shape[0]
            self.coef_ -= self.lr * gradients  

            if i % 500 == 0:
                cost = -np.mean(np.sum(Y * np.log(y_pred + 1e-15), axis=1))
                print('===================== The Cost and Coeffiecients are ============================')
                print(f"Epoch {i}, Cost: {cost:.4f}")        
                print(self.coef_)  

        self.intercept_ = self.coef_[:, 0]
        self.coef_ = np.round(self.coef_[:, 1:], 4)
    
    def predict_proba(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        z = X @ np.c_[self.intercept_, self.coef_].T
        return self._sigmoid(z)

    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.array([self.class_dict_[i] for i in indices])

    def score(self, X, Y):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y_pred = self.predict(X)
        
        if isinstance(Y, pd.DataFrame):
            y_true = Y.idxmax(axis=1)
        elif hasattr(Y, 'ndim') and Y.ndim == 2:  
            y_true = np.array([self.class_dict_[i] for i in np.argmax(Y, axis=1)])
        else: 
            y_true = Y
            
        return accuracy_score(y_true, y_pred)

df = pd.read_csv('Data/Diabetes.csv')
X = df.iloc[:, 2:-1]
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
X = X.astype(float)
X = (X - X.mean())/X.std()
Y = df.iloc[:, -1].str.strip()
Y = pd.get_dummies(Y) 

obj1 = GradientDescent(0.7, 20000)
obj1.fit(X, Y)
  
y_true = Y.idxmax(axis=1) 
y_pred = obj1.predict(X)
    
sklearn_lr = LogisticRegression(multi_class='multinomial', 
                              solver='lbfgs',
                              max_iter=1000,
                              penalty=None,
                              random_state=42)
sklearn_lr.fit(X, Y.idxmax(axis=1)) 

np.set_printoptions(suppress=True, precision=4)

y_pred_scikit = sklearn_lr.predict(X)

print("\nCustom GD Coefficients:")
print(np.round(obj1.coef_, 4))

print("\nScikit-learn Coefficients:")
print(np.round(sklearn_lr.coef_, 4))

print("\nCustom GD Intercepts:")
print(np.round(obj1.intercept_, 4))

print("\nScikit-learn Intercepts:")
print(np.round(sklearn_lr.intercept_, 4))

custom_score = obj1.score(X, Y)
sklearn_score = sklearn_lr.score(X, Y.idxmax(axis=1))

print("\nCustom GD Accuracy:", round(custom_score, 4))
print("Scikit-learn Accuracy:", round(sklearn_score, 4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix(Custom Gradient):\n", cm)

cm = confusion_matrix(y_true, y_pred_scikit)
print("Confusion Matrix(Scikit-learn):\n", cm)
