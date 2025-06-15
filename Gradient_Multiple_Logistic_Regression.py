import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.linear_model import LogisticRegression

class GradientDescent:
    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.coef_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.zeros(X.shape[1])
        n = len(X)
      
        for i in range(self.epochs):
            z = np.dot(X, self.coef_)
            y_pred = self._sigmoid(z)
            error = y_pred - Y
            gradients = (1/n) * np.dot(X.T, error)
            self.coef_ = self.coef_ - (self.lr * gradients)

            if i % 500 == 0:
                cost = - (1/n) * np.sum(Y * np.log(y_pred + 1e-15) + (1 - Y) * np.log(1 - y_pred + 1e-15))
                print(f"Epoch {i}, Cost: {cost:.4f}")

    def predict_proba(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(np.dot(X, self.coef_))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, Y):
        y_pred = self.predict(X)
        return np.mean(y_pred == Y) 


df = pd.read_csv('Default.csv')
encoding_dict = {'No': 0, 'Yes': 1}
df['default'] = df['default'].map(encoding_dict)
df['student'] = df['student'].map(encoding_dict)


df['balance'] = (df['balance'] - df['balance'].mean()) / df['balance'].std()
df['income'] = (df['income'] - df['income'].mean()) / df['income'].std()

X = df.iloc[:, 1:]
Y = df.iloc[:, 0]


obj1 = GradientDescent(learning_rate=0.1, epochs=20000)
obj1.fit(X, Y)


model = LogisticRegression(penalty=None)  
model.fit(X, Y)


y_pred_custom = obj1.predict(X)
y_prob_custom = obj1.predict_proba(X)


y_pred_sk = model.predict(X)
y_prob_sk = model.predict_proba(X)[:, 1]

print("\n=== Custom GradientDescent ===")
print(f"Accuracy: {obj1.score(X, Y):.4f}")
print(f"Log Loss: {log_loss(Y, y_prob_custom):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(Y, y_pred_custom))
print("Classification Report:")
print(classification_report(Y, y_pred_custom))

print("\n=== Scikit-Learn LogisticRegression ===")
print(f"Accuracy: {model.score(X, Y):.4f}")
print(f"Log Loss: {log_loss(Y, y_prob_sk):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(Y, y_pred_sk))
print("Classification Report:")
print(classification_report(Y, y_pred_sk))