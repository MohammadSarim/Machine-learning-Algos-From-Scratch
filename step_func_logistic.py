import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Wage.csv')
df = df[['age', 'wage']]

df['X_bins'] = pd.cut(df['age'], 4)

X = pd.get_dummies(df['X_bins'], drop_first=True, dtype=np.float64)


Y = df['wage'].apply(lambda x: 1 if x > 250 else 0)


log = LogisticRegression().fit(X, Y)


y_train_pred = log.predict(X)
accuracy = np.mean(Y == y_train_pred)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

age_grid = np.linspace(df.age.min(), df.age.max(), 1000)
age_bins = pd.cut(age_grid, bins=df['X_bins'].cat.categories)

X_grid = pd.get_dummies(age_bins, drop_first=True, dtype=np.float64)
X_grid = X_grid.reindex(columns=X.columns, fill_value=0)

y_prob = log.predict_proba(X_grid)[:, 1]

plt.figure(figsize=(8, 5))
plt.title('Logistic Regression with Step Function', fontsize=14)
plt.scatter(df.age, Y, facecolor='None', edgecolor='k', alpha=0.3, label='Actual Labels (0/1)')
plt.plot(age_grid, y_prob, color='green', linewidth=2, label='Predicted Probability (wage > 250)')
plt.xlabel('Age')
plt.ylabel('Probability')
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
