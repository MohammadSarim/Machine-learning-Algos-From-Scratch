import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Data/Wage.csv')
df = df[['age', 'wage']]

df['X_bins'] = pd.cut(df['age'], 4)


X = pd.get_dummies(df['X_bins'], drop_first=True, dtype=np.float64)
Y = df['wage']


reg = LinearRegression().fit(X, Y)

age_grid = np.linspace(df.age.min(), df.age.max(), 1000)
age_bins = pd.cut(age_grid, bins=df['X_bins'].cat.categories)

X_grid = pd.get_dummies(age_bins, drop_first=True, dtype=np.float64)
X_grid = X_grid.reindex(columns=X.columns, fill_value=0)

y_pred = reg.predict(X_grid)

# Plot
plt.figure(figsize=(8, 5))
plt.title('Piecewise Constant Regression', fontsize=14)
plt.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.3, label='Actual Data')
plt.plot(age_grid, y_pred, c='green', linewidth=2, label='Step Prediction')
plt.xlabel('Age')
plt.ylabel('Wage')
plt.ylim(ymin=0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
