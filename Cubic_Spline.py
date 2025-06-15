import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('Data/Wage.csv')
df = df[['wage', 'age']]

cut = 50

# Add polynomial terms
df['age_2'] = df['age'] ** 2
df['age_3'] = df['age'] ** 3

# Add truncated cubic term (zero before cut, cubic after cut)
df['trunc_3'] = np.where(df['age'] > cut, (df['age'] - cut) ** 3, 0)

# Define features and target
X = df[['age', 'age_2', 'age_3', 'trunc_3']]
Y = df['wage']

# Fit linear model
model = LinearRegression()
model.fit(X, Y)

# Predictions on original data
pred = model.predict(X)
residuals = Y - pred
n = len(Y)  # number of observations
p = X.shape[1]  # number of predictors

# Estimate residual variance
res_var = np.sum(residuals**2) / (n - p - 1)

# Compute (X'X)^-1
X_mat = np.column_stack([np.ones(n), X.values])  # add intercept term manually
XtX_inv = np.linalg.inv(X_mat.T.dot(X_mat))

# Prepare prediction points
age_range = np.linspace(df['age'].min(), df['age'].max(), 300)
X_plot = pd.DataFrame({
    'age': age_range,
    'age_2': age_range**2,
    'age_3': age_range**3,
    'trunc_3': np.where(age_range > cut, (age_range - cut)**3, 0)
})
X_plot_mat = np.column_stack([np.ones(len(age_range)), X_plot.values])  # add intercept

# Predicted values for smooth curve
Y_plot = model.predict(X_plot)

# Calculate standard errors for each prediction
se_fit = np.sqrt(np.sum(X_plot_mat.dot(XtX_inv) * X_plot_mat, axis=1) * res_var)

# Compute 95% confidence interval
alpha = 0.05
t_val = stats.t.ppf(1 - alpha/2, df=n - p - 1)

upper = Y_plot + t_val * se_fit
lower = Y_plot - t_val * se_fit

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['age'], df['wage'], color='lightgray', label='Actual Wage')
ax.plot(age_range, Y_plot, color='red', label='Cubic Spline')
ax.fill_between(age_range, lower, upper, color='red', alpha=0.3, label='95% Confidence Interval')
plt.axvline(cut, color='k', linestyle='--', alpha=0.3)
plt.legend()

ax.grid(True)
plt.show()
