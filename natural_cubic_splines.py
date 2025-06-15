import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

# Load data
df = pd.read_csv('Data/Wage.csv')
df = df[['wage', 'age']]

cut = 50  # interior knot
a = df['age'].min()
b = df['age'].max()

# Define truncated cubic function
def truncated_cubic(x, knot):
    return np.where(x > knot, (x - knot)**3, 0)

# Compute the natural cubic spline basis function (one knot case)
def natural_spline_basis(x, knot, a, b):
    d_k = (truncated_cubic(x, knot) - truncated_cubic(x, b)) / (b - knot)
    d_b = (truncated_cubic(x, a) - truncated_cubic(x, b)) / (b - a)
    return d_k - d_b

# Construct design matrix with intercept, linear term, and natural spline basis
X = pd.DataFrame({
    'age': df['age'],
    'ns_basis': natural_spline_basis(df['age'], cut, a, b)
})
Y = df['wage']

# Fit linear regression with intercept automatically included by sklearn
model = LinearRegression()
model.fit(X, Y)

# Residuals and variance estimate
pred = model.predict(X)
residuals = Y - pred
n = len(Y)
p = X.shape[1] + 1  # +1 for intercept
res_var = np.sum(residuals**2) / (n - p)

# Prepare design matrix with intercept column for variance calculations
X_mat = np.column_stack([np.ones(n), X.values])
XtX_inv = np.linalg.inv(X_mat.T @ X_mat)

# Prediction points for smooth curve
age_range = np.linspace(a, b, 300)
X_plot = pd.DataFrame({
    'age': age_range,
    'ns_basis': natural_spline_basis(age_range, cut, a, b)
})
X_plot_mat = np.column_stack([np.ones(len(age_range)), X_plot.values])

# Predictions and standard errors for CI
Y_plot = model.predict(X_plot)
se_fit = np.sqrt(np.sum(X_plot_mat @ XtX_inv * X_plot_mat, axis=1) * res_var)

# 95% confidence intervals
alpha = 0.05
t_val = stats.t.ppf(1 - alpha / 2, df=n - p)

upper = Y_plot + t_val * se_fit
lower = Y_plot - t_val * se_fit

# Plot results
plt.figure(figsize=(12,8))
plt.scatter(df['age'], df['wage'], color='lightgray', label='Actual Wage')
plt.plot(age_range, Y_plot, color='red', label='Natural Cubic Spline Fit')
plt.fill_between(age_range, lower, upper, color='red', alpha=0.3, label='95% Confidence Interval')
plt.axvline(cut, color='black', linestyle='--', alpha=0.3)
plt.title('Natural Cubic Spline with 1 Knot at Age 50')
plt.xlabel('Age')
plt.ylabel('Wage')
plt.legend()
plt.grid(True)
plt.show()
