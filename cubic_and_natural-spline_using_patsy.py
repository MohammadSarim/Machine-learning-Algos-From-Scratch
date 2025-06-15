import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import patsy

# Load data
df = pd.read_csv('Wage.csv')
df = df[['wage', 'age']]

cut = 50  # Knot location
n = len(df)
alpha = 0.05

# =============================
# 1) Truncated power basis spline
# =============================
df['age_2'] = df['age'] ** 2
df['age_3'] = df['age'] ** 3
df['trunc_3'] = np.where(df['age'] > cut, (df['age'] - cut) ** 3, 0)

X1 = df[['age', 'age_2', 'age_3', 'trunc_3']]
Y = df['wage']

model1 = LinearRegression().fit(X1, Y)
pred1 = model1.predict(X1)
residuals1 = Y - pred1
p1 = X1.shape[1]

res_var1 = np.sum(residuals1**2) / (n - p1 - 1)

X1_mat = np.column_stack([np.ones(n), X1.values])
XtX_inv1 = np.linalg.inv(X1_mat.T.dot(X1_mat))

# =============================
# 2) Natural cubic spline with patsy 'cr' basis
# =============================
age = df['age'].values

spline_basis = patsy.dmatrix("cr(age, knots=(50,))", {"age": age}, return_type='dataframe')
print(spline_basis)
model2 = LinearRegression().fit(spline_basis, Y)
pred2 = model2.predict(spline_basis)
residuals2 = Y - pred2
p2 = spline_basis.shape[1]

res_var2 = np.sum(residuals2**2) / (n - p2 - 1)

# IMPORTANT: DO NOT add intercept manually here, patsy includes it
X2_mat = spline_basis.values
XtX_inv2 = np.linalg.inv(X2_mat.T.dot(X2_mat))

# =============================
# Prediction grid for plotting
# =============================
age_range = np.linspace(df['age'].min(), df['age'].max(), 300)

# For truncated power spline
age_2 = age_range ** 2
age_3 = age_range ** 3
trunc_3 = np.where(age_range > cut, (age_range - cut) ** 3, 0)
X1_pred_mat = np.column_stack([np.ones(len(age_range)), age_range, age_2, age_3, trunc_3])

Y1_plot = model1.predict(np.column_stack([age_range, age_2, age_3, trunc_3]))
se_fit1 = np.sqrt(np.sum(X1_pred_mat.dot(XtX_inv1) * X1_pred_mat, axis=1) * res_var1)
t_val1 = stats.t.ppf(1 - alpha/2, df=n - p1 - 1)
upper1 = Y1_plot + t_val1 * se_fit1
lower1 = Y1_plot - t_val1 * se_fit1

# For natural cubic spline (patsy) - variable name fix here!
spline_basis_pred = patsy.dmatrix("cr(age, knots=(50,))", {"age": age_range}, return_type='dataframe')
X2_pred_mat = spline_basis_pred.values

Y2_plot = model2.predict(spline_basis_pred)
se_fit2 = np.sqrt(np.sum(X2_pred_mat.dot(XtX_inv2) * X2_pred_mat, axis=1) * res_var2)
t_val2 = stats.t.ppf(1 - alpha/2, df=n - p2 - 1)
upper2 = Y2_plot + t_val2 * se_fit2
lower2 = Y2_plot - t_val2 * se_fit2

# =============================
# Plotting both fits side-by-side
# =============================
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Plot truncated power basis spline
axes[0].scatter(df['age'], df['wage'], color='lightgray', label='Actual Wage')
axes[0].plot(age_range, Y1_plot, color='red', label='Truncated Power Basis')
axes[0].fill_between(age_range, lower1, upper1, color='red', alpha=0.3, label='95% CI')
axes[0].axvline(cut, color='k', linestyle='--', alpha=0.3)
axes[0].set_title("Cubic Spline with Truncated Power Basis")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Wage")
axes[0].legend()
axes[0].grid(True)

# Plot natural cubic spline
axes[1].scatter(df['age'], df['wage'], color='lightgray', label='Actual Wage')
axes[1].plot(age_range, Y2_plot, color='blue', label='Natural Cubic Spline (patsy)')
axes[1].fill_between(age_range, lower2, upper2, color='blue', alpha=0.3, label='95% CI')
axes[1].axvline(cut, color='k', linestyle='--', alpha=0.3)
axes[1].set_title("Natural Cubic Spline with patsy")
axes[1].set_xlabel("Age")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
