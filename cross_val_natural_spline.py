import pandas as pd
import numpy as np
from patsy import dmatrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Wage.csv")
df = df[['wage', 'age']].dropna()

X_base = df['age']
y = df['wage']

# Try 0 to 10 knots
knot_candidates = range(0, 11)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
avg_mse_list = []

# Cross-validation for each number of knots
for k in knot_candidates:
    fold_mse = []

    if k == 0:
        # No interior knots: just a natural spline with no internal flexibility
        for train_index, val_index in kf.split(df):
            X_train = X_base.iloc[train_index]
            y_train = y.iloc[train_index]
            X_val = X_base.iloc[val_index]
            y_val = y.iloc[val_index]

            basis_train = dmatrix("cr(age, df=3)", {"age": X_train}, return_type='dataframe')
            basis_val = dmatrix("cr(age, df=3)", {"age": X_val}, return_type='dataframe')

            model = LinearRegression()
            model.fit(basis_train, y_train)
            preds = model.predict(basis_val)
            mse = mean_squared_error(y_val, preds)
            fold_mse.append(mse)
    else:
        knots = np.quantile(X_base, np.linspace(0, 1, k + 2)[1:-1])  # interior knots only
        for train_index, val_index in kf.split(df):
            X_train = X_base.iloc[train_index]
            y_train = y.iloc[train_index]
            X_val = X_base.iloc[val_index]
            y_val = y.iloc[val_index]

            basis_train = dmatrix("cr(age, knots=knots)", {"age": X_train, "knots": knots}, return_type='dataframe')
            basis_val = dmatrix("cr(age, knots=knots)", {"age": X_val, "knots": knots}, return_type='dataframe')

            model = LinearRegression()
            model.fit(basis_train, y_train)
            preds = model.predict(basis_val)
            mse = mean_squared_error(y_val, preds)
            fold_mse.append(mse)

    avg_mse = np.mean(fold_mse)
    avg_mse_list.append(avg_mse)
    print(f"Knots: {k+1}, Average CV MSE: {avg_mse:.2f}")

# Find optimal number of knots
optimal_k = knot_candidates[np.argmin(avg_mse_list)]
print(f"\n✅ Optimal number of knots: {optimal_k+1}")

# Plot CV MSE vs number of knots
plt.figure(figsize=(10, 6))
plt.plot(knot_candidates, avg_mse_list, marker='o', linestyle='-')
plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.title('Cross-Validation MSE vs. Number of Knots (Natural Cubic Spline)')
plt.xlabel('Number of Knots')
plt.ylabel('Average CV MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
