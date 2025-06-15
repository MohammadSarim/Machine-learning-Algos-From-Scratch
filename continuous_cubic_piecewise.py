import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('Data/Wage.csv')
df = df[['wage', 'age']]
cut = 50

# Add indicator for right side
df['is_right'] = (df['age'] >= cut).astype(int)

# Create separate cubic features for left and right
df['age_l'] = df['age'] * (1 - df['is_right'])
df['age_l2'] = df['age_l'] ** 2
df['age_l3'] = df['age_l'] ** 3

df['age_r'] = df['age'] * df['is_right']
df['age_r2'] = df['age_r'] ** 2
df['age_r3'] = df['age_r'] ** 3

# Define X and y
X = df[['age', 'age_l2', 'age_l3', 'age_r2', 'age_r3']]
y = df['wage']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Smooth curve for plotting
age_range = np.linspace(df['age'].min(), df['age'].max(), 300)
is_right = (age_range >= cut).astype(int)
age_l = age_range * (1 - is_right)
age_r = age_range * is_right

X_plot = pd.DataFrame({
    'age': age_range,
    'age_l2': age_l ** 2,
    'age_l3': age_l ** 3,
    'age_r2': age_r ** 2,
    'age_r3': age_r ** 3,
})
Y_plot = model.predict(X_plot)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['age'], df['wage'], color='lightgray', label='Actual Wage')
ax.plot(age_range, Y_plot, color='red', label='Continuous Piecewise Cubic')
plt.axvline(cut, color='k', linestyle='--', alpha=0.3)
plt.legend()
plt.show()
