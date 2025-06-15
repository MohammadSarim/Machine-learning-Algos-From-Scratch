import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('Wage.csv')
df = df[['wage', 'age']]

polynomial_degree = 3
cut = 50

for deg in range(2, polynomial_degree+1):
    df[f'age_{deg}'] = df['age']**deg

df_less_50 = df[df['age'] < 50]
df_greater_50 = df[df['age'] >= 50]

X_less_50 = df_less_50[['age', 'age_2', 'age_3']]
Y_less_50 = df_less_50[['wage']]

X_greater_50 = df_greater_50[['age', 'age_2', 'age_3']]
Y_greater_50 = df_greater_50[['wage']]

X_train_less_50, X_test_less_50, Y_train_less_50, Y_test_less_50 = train_test_split(X_less_50, Y_less_50, test_size=0.2, random_state=42)
X_train_greater_50, X_test_greater_50, Y_train_greater_50, Y_test_greater_50 = train_test_split(X_greater_50, Y_greater_50, test_size=0.2, random_state=42)

lr_less_50 = LinearRegression()
lr_less_50.fit(X_train_less_50, Y_train_less_50)

Y_pred_less_50 = lr_less_50.predict(X_test_less_50)

lr_greater_50 = LinearRegression()
lr_greater_50.fit(X_train_greater_50, Y_train_greater_50)

Y_pred_greater_50 = lr_greater_50.predict(X_test_greater_50)


# Create smooth curves
age_range_less_50 = np.linspace(X_test_less_50['age'].min(), X_test_less_50['age'].max(), 300)
age_range_greater_50 = np.linspace(X_test_greater_50['age'].min(), X_test_greater_50['age'].max(), 300)

# Create polynomial features
X_plot_less_50 = pd.DataFrame({
    'age': age_range_less_50,
    'age_2': age_range_less_50**2,
    'age_3': age_range_less_50**3
})
X_plot_greater_50 = pd.DataFrame({
    'age': age_range_greater_50,
    'age_2': age_range_greater_50**2,
    'age_3': age_range_greater_50**3
})

# Predict
Y_plot_less_50 = lr_less_50.predict(X_plot_less_50)
Y_plot_greater_50 = lr_greater_50.predict(X_plot_greater_50)

# Plot actual data
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['age'], df['wage'], color='lightgray', label='Actual Wage')

# Plot smooth curves
ax.plot(age_range_less_50, Y_plot_less_50, color='red', label='Polynomial Fit: Age < 50')
ax.plot(age_range_greater_50, Y_plot_greater_50, color='red', label='Polynomial Fit: Age ≥ 50')

# Vertical cut line
plt.axvline(cut, color='k', linestyle='--', alpha=0.3)

plt.legend()
plt.show()
