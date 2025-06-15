import pandas as pd
from sklearn.preprocessing import LabelEncoder
from patsy import dmatrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from pygam import LinearGAM, s, f

df = pd.read_csv('Wage.csv')

df = df.dropna()

df['education'] = df['education'].astype('category')

year_spline = dmatrix("cr(year, df=4)", data=df, return_type='dataframe')
age_spline = dmatrix("cr(age, df=5)", data=df, return_type='dataframe')

edu_dummies = pd.get_dummies(df['education'], drop_first=True)

X_ns = pd.concat([year_spline, age_spline, edu_dummies.reset_index(drop=True)], axis=1)

model_ns = LinearRegression().fit(X_ns, df['wage'])

X = df[['year', 'age', 'education']].copy()

# Convert 'education' to numeric codes
le = LabelEncoder()
X['education'] = le.fit_transform(X['education'])

y = df['wage']

# Fit GAM with smoothing splines for year and age, factor term for education
gam = LinearGAM(s(0, n_splines=4) + s(1, n_splines=5) + f(2)).fit(X, y)

edof_per_coef = gam.statistics_['edof_per_coef']
terms = gam.terms

start_idx = 0
edof_per_term = []
for term in terms:
    n_coefs = term.n_coefs  # number of coefficients for this term
    term_edof = edof_per_coef[start_idx : start_idx + n_coefs].sum()
    edof_per_term.append(term_edof)
    start_idx += n_coefs

print("Effective degrees of freedom per term:", edof_per_term)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
titles = ['year', 'age', 'education']

for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=0.95)[1], c='r', ls='--')
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()
