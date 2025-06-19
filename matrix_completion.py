import pandas as pd
import numpy as np
from scipy.linalg import svd

k = 2
max_iter = 100
tolerance = 1e-5

X = pd.DataFrame([
    [5.0,   np.nan, 3.0],
    [2.0,   4.0,    np.nan],
    [np.nan, 6.0,   1.0]
])

X_original = X.copy()
X_filled = X.fillna(X.mean())

prev_error = np.inf

for iteration in range(max_iter):
    U, s, Vt = svd(X_filled)
    X_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    X_filled[X_original.isna()] = X_k

    error = np.sum((X - X_k)**2, axis=1).sum()
    print(f"Iteration {iteration+1}, Observed Error: {error:.6f}")

    if abs(prev_error - error) < tolerance:
        print("Converged.")
        break
    prev_error = error

print("\nFinal Completed Matrix:")
print(np.round(X_filled, 4))
