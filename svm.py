import numpy as np
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt

# 1️⃣ Generate *non*-linearly separable data
np.random.seed(88)
n = 50
X_pos = np.random.randn(n, 2) + [1.5, 1.5]
y_pos = np.ones(n)

X_neg = np.random.randn(n, 2) + [-1.5, -1.5]
y_neg = -np.ones(n)

# Add some overlap
X_pos[:5] = np.random.randn(5, 2) + [-1, -1]
X_neg[:5] = np.random.randn(5, 2) + [1, 1]

X = np.vstack((X_pos, X_neg))
y = np.hstack((y_pos, y_neg))

# 2️⃣ Compute Gram matrix
K = X @ X.T

# 3️⃣ Setup QP parameters for soft-margin SVM
C = 1.0

P = cvxopt.matrix(np.outer(y, y) * K)
q = cvxopt.matrix(-1 * np.ones(X.shape[0]))

A = cvxopt.matrix(y, (1, X.shape[0]), tc='d')
b_ = cvxopt.matrix(0.0)

G_std = -np.eye(X.shape[0])
h_std = np.zeros(X.shape[0])

G_slack = np.eye(X.shape[0])
h_slack = C * np.ones(X.shape[0])

G = cvxopt.matrix(np.vstack((G_std, G_slack)))
h = cvxopt.matrix(np.hstack((h_std, h_slack)))

# 4️⃣ Solve QP
cvxopt.solvers.options['show_progress'] = True
solution = cvxopt.solvers.qp(P, q, G, h, A, b_)

# 5️⃣ Extract multipliers and support vectors
multipliers = np.ravel(solution['x'])
has_positive_multiplier = multipliers > 1e-5

sv_multipliers = multipliers[has_positive_multiplier]
support_vectors = X[has_positive_multiplier]
support_vectors_y = y[has_positive_multiplier]

# 6️⃣ Compute w
w = np.sum((sv_multipliers * support_vectors_y)[:, np.newaxis] * support_vectors, axis=0)

# 7️⃣ Compute b
b = np.mean([y_k - np.dot(w, x_k) for (y_k, x_k) in zip(support_vectors_y, support_vectors)])

print(f"Weight vector (w): {w}")
print(f"Bias (b): {b}")

# 8️⃣ Plot the data, support vectors, decision boundary, margin
plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(X_pos[:,0], X_pos[:,1], color='b', label='Positive class')
plt.scatter(X_neg[:,0], X_neg[:,1], color='r', label='Negative class')

# Highlight support vectors
plt.scatter(support_vectors[:,0], support_vectors[:,1], s=100,
            facecolors='none', edgecolors='k', label='Support vectors')

# Plot decision boundary and margins
xx = np.linspace(np.min(X[:,0])-1, np.max(X[:,0])+1, 100)
yy = np.linspace(np.min(X[:,1])-1, np.max(X[:,1])+1, 100)
XX, YY = np.meshgrid(xx, yy)
XY = np.c_[XX.ravel(), YY.ravel()]
Z = XY @ w + b
Z = Z.reshape(XX.shape)

plt.contour(XX, YY, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
plt.legend()
plt.title(f'Soft-Margin SVM Decision Boundary (C={C})')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()
