import numpy as np
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt

# 1️⃣ Generate linearly separable data
np.random.seed(88)
n = 50
X_pos = np.random.randn(n, 2) + [2, 2]
y_pos = np.ones(n)

X_neg = np.random.randn(n, 2) + [-2, -2]
y_neg = -np.ones(n)

X = np.vstack((X_pos, X_neg))
y = np.hstack((y_pos, y_neg))

# 2️⃣ Compute Gram matrix
K = X @ X.T

# 3️⃣ Setup QP parameters for hard-margin SVM
P = cvxopt.matrix(np.outer(y, y) * K)
q = cvxopt.matrix(-1 * np.ones(X.shape[0]))

A = cvxopt.matrix(y, (1, X.shape[0]))
b = cvxopt.matrix(0.0)

G = cvxopt.matrix(-np.eye(X.shape[0]))
h = cvxopt.matrix(np.zeros(X.shape[0]))

# 4️⃣ Solve QP
cvxopt.solvers.options['show_progress'] = True
solution = cvxopt.solvers.qp(P, q, G, h, A, b)

# 5️⃣ Extract multipliers and support vectors
multipliers = np.ravel(solution['x'])
has_positive_multiplier = multipliers > 1e-7

sv_multipliers = multipliers[has_positive_multiplier]
support_vectors = X[has_positive_multiplier]
support_vectors_y = y[has_positive_multiplier]

# 6️⃣ Compute w
w = np.sum((sv_multipliers * support_vectors_y)[:, np.newaxis] * support_vectors, axis=0)

# 7️⃣ Compute b
b = support_vectors_y[0] - np.dot(w, support_vectors[0])

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
plt.title('Hard-Margin SVM Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.show()
