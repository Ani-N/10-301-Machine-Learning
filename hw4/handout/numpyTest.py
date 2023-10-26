import numpy as np

X = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 11],
             [34, 1, 9],
             [1, 5, 17]])

v = np.array([12, 14, 7])

w = np.array([1, 2, 3])

print(np.shape(X))
print(np.shape(v))

u = np.zeros(5)

for i in range(5):
    for j in range(3):
        u[i] += X[i,j]*v[j]

print(u)
print(X @ v)
print(np.matmul(X, v))
print(np.dot(X, v))


print(np.dot(v.T, w))