import numpy as np

# licznik iteracji
# punkty poczatkowe:
x1 = np.array([1.2, 1.2])


def rosenbrock(x):
    a = 1
    b = 100
    return (a - x1[0]) ** 2 + b * (x1[1] - x1[0] ** 2) ** 2


def gradient(x):
    a = 1
    b = 100
    grad = np.zeros(2)
    grad[0] = -2 * (a - x[0]) - 4 * b * (x[1] - x[0] ** 2) * x[0]
    grad[1] = 2 * b * (x[1] - x[0] ** 2)
    return grad


def hessian(x):
    a = 1
    b = 100
    hess = np.zeros((2, 2))
    hess[0, 0] = 2 - 4 * b * x[1] + 12 * b * x[0] ** 2
    hess[0, 1] = -4 * b * x[0]
    hess[1, 0] = -4 * b * x[0]
    hess[1, 1] = 2 * b
    return hess


# metoda Newtona
def newtonsMethod(max_iter, tol, x):
    j = 0
    for i in range(max_iter):
        grad = gradient(x)
        hess = hessian(x)
        dx = np.linalg.solve(hess, -grad)
        x = x + dx
        j = j + 1
        if np.linalg.norm(dx) < tol:
            return x, j
            break


# Print the result
print("Minimum at", newtonsMethod(1000, 1e-6, x1))
print("Minimum value", rosenbrock(x1))
