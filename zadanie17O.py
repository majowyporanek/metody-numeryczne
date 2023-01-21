import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(x):
    a = 1
    b = 100
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


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


# Newton's method
def Newton(X):
    history = []
    x = X
    j = 0
    tol = 1e-6
    max_iter = 1000
    for i in range(max_iter):
        grad = gradient(X)
        hess = hessian(X)
        dx = np.linalg.solve(hess, -grad)
        X = X + dx
        history.append(X)
        j = j + 1
        if np.linalg.norm(dx) < tol:
            value = rosenbrock(X)
            return x, j, X, value, history
            break


min_iter = np.array([0, 100])
average_iter = 0
n = 10

for i in range(n):
    x_start = np.array([np.random.uniform(-100, 100), np.random.uniform(1, 100)])
    print("Punkt startowy: : ", x_start)
    wynik = Newton(x_start)
    average_iter = wynik[1] + average_iter
    print("Uzyskane minimum:", wynik[2], "ilosc iteracji : ", wynik[1])
    if min_iter[1] > wynik[1]:
        min_iter = wynik
    print("")

print("Srednia ilosc iteracji, aby zblizyc sie do minimum: ", average_iter / n)
print("Punkt startowy, ktory potrzebowal najmniejsza ilosc iteracji = ", min_iter[1], ": ", min_iter[0])

print("")
pointhistory = Newton(min_iter[0])[4]
results = np.empty(min_iter[1])
xvalues = np.empty(len(pointhistory))
yvalues = np.empty(len(pointhistory))

for i in range(len(pointhistory)):
    xvalues[i] = pointhistory[i][0]
    yvalues[i] = pointhistory[i][1]
    results[i] = rosenbrock(pointhistory[i])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xvalues, yvalues, results, label="Położenia kolejnych minimalizacji przy liczbie iteracji = " + str(min_iter[1]))

plt.legend(loc='upper left', fontsize=8)
plt.show()

