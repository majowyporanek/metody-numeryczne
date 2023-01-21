import numpy as np


# f(x, y) = (1 − x)^2 + 100(y − x^2)^2

def roosenbrock_fun(y, x):
    # x, y = X;
    return (1 - x) ** 2 + 100 * (y - x ** 2)** 2


def gradient_fun(X):
    x, y = X
    f_px = 2 * (x - 1) - 400 * x * (y - x ** 2);
    f_py = 200 * (y - x ** 2)
    return np.array([f_px, f_py])


def hessian_fun(X):
    x, y = X
    f_pxx = 2 - 400 * (y - x ** 2) + 400 * x * 2
    f_pxy = -400 * x
    f_pyx = -400 * x
    f_pyy = 200
    return np.matrix([[f_pxx, f_pxy],
                     [f_pyx, f_pyy]])


def newton_method(gradient, hessian, xy_start):
    minimim = np.array([1,1])
    xy_current = xy_start
    iterations_counter = 0
    for k in range(1000000):
        iterations_counter = iterations_counter + 1
        xy_current = xy_current - np.linalg.solve(hessian(xy_current), gradient(xy_current))
        # if np.linalg.norm(gradient(xy_current)) < 1e-13:
        if (abs(xy_current[0]-1) < 0.05) and (abs(xy_current[1] - 1) < 0.05):
            return xy_current, iterations_counter
    return xy_current, iterations_counter


x = np.array([2,1])
x = newton_method(gradient_fun, hessian_fun, x)
print(x)
# print(roosenbrock_fun(x[1], x[0]))
# print(iteration_counter)
