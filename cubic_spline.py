import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


def fx(x_num):
    return 1 / (1 + 5 * (x_num ** 2))


x = np.array([-7 / 8, -5 / 8, -3 / 8, -1 / 8, 1 / 8, 3 / 8, 5 / 8, 7 / 8])
y = np.zeros(8)

for i in range(len(x)):
    y[i] = fx(x[i])

ncs = scipy.interpolate.CubicSpline(x, y, bc_type='natural')
x_new = np.linspace(-1.00, 1.00)
y_new = ncs(x_new)

plt.plot(x_new, ncs(x_new))
plt.scatter(x, y)  # punkty do konstrukcji
plt.title("Naturalny Splajn Kubiczny")
plt.show()
