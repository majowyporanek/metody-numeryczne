import cmath
import numpy
import numpy as np


# z(i+1) = n * p_n(zi) / p1_n(zi) + sqrt((n-1) [ p1_n(zi)^2 - n*p_n(zi)*p2_n(zi))
def laguerre_method(coef, start_x):
    precision = 1e-10
    err = np.inf
    max_iter = 100
    z0 = start_x
    p = np.poly1d(coef)
    p1 = np.polyder(p, 1)
    p2 = np.polyder(p, 2)
    n = len(coef) - 1

    i = 0
    while i < max_iter and err > precision:
        p_nzi = np.polyval(p, z0)
        p1_nzi = np.polyval(p1, z0)
        p2_nzi = np.polyval(p2, z0)
        under_sqrt = (n - 1) * ((n - 1) * p1_nzi ** 2 - n * p_nzi * p2_nzi)
        sqrt_val = cmath.sqrt(under_sqrt)

        m1 = p1_nzi + sqrt_val
        m2 = p1_nzi + sqrt_val
        if np.abs(m1) > np.abs(m2):
            m = m1
        else:
            m = m2

        z1 = z0 - (n * p_nzi) / m
        err = np.abs(z1 - z0)
        z0 = z1
        i = i + 1
    return z1



def find_roots(polynom, starting_point):
    m = len(polynom) - 1
    roots = np.empty(m, dtype=complex)
    pm = polynom
    k = 0
    while m >= 2:
        zk = laguerre_method(pm, starting_point)
        if k == 0:
            roots[k] = zk
        else:
            div = np.poly1d([1, -roots[k - 1]])
            pmm, pmm2 = np.polydiv(pm, div)
            if len(pmm) == 2:
                pm = pmm
                break
            zz = laguerre_method(pmm, starting_point)
            zk = laguerre_method(polynom, zz)
            roots[k] = zk
            pm = pmm

        m = m - 1
        k = k + 1
    # roots of quadratic func
    quadr_roots = numpy.roots(pm)
    xnn = quadr_roots[len(quadr_roots) - 1]
    xn = laguerre_method(polynom, xnn)
    xnm1 = complex(xn.real, -xn.imag)
    roots[k] = xnm1
    roots[k + 1] = xnm1
    return roots


# starting guess point
x = complex(-3, 0)
print("\nZadanie numeryczne 13.O: \n")
# 12a
arr12a = np.array([243, -486, 783, -990, 558, -28, -72, 16], dtype=numpy.float64)
print("Rozwiazania rownania 12a: ")
print(find_roots(arr12a, x))
print("")
# 12b
arr12b = np.array([1, 1, 3, 2, -1, -3, -11, -8, -12, -4, 4], dtype=numpy.float64)
print("Rozwiazania rownania 12b: ")
print(find_roots(arr12b, x))
print("")
# 12c
arr12c = np.array([1, complex(0, 1), -1, complex(0, -1), 1])
print("Rozwiazania rownania 12c: ")
print(find_roots(arr12c, x))
