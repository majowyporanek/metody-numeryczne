import cmath

import numpy as np
from numpy.polynomial.polynomial import polyval

# x^n + x^x-1.....+x^0
arr_a = np.array([243, -486, 783, -990, 558, -28, -72, 16], dtype=complex)
arr_b = np.array([1, 1, 3, 2, -1, -3, -11, -8, -12, -4, -4], dtype=complex)
arr_c = np.array([1, complex(0, 1), -1, complex(0, -1), 1], dtype=complex)


def find_value(polyarr, x):
    poly_flip = np.flip(polyarr)
    return polyval(x, poly_flip)


def find_derivative(polyarr, m):
    return np.polyder(polyarr, m);


def poly_deflation(root, oldPol):
    divider = np.array([1, -root], dtype=complex)
    q, r = np.polydiv(oldPol, divider)
    return np.array(q, dtype=complex)


def LaguerreMethodRealAndComplex(start, pol):
    z = start
    fun, der, secDer, denominator = 0, 0, 0, 0
    tmp = 100

    while abs(np.polyval(pol, z) - np.polyval(pol, tmp)) > 1e-13:
        tmp = z
        fun = np.polyval(pol, z) * (len(pol) - 1)
        der = np.polyval(np.polyder(pol), z)
        secDer = np.polyval(np.polyder(np.polyder(pol)), z)
        denominator = ((der ** 2 * ((len(pol) - 1) - 1)
                        - (fun * secDer)) * ((len(pol) - 1) - 1)) ** 0.5
        denominator = der - denominator if abs(der - denominator) > abs(der + denominator) else der + denominator
        z = z - (fun / denominator)

    return z


def calc_Roots(polynomial):
    start_point = complex(1, 0)
    n = len(polynomial) - 1
    roots = np.empty(n, dtype=complex)
    roots[0] = LaguerreMethodRealAndComplex(start_point, polynomial)
    poly = polynomial
    i = 0
    while n > 3:
        deflated_poly = poly_deflation(roots[i], poly)
        z1 = LaguerreMethodRealAndComplex(start_point, deflated_poly)
        z = LaguerreMethodRealAndComplex(z1, poly)
        i = i + 1
        roots[i] = z
        poly = deflated_poly
        n = len(poly) - 1

    poly = poly_deflation(roots[i], poly)
    a = poly[0]
    b = poly[1]
    c = poly[2]

    d = (b ** 2) - (4 * a * c)

    # find two solutions
    sol1 = (-b - cmath.sqrt(d)) / (2 * a)
    sol2 = (-b + cmath.sqrt(d)) / (2 * a)

    i = i + 1
    roots[i] = sol1
    i = i + 1
    roots[i] = sol2

    for j in range(len(roots)):
        print(str(j + 1) + ": " + str(roots[j]))


print("12 a wyniki: ")
calc_Roots(arr_a)

print(" ")
print("12 b wyniki: ")
calc_Roots(arr_b)

print(" ")
print("12 c wyniki: ")
calc_Roots(arr_c)
