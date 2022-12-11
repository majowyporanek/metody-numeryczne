import numpy
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg
from numpy import linalg as LA

# tworze macierz z zadania:
Aa = np.zeros((127, 127))
ee = np.ones(127)
xx = np.zeros(127)  # poczatkowe wartosci dla x-ow

np.fill_diagonal(Aa, 4)

for k in range(len(Aa) - 1):
    Aa[k][k + 1] = 1
    Aa[k + 1][k] = 1

for k in range(len(Aa) - 4):
    Aa[k][k + 4] = 1
    Aa[k + 4][k] = 1


def sparse_gauss_seidel(a, x, e, iterat):
    a_csr = sparse.csr_matrix(a)
    for m in range(iterat):
        for i in range(0, a_csr.shape[0]):
            row = a_csr.getrow(i).indices.tolist()
            d = e[i]
            for j in range(len(row)):
                if row[j] != i:
                    d -= a[i][row[j]] * x[row[j]]
                x[i] = d / a[j][j]
    return x

print("Rozwiazania po 5 iteracjach metodą gauss-seidel:")
print(sparse_gauss_seidel(Aa,xx,ee,5))
# Metoda gradientów sprzężonych
print(" ")
print("Rozwiazania po 5 iteracjach metodą gradientow sprzezonych:")
P = csc_matrix(Aa)
X = cg(P, ee, maxiter=5)

print(X[0])

print(" ")
print("Normy GAUSS-SEIDEL")
for i in range(1,16):
    xx = np.zeros(127)
    x = sparse_gauss_seidel(Aa,xx,ee,i)
    xx = np.zeros(127)
    x2 = sparse_gauss_seidel(Aa,xx,ee,i+1)
    tonorm = numpy.subtract(x,x2)
    print(LA.norm(tonorm))

print(" ")
print("Normy METODA GRADIENTÓW SPRZĘŻONYCH:")
for i in range(0,14):
    x = cg(P,ee,maxiter=i)
    y = cg(P,ee,maxiter=i+1)
    tonorm = numpy.subtract(x[0],y[0])
    print(LA.norm(tonorm))

