import numpy as np
from scipy import sparse

# tworze macierz z zadania:
Aa = np.zeros((127, 127))
ee = np.ones(127)
xx = np.zeros(127)  #poczatkowe wartosci dla x-ow

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
                x[i] = d/a[j][j] #czyli 4 w tym przypadku
    return x

print("Wyniki po sparse gauss seidel function:")
print(sparse_gauss_seidel(Aa,xx,ee, 5))
