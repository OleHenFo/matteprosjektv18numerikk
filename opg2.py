import scipy as sp
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

density = 480
L = 10
E = 1.3 * (10 ** 10)
w = 0.3
d = 0.03
g = 9.81
I = (w * (d ** 3)) / 12
f = -density * w * d * g


def lagA(n):
    e = sp.ones(n)
    A = spdiags([e, -4 * e, 6 * e, -4 * e, e], [-2, -1, 0, 1, 2], n, n)
    A = lil_matrix(A)
    B = csr_matrix([[16, -9, 8 / 3, -1 / 4],
                    [16 / 17, -60 / 17, 72 / 17, -28 / 17],
                    [-12 / 17, 96 / 17, -156 / 16, 72 / 17]])
    A[0, 0:4] = B[0, :]
    A[n - 2, n - 4:n] = B[1, :]
    A[n - 1, n - 4:n] = B[2, :]
    return A


def solveForYc(n, c):
    A = csr_matrix(lagA(n))
    h = L / n
    print(f)

    const = ((h ** 4) / (E * I)) * f
    b = sp.asmatrix([const] * n).T
    # b[n-1] = b[n-1]*50

    y = spsolve(A, b)
    return y



for i in range(0, 20):
    x = i/10
    print(((f*(x**2))/(24*E*I))*((6*(L**2))-(4*L*x)+x**2))
print(solveForYc(10, 0))
