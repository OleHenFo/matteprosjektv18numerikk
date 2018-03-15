import scipy as sp
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


n = 20

density = 480
L = 2
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
                    [-12 / 17, 96 / 17, -156 / 17, 72 / 17]])
    A[0, 0:4] = B[0, :]
    A[n - 2, n - 4:n] = B[1, :]
    A[n - 1, n - 4:n] = B[2, :]
    return A


def solveForYc(n, c):
    A = csr_matrix(lagA(n))
    h = L / n
    const = ((h ** 4) / (E * I)) * f
    b = np.array([const] * n)
    y = spsolve(A, b)
    return y[c]


def fasit(n):
    svar = np.zeros(n)
    for i in range(1, n + 1):
        x = (L / n) * i
        svar[i-1] = ((f * (x ** 2)) / (24 * E * I)) * ((6 * (L ** 2)) - (4 * L * x) + x ** 2)
    return svar


def linsolv():
    n = 10
    list = []
    while n < 10 * 2 ** 11:
        list.append(solveForYc(n, n - 1))
        n *= 2
    return list


def findMaxError():
    list = linsolv()
    ans = fasit(10)[9]
    for i in range(0, len(list)):
        list[i] -= ans
    return list


# Oppgave 2
A = lagA(n)
print(A.todense())

# Oppgave 4 (Må fikses)
"""
def korrektY(f, E, I, x, L):
    return (f / (23 * E * I)) * x ** 2 * (x ** 2 - 4 * L * x + 6 * L ** 2)


x = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
y_e = len(x) * [0]

for i in range(0, len(x) - 1):
    y_e[i] = (korrektY(f, E, I, x[i], L))

A = Oppg2.lagA(len(y_e))

print("Prøver:")
fjerdeDerivertAvY = A.dot(y_e)
print("Done")
print(fjerdeDerivertAvY)
"""

# Oppgave 5
error = findMaxError()
print(error)