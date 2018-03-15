from Oppg2 import fasit, solveForYc
import numpy as np


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


print(findMaxError())
