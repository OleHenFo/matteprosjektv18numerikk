import scipy as sp
import numpy as np
import matplotlib.pyplot as pl
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


n = 10

density = 480
L = 2
h = L/n
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


def solveForYc(n):
    A = csr_matrix(lagA(n))
    h = L / n
    const = ((h ** 4) / (E * I)) * f
    b = np.array([const] * n)
    y = spsolve(A, b)
    return y


def lagFasit(n):
    svar = np.zeros(n)
    for i in range(1, n+1):
        x = (L / n) * i
        svar[i-1] = ((f * (x ** 2)) / (24 * E * I)) * ((6 * (L ** 2)) - (4 * L * x) + x ** 2)
    return svar


def linsolv():
    nyN = 10
    list = []
    while nyN < 10 * 2 ** 11:
        list.append(solveForYc(nyN)[nyN - 1])
        nyN *= 2
    return list


def findMaxError():
    list = linsolv()
    ans = lagFasit(10)[9]
    for i in range(0, len(list)):
        list[i] -= ans
    maxIndex = np.argmax(np.abs(list))
    max = np.max(np.abs(list))
    return max, maxIndex, list


def getConditionNumber():
    nyN = 10
    condlist = []
    while nyN < 5000:
        m = lagA(nyN).todense()
        condNr = np.linalg.cond(m,1)
        condlist.append(int(condNr))
        nyN *= 2
    return condlist


def withADudeOnIt():
    myN = 1280 # Using n = 20, to correctly add the force of the person, while keeping the condition number lowest possible
    A = csr_matrix(lagA(myN))
    myH = L / myN
    const = ((myH ** 4) / (E * I))
    b = np.array([f] * myN)
    edge = 1.7
    for i in range(myN):
        index = i*myH
        if(index>edge and index<L):
            b[i] -= g*50/0.3
        b[i] *= const
    y = spsolve(A, b)
    return y


def withASineOnItNum(nn):
    p = 100
    A = csr_matrix(lagA(nn))
    hh = L/nn
    const = ((hh ** 4) / (E * I))
    b = np.array([f]*nn)
    for i in range(1, nn+1):
        x = (L / nn) * i
        b[i-1] -= (p * g * np.sin((np.pi/L)*x))
        b[i-1] = b[i-1] * const
    y = spsolve(A, b)
    return y


def withASineOnItFasit(nn):
    p = 100
    y = np.zeros(nn)
    for i in range(1, nn+1):
        x = (L / nn) * i
        y[i-1] = ((f/(24*E*I))*x**2) * (x**2 - 4*L*x + 6*L**2) - ((p*g*L)/(E*I*np.pi))*(((L**3)/(np.pi**3))*np.sin((np.pi/L)*x) - (x**3)/6 + (L/2)*x**2 - ((L**2)/(np.pi**2))*x)
    return y


def errorInSine(nn):
    numer = withASineOnItNum(nn)
    fasit = withASineOnItFasit(nn)
    error = np.abs(fasit - numer)
    return error


def allErrorsInSine():
    nyNN = 10
    errorList = []
    while nyNN<22000:
        errorList.append(errorInSine(nyNN)[nyNN-1])
        nyNN *=2
    return errorList

def teoretiskFeil():
    minN = 10
    errorList =[]
    while minN<22000:
        errorList.append((L**2)/minN**2)
        minN *=2

    print(errorList)
    return errorList

def finnFinN():
    errors = allErrorsInSine()
    minIndex = np.argmin(np.abs(errors))
    return (2**minIndex)*10


# Oppgave 2
print("Oppgave 2: ")
A = lagA(n)
print(A.todense())
print("--------------------------------")
print("")

# Oppgave 3
print("Oppgave 3: ")
print("Ikke-eksakt y-vektor:")
y_c = solveForYc(10)
print(y_c)
print()
print("--------------------------------")
print("")

# Oppgave 4 (Må fikses)
print("Oppgave 4: ")
xx = np.arange(0,n,1)
yy = lagFasit(n)
pl.plot(xx, yy)
pl.plot(y_c)
pl.axis([0, n, -0.2, 0.2])
pl.title("With nothing on it")
pl.show()

y_e = lagFasit(n)
print("Eksakt y-vektor:")
print(y_e)
print()

print("Nummerisk fjerdederivert (1/h^4)*Ay_e:")
nummeriskFjerdederivert = 1/h**4 * np.dot(A.todense(), y_e)
print(nummeriskFjerdederivert)
print()

print("Eksakt fjerdederivert f/EI:")
eksaktFjerdederivert = [f/(E*I)]*10
print(eksaktFjerdederivert)
print()

print("Avstand mellom eksakt og nummerisk")
feilVektor = nummeriskFjerdederivert - eksaktFjerdederivert
print(feilVektor)
print()

print("Forward error:")
forwardError = np.max(np.abs(feilVektor))
print(forwardError)
print("Relativ error:")
relForErr = forwardError/(np.max(np.abs(eksaktFjerdederivert)))
print(relForErr)
backwardsError = 2**-52
print(relForErr/backwardsError)

print("--------------------------------")
print("")

# Oppgave 5
print("Oppgave 5: ")
error, errorIndex, errorList = findMaxError()
condList = getConditionNumber()
print("List of errors at x= L with n multiplying by two for each element:")
print(errorList)
print ("The largest error is "+str(error)+", which is number "+ str(errorIndex)+" in our list. This makes sense when we "
                                                                          "look at the condition number for the different matrices : \n    ")
print(condList)
print("Note that we stopped at n = 1280 as computations take too long for larger matrices ")
print("--------------------------------")
print("")


# Oppgave 6
print("Oppgave 6:")
nn = 10
xx = np.arange(0,nn,1)
yy = withASineOnItFasit(nn)
yy2 = withASineOnItNum(nn)
pl.plot(xx, yy, 'b')
pl.plot(yy2, 'r')
pl.axis([0, nn, -0.7, 0.7])
pl.title("With a Sine on it")
pl.show()
print("Error")
condAmach= np.zeros(len(condList))
for i in range (len(condList)):
    condAmach[i] = condList[i]*2**(-52)
yy = allErrorsInSine()
xx = (2**np.arange(0,len(allErrorsInSine()),1))*10
xx1 = (2**np.arange(0,len(allErrorsInSine())-3,1))*10

print("CondAmach: "+str(condAmach))
print(yy)
pl.loglog(xx, yy)
pl.loglog(xx1, condAmach, 'g')
pl.loglog(xx, teoretiskFeil(), 'r')
pl.title("With a Sine on it (Error)")
pl.show()
print("Finn fin N:")
print(finnFinN())
print("--------------------------------")
print("")

# Oppgave 7
print("Oppgave 7:")
print("Position of the board with a man on top of it:")
print(withADudeOnIt())
xx = np.arange(0, 1280 ,1)
yy = withADudeOnIt()
pl.plot(xx, yy)
pl.axis([0, 1280, -0.7, 0.7])
pl.title("With a Dude on it")
pl.show()
print("--------------------------------")
print("")

