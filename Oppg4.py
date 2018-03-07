density = 480
L=2
E = 1.3 *10**10
w = 0.3
d = 0.03
g = 9.81
I = (w*(d**3))/12
f = -density*w*d*g

def korrektY(f,E,I,x,L):
    return (f/(23*E*I))*x**2*(x**2-4*L*x+6*L**2)

x = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

for i in range(0, len(x)-1):
    print(korrektY(f, E, I, x[i], L))