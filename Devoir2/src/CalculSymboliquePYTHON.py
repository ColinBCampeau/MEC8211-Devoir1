
from sympy import *

r, t, T0, C0, R, k, D, k= symbols('r t T0 C0 R k D k')

C = Function('C')(r, t) # D√©finition de la fonction symbolique et de ses variables


C = C0*exp(k*t)*(r)**3

#C = exp(r**2)*exp(k*t)


dCdt = diff(C, t)

grad2C = diff((r*diff(C,r)),r)/r

S = k*C

C_r_t = dCdt - D * grad2C + S


dC_dr_r_0 = diff(C,r).subs(r, 0)

C_R = C.subs(r, R)

C_0 = C.subs(r, 0)


print("")
print("----------")
print("RESULTATS:")
print("----------")
# print("dT/dt:", dTdt) # D√©rvi√©e temporelle de la solution choisie
# print("d^2T/dx^2:", d2Tdx2) # Laplacien de la solution choisie
# print("dT/dx at x=0:", dTdx_at_x0) # Si on souhaite tester une possible condition de Neumann (condition de flux), permet d'obtenir l'expression de la condition de Neumann √  x = 0.
# print("Terme source √  ajouter:", dTdt -alpha*d2Tdx2) # obtention de l'expression du terme source √  rajouter
print("C = ", C)
print("C_r_t = ", C_r_t)
print("dC_dr_r_0 = ", dC_dr_r_0)
print("C_R = ", C_R)
print("----------")
print("")