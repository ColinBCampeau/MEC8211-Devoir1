"""

OUTIL NUM�RIQUE DE SIMULATION DE LA DIFFUSION DU SEL DANS UN PILIER EN B�TON POREUX

                        >FICHIER FONCTITON MMS<

AUTEUR: EROJ MOHAMMAD ISHOQ, COLIN BISSONNETTE-CAMPEAU, TRISTAN ANCEL-S�GUIN
CR�ATION: 2 MARS 2024
MISE � JOUR: 6 MARS 2024

"""


#%%===================== IMPORTATION DES MODULES ==========================%%#
from sympy import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import scipy as sp
try:
    from fonction import *
    from classe import *
except:
    pass

#%%===================== FONCTION MMS ==========================%%#

# Importation des symboles n�cessaires pour les calculs symboliques.
r, t, Ce, k, D, R = symbols('r t Ce k D R')

# D�finition de la fonction C qui d�pend de r et de t.
C = Function('C')(r, t)

# Expression de la concentration C en fonction de r et de t.
C = Ce*exp(k*t)*(r)**3

# Calcul de la d�riv�e de C par rapport au temps.
dCdt = diff(C, t)

# Calcul du laplacien de C en coordonn�es sph�riques (ici simplifi� comme grad2C).
grad2C = diff((r*diff(C,r)),r)/r

# D�finition du terme source S comme �tant proportionnel � C.
S = k*C

# �quation diff�rentielle r�gissant l'�volution de la concentration C.
C_r_t = dCdt - D * grad2C + S

# Calcul de la d�riv�e de C par rapport � r et �valuation � r=0.
dC_dr_r_0 = diff(C,r).subs(r, 0)

# �valuation de la concentration C � la fronti�re r=R.
C_R = C.subs(r, R)

# �valuation de la concentration C en r=0.
C_0 = C.subs(r, 0)

#%%===================== GRAPHIQUE ==========================%%#

# Vecteur de positions radiales.
r_vect = prm.vec_r

# Cr�ation d'un vecteur temporel.
t_vect = np.linspace(0, prm.t_fin, 5)

# Trac� de l'�volution de la concentration en fonction de la position radiale pour diff�rents temps.
for i in range(len(t_vect)):
    C_vect = prm.Ce*np.exp(prm.k*t_vect[i])*(r_vect)**3
    plt.plot(r_vect, C_vect, label="t = {:.2e} s".format(t_vect[i]))

plt.legend()
plt.title("�volution de la concentration en fonction de\nla position radiale pour diff�rent temps")
plt.xlabel("position radiale [m]")
plt.ylabel(r"$\hat{C}$ [mol/$m^3$]")
plt.grid()
plt.savefig("MMS_r.png", dpi=300,bbox_inches='tight')
plt.show()

# Cr�ation d'un nouveau vecteur temporel en prenant en compte une fonction delta t.
t_vect = np.arange(0, prm.t_fin + f_delta_t(prm), f_delta_t(prm))

# Cr�ation d'un vecteur de positions radiales.
r_vect = np.linspace(0, prm.R, 5)

# Trac� de l'�volution de la concentration en fonction du temps pour diff�rentes positions radiales.
for i in range(len(r_vect)):
    C_vect = prm.Ce*np.exp(prm.k*t_vect)*(r_vect[i])**3
    plt.plot(t_vect, C_vect, label = "r = " + str(r_vect[i])+" m")

plt.legend()
plt.title("�volution de la concentration en fonction du\ntemps pour diff�rent position radiale")
plt.xlabel("temps [s]")
plt.ylabel(r"$\hat{C}$ [mol/$m^3$]")
plt.grid()
plt.savefig("MMS_t.png", dpi=300,bbox_inches='tight')
plt.show()

# Affichage des r�sultats des calculs symboliques.
print("")
print("----------")
print("RESULTATS:")
print("----------")
print("C = " + str(C))
print("C_r_t = " + str(C_r_t))
print("dC_dr_r_0 = " + str(diff(C,r))+ " = " + str( dC_dr_r_0))
print("C_R = " + str(C_R))
print("----------")
print("")
