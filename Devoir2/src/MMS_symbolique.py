"""

OUTIL NUMÉRIQUE DE SIMULATION DE LA DIFFUSION DU SEL DANS UN PILIER EN BÉTON POREUX

                        >FICHIER FONCTITON MMS<

AUTEUR: EROJ MOHAMMAD ISHOQ, COLIN BISSONNETTE-CAMPEAU, TRISTAN ANCEL-SÉGUIN
CRÉATION: 2 MARS 2024
MISE À JOUR: 6 MARS 2024

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

# Importation des symboles nécessaires pour les calculs symboliques.
r, t, Ce, k, D, R = symbols('r t Ce k D R')

# Définition de la fonction C qui dépend de r et de t.
C = Function('C')(r, t)

# Expression de la concentration C en fonction de r et de t.
C = Ce*exp(k*t)*(r)**3

# Calcul de la dérivée de C par rapport au temps.
dCdt = diff(C, t)

# Calcul du laplacien de C en coordonnées sphériques (ici simplifié comme grad2C).
grad2C = diff((r*diff(C,r)),r)/r

# Définition du terme source S comme étant proportionnel à C.
S = k*C

# Équation différentielle régissant l'évolution de la concentration C.
C_r_t = dCdt - D * grad2C + S

# Calcul de la dérivée de C par rapport à r et évaluation à r=0.
dC_dr_r_0 = diff(C,r).subs(r, 0)

# Évaluation de la concentration C à la frontière r=R.
C_R = C.subs(r, R)

# Évaluation de la concentration C en r=0.
C_0 = C.subs(r, 0)

#%%===================== GRAPHIQUE ==========================%%#

# Vecteur de positions radiales.
r_vect = prm.vec_r

# Création d'un vecteur temporel.
t_vect = np.linspace(0, prm.t_fin, 5)

# Tracé de l'évolution de la concentration en fonction de la position radiale pour différents temps.
for i in range(len(t_vect)):
    C_vect = prm.Ce*np.exp(prm.k*t_vect[i])*(r_vect)**3
    plt.plot(r_vect, C_vect, label="t = {:.2e} s".format(t_vect[i]))

plt.legend()
plt.title("Évolution de la concentration en fonction de\nla position radiale pour différent temps")
plt.xlabel("position radiale [m]")
plt.ylabel(r"$\hat{C}$ [mol/$m^3$]")
plt.grid()
plt.savefig("MMS_r.png", dpi=300,bbox_inches='tight')
plt.show()

# Création d'un nouveau vecteur temporel en prenant en compte une fonction delta t.
t_vect = np.arange(0, prm.t_fin + f_delta_t(prm), f_delta_t(prm))

# Création d'un vecteur de positions radiales.
r_vect = np.linspace(0, prm.R, 5)

# Tracé de l'évolution de la concentration en fonction du temps pour différentes positions radiales.
for i in range(len(r_vect)):
    C_vect = prm.Ce*np.exp(prm.k*t_vect)*(r_vect[i])**3
    plt.plot(t_vect, C_vect, label = "r = " + str(r_vect[i])+" m")

plt.legend()
plt.title("Évolution de la concentration en fonction du\ntemps pour différent position radiale")
plt.xlabel("temps [s]")
plt.ylabel(r"$\hat{C}$ [mol/$m^3$]")
plt.grid()
plt.savefig("MMS_t.png", dpi=300,bbox_inches='tight')
plt.show()

# Affichage des résultats des calculs symboliques.
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
