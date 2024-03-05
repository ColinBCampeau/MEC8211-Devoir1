"""

OUTIL NUMÉRIQUE DE SIMULATION DE LA DIFFUSION DU SEL DANS UN PILIER EN BÉTON POREUX

                            >FICHIER VÉRIFICATION MMS<

AUTEUR: EROJ MOHAMMAD ISHOQ, COLIN BISSONNETTE-CAMPEAU, TRISTAN ANCEL-SÉGUIN
CRÉATION: 28 FÉVRIER 2024
MISE À JOUR: 28 FÉVRIER 2024

"""

    
#%%===================== IMPORTATION DES MODULES ==========================%%#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
try:
    from fonction import *
    from classe import *
except:
    pass

#%%========================= CAS 1 ==========================%%#

N = prm.N

delta_r = prm.delta_r

T = prm.t_fin

dt = prm.delta_t

t_vect = np.arange(0, T+dt,dt)

R = prm.R

vec_r = prm.vec_r

C0 = prm.Ce

k = prm.k

D = prm.D


C_vect = np.array([])

for i in range(len(t_vect)):
    S_vect = np.array([])
   
    for j in range(len(vec_r)):
        #C_r_t =  -9*C0*D*vec_r[j]*np.exp(k*t_vect[i]) + 2*C0*k*vec_r[j]**3*np.exp(k*t_vect[i])
        C_r_t = -9*C0*D*vec_r[j]*np.exp(t) + C0*k*vec_r[j]**3*exp(t) + C0*r**3*exp(t)
        
        S_vect = np.append(S_vect, C_r_t)


        
        

C = C0*np.exp(k*t_vect[-1])*vec_r**3

plt.plot(vec_r, C)
plt.show()


        
print(0.25*C0*np.exp(k*t_vect[-1]))

    