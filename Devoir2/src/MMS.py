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



#t_vect = np.array([0, dt*2])



for i in range(len(t_vect)):
    S_vect = np.array([])
    C_vect = np.array([])
    for j in range(len(vec_r)):
        C_r_t =  -9*C0*D*vec_r[j]*np.exp(k*t_vect[i]) + 2*C0*k*vec_r[j]**3*np.exp(k*t_vect[i])
        S_vect = np.append(S_vect, C_r_t)
        
        C = C0*np.exp(k*t_vect[i])*vec_r[j]**3
        
        C_vect = np.append(C_vect, C)
        


    