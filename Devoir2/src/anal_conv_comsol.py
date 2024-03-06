"""

OUTIL NUMÉRIQUE DE SIMULATION DE LA DIFFUSION DU SEL DANS UN PILIER EN BÉTON POREUX

                        >FICHIER ANALYSE CONVERGENCE - COMSOL<

AUTEUR: EROJ MOHAMMAD ISHOQ, COLIN BISSONNETTE-CAMPEAU, TRISTAN ANCEL-SÉGUIN
CRÉATION: 2 MARS 2024
MISE À JOUR: 6 MARS 2024

"""

    
#%%===================== IMPORTATION DES MODULES ==========================%%#


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



#%%============= ANALYSE CONVERGENCE - SPATIALE ===============%%#
N_vect = np.array([2,5,10,20,40,80])
dr_vect = 0.5/N_vect
L2_vect = np.array([0.4267334,0.08151597,0.02288,0.0058953,0.001480299,0.0003631455])

dr_reg = dr_vect[3:]
L2_reg = L2_vect[3:]
log_dr_reg = np.log(dr_reg)
log_L2_reg = np.log(L2_reg)
pente_reg, ordonne_reg, _, _, _ = linregress(log_dr_reg, log_L2_reg)
f_reg = np.exp(ordonne_reg) * dr_vect**pente_reg
plt.plot(dr_vect, L2_vect, 'o', label='MMS', markersize=8)
plt.plot(dr_vect, f_reg, 'r--', label = r'Erreur $L_2 = {:.2e} \cdot \Delta r^{{{:.2f}}}$'.format(np.exp(ordonne_reg), pente_reg))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta r$ $[m]$')
plt.ylabel(r'Erreur $L_2$ $[mol/m^3]$')
plt.title(r'Erreur de la norme $L_2$ en fonction de la taille des éléments')
plt.legend()
plt.grid()
plt.savefig("L2_comsol_dr.png", dpi=300,bbox_inches='tight')
plt.show()

#%%============= ANALYSE CONVERGENCE - TEMPORELLE ===============%%#
dt_vect = np.array([1e5,5e5,1e6,2e6,5e6,1e7,1e8])
L2_t_vect = np.array([0.0006732,0.00186467,0.003429,0.00655844,0.01577,0.030529,0.22692])

dt_reg = dt_vect[3:-2]
L2_t_reg = L2_t_vect[3:-2]
log_dt_reg = np.log(dt_reg)
log_L2_t_reg = np.log(L2_t_reg)
pente_reg, ordonne_reg, _, _, _ = linregress(log_dt_reg, log_L2_t_reg)
f_reg = np.exp(ordonne_reg) * dt_vect**pente_reg
plt.plot(dt_vect, L2_t_vect, 'o', label='MMS', markersize=8)
plt.plot(dt_vect, f_reg, 'r--', label= r'Erreur $L_2 = {:.2e} \cdot \Delta t^{{{:.2f}}}$'.format(np.exp(ordonne_reg), pente_reg))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta t$ [s]')
plt.ylabel(r'Erreur $L_2$ $[mol/m^3]$')
plt.title(r'Erreur de la norme $L_2$ en fonction du pas de temps')
plt.legend()
plt.grid()
plt.savefig("L2_comsol_dt.png", dpi=300,bbox_inches='tight')
plt.show()
