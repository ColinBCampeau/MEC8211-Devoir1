"""

OUTIL NUMÉRIQUE DE SIMULATION DE LA DIFFUSION DU SEL DANS UN PILIER EN BÉTON POREUX

                        >FICHIER ANALYSE CONVERGENCE - MMS<

AUTEUR: EROJ MOHAMMAD ISHOQ, COLIN BISSONNETTE-CAMPEAU, TRISTAN ANCEL-SÉGUIN
CRÉATION: 6 MARS 2024
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

#%%============= ANALYSE CONVERGENCE - MMS ===============%%#

# Initialisation des paramètres de la simulation
prm.N = 81  # Définit le nombre de points dans le domaine spatial

# Génération des vecteurs pour les tests de convergence
N_vect = np.append([3,5,9], np.arange(20, (prm.N-1)*2+20, 20)[:4]+1)  # Crée un vecteur contenant différentes tailles de grille

# Initialisation des vecteurs pour stocker les résultats
dr_vect = np.array([])  # Vecteur pour les distances entre les points de grille
L2_vect = np.array([])  # Vecteur pour les erreurs L2 spatiales
L2_t_vect = np.array([])  # Vecteur pour les erreurs L2 temporelles
dt_vect = np.array([])  # Vecteur pour les pas de temps

# Boucle sur les différentes tailles de grille pour effectuer les simulations
for i in range(len(N_vect)):
    
    prm.N = N_vect[i]  # Mise à jour de la taille de la grille
    N = prm.N
    prm.delta_r = f_delta_r(prm)  # Calcul du pas spatial
    delta_r = prm.delta_r
    prm.delta_t = f_delta_t(prm)  # Calcul du pas temporel
    delta_t = prm.delta_t
    prm.vec_r = np.arange(0, prm.R+delta_r, delta_r)  # Création du vecteur des positions radiales
    vec_r = prm.vec_r
    vec_t = np.arange(0, prm.t_fin+prm.delta_t, prm.delta_t)  # Création du vecteur temporel
    
    # Création des coefficients pour l'équation différentielle
    D = np.zeros(N)
    F = np.zeros(N)
    D[1:] = -prm.D*delta_t/delta_r**2+prm.D*delta_t/(2*vec_r[1:]*delta_r)  # Coefficient devant C_i-1
    E = 1 + 2*prm.D*delta_t/delta_r**2 + prm.k*delta_t  # Coefficient devant C_i
    F[1:] = -(prm.D*delta_t/(2*vec_r[1:]*delta_r)+prm.D*delta_t/delta_r**2)  # Coefficient devant C_i+1

    # Simulation et calcul des erreurs L2
    C_mms_2, Cr0_mms_2 = f_diff(D,E,F,2,"mms",prm)  # Simulation de la concentration
    C_MMS = prm.Ce*np.exp(prm.k*vec_t[-1])*vec_r**3  # Solution analytique spatiale
    C_MMS_t = prm.Ce*np.exp(prm.k*vec_t)*vec_r[int((prm.N-1)/2)]**3  # Solution analytique temporelle
    
    # Stockage des résultats
    dr_vect = np.append(dr_vect, prm.delta_r)
    L2_vect = np.append(L2_vect, f_L2(C_mms_2, C_MMS))
    L2_t_vect = np.append(L2_t_vect,f_L2(Cr0_mms_2[:-2], C_MMS_t[:-2]))
    dt_vect = np.append(dt_vect, prm.delta_t)

# Analyse de la convergence spatiale
dr_reg = dr_vect[5:]
L2_reg = L2_vect[5:]
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
plt.savefig("L2_mms_dr.png", dpi=300,bbox_inches='tight')
plt.show()

# Analyse de la convergence temporelle
dt_reg = dt_vect[5:]
L2_t_reg = L2_t_vect[5:]
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
plt.savefig("L2_mms_dt.png", dpi=300,bbox_inches='tight')
plt.show()
