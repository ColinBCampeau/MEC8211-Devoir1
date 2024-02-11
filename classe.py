"""

OUTIL NUMÉRIQUE DE SIMULATION DE LA DIFFUSION DU SEL DANS UN PILIER EN BÉTON POREUX
                            >FICHIER CLASSE<

AUTEUR: EROJ MOHAMMAD ISHOQ, COLIN BISSONNETTE-CAMPEAU, TRISTAN SÉGUIN
CRÉATION: 04 FÉVRIER 2024
MISE À JOURS: 11 FÉVRIER 2024

"""

#%%===================== IMPORTATION DES MODULES ==========================%%#

import numpy as np

#%%===================== Classe paramètre ==========================%%#
class prm:
    
    R = 0.5 # Rayon [m]
    
    D = 10e-10 # Coefficient de diffusion [-]
    
    Ce = 12 # Concentration de l'eau [mol/m^3]
    
    k = 4e-9 # Coefficnent de réaction[s^-1]
    
    S = 8e-9 # Coefficient terme source constant [mol/m^3/s]
    
    N = 6 # Nombre de points [-]
    
    delta_r = R/(N-1) # Intervalle spatial [m]
    
    delta_t = 10000000 # Intervalle temporel [s]
    
    # Création du vecteur de position
    vec_r = np.zeros(N)
    for i in range(len(vec_r)):
        vec_r[i] = delta_r*i
    
    