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
h = dr_vect
vec_l2 = L2_vect

# Ajuster une loi de puissance à toutes les valeurs (en utilisant np.polyfit avec logarithmes)
coefficients = np.polyfit(np.log(h[5:]), np.log(vec_l2[5:]), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(h[0])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(h, vec_l2, marker='o', color='b', label='Données numériques obtenues')
plt.plot(h, fit_function(h), linestyle='--', color='r', label='Régression en loi de puissance')

# Marquer la valeur extrapolée
#plt.scatter(h_values[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

# Ajouter des étiquettes et un titre au graphique
plt.title('Convergence d\'ordre 2\n de l\'erreur $L_2$ en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

plt.xlabel('Taille de maille $Δx$ (m)', fontsize=12, fontweight='bold')  
plt.ylabel('Erreur $L_2$ (mol/m$^3$)', fontsize=12, fontweight='bold')

# Rendre les axes plus gras
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)

# Placer les marques de coche à l'intérieur et les rendre un peu plus longues
plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)

# Afficher l'équation de la régression en loi de puissance
equation_text = f'$L_2 = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

# Déplacer la zone de texte
equation_text_obj.set_position((0.5, 0.4))


# Afficher le graphique
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig('L2_mms_dr.png',dpi=200)
plt.show()

# Analyse de la convergence temporelle
v_delta_t = dt_vect
vec_l2_t = L2_t_vect

# Ajuster une loi de puissance à toutes les valeurs (en utilisant np.polyfit avec logarithmes)
coefficients = np.polyfit(np.log(v_delta_t[5:]), np.log(vec_l2_t[5:]), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(v_delta_t[0])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(v_delta_t, vec_l2_t, marker='o', color='b', label='Données numériques obtenues')
plt.plot(v_delta_t, fit_function(v_delta_t), linestyle='--', color='r', label='Régression en loi de puissance')

# Marquer la valeur extrapolée
#plt.scatter(h_values[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

# Ajouter des étiquettes et un titre au graphique
plt.title('Convergence d\'ordre 1\n de l\'erreur $L_2$ en fonction de $Δt$',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

plt.xlabel('Taille d\'intervalle $Δt$ (s)', fontsize=12, fontweight='bold')  
plt.ylabel('Erreur $L_2$ (mol/m$^3$)', fontsize=12, fontweight='bold')

# Rendre les axes plus gras
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)

# Placer les marques de coche à l'intérieur et les rendre un peu plus longues
plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)

# Afficher l'équation de la régression en loi de puissance
equation_text = f'$L_2 = {np.exp(coefficients[1]):.2e} \\times Δt^{{{exponent:.4f}}}$'
equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

# Déplacer la zone de texte
equation_text_obj.set_position((0.5, 0.4))


# Afficher le graphique
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig('L2_mms_dt.png',dpi=200)
plt.show()
