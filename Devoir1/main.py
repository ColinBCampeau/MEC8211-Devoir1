"""

OUTIL NUMÉRIQUE DE SIMULATION DE LA DIFFUSION DU SEL DANS UN PILIER EN BÉTON POREUX

                            >FICHIER PRINCIPAL<

AUTEUR: EROJ MOHAMMAD ISHOQ, COLIN BISSONNETTE-CAMPEAU, TRISTAN ANCEL-SÉGUIN
CRÉATION: 04 FÉVRIER 2024
MISE À JOUR: 11 FÉVRIER 2024

"""

    
#%%===================== IMPORTATION DES MODULES ==========================%%#

import numpy as np
import matplotlib.pyplot as plt
try:
    from fonction import *
    from classe import *
except:
    pass

#%%========================= CAS 1 ==========================%%#

N = prm.N

delta_r = prm.delta_r

delta_t = prm.delta_t

R = prm.R

vec_r = prm.vec_r
    

# Création des coefficients

B = np.zeros(N)
C = np.zeros(N)

A = -prm.D*delta_t/delta_r**2    # Coefficient devant C_i-1 (scalaire)
B[1:] = 1 + prm.D*delta_t/(vec_r[1:]*delta_r) + 2*prm.D*delta_t/delta_r**2 #+ k*delta_t  # Coefficient devant C_i (vecteur)
C[1:] = -(prm.D*delta_t/(vec_r[1:]*delta_r)+prm.D*delta_t/delta_r**2)  # Coefficient devant C_i+1 (vecteur)

# Génération du vecteur concentration actuelle cas 1 obtenue de manière numérique
C_act_1 = f_diff(A,B,C,1,prm)
    
    
#%%========================= CAS 2 ==========================%%#

# Création des coefficients

D = np.zeros(N)
F = np.zeros(N)

D[1:] = -prm.D*delta_t/delta_r**2+prm.D*delta_t/(2*vec_r[1:]*delta_r)    # Coefficient devant C_i-1 (vecteur)
E = 1 + 2*prm.D*delta_t/delta_r**2 #+ k*delta_t  # Coefficient devant C_i (scalaire)
F[1:] = -(prm.D*delta_t/(2*vec_r[1:]*delta_r)+prm.D*delta_t/delta_r**2)  # Coefficient devant C_i+1 (vecteur)

# Génération du vecteur concentration actuelle cas 2 obtenue de manière numérique
C_act_2 = f_diff(D,E,F,2,prm)

#%%========================= SOLUTION ANALYTIQUE ==========================%%#

C_anal = sol_analytique(vec_r,prm)


#%%================= VÉRIFICATION DES RÉSULTATS NUMÉRIQUES ==================%%#

# Concatenation des vecteurs C_act_1 et C_act_2 dans un matrice
C_act_vect = np.array([C_act_1, C_act_2])

for i in range (len(C_act_vect)):
    
    L1 = f_L1(C_act_vect[i], C_anal) # Calcul de l'erreur L1
    
    L2 = f_L2(C_act_vect[i], C_anal) # Calcul de l'erreur L2
    
    Linf = f_Linf(C_act_vect[i], C_anal) # Calcul de l'erreur Linf
    
    # affichage des résultats
    print('Cas '+ str(i+1))
    print('Nombre de points :',prm.N)
    print('Erreur L1 : ',L1)
    print('Erreur L2 : ',L2)
    print('Erreur Linf : ',Linf)
    
# graphique profil de concentration analytique vs numérique
plt.scatter(vec_r,C_anal,label='Cas analytique',color='g')   
plt.plot(vec_r,C_act_1,label='Cas 1 - numérique')
plt.plot(vec_r,C_act_2,label='Cas 2 - numérique')
plt.xlabel("Position radiale [m]")
plt.ylabel("Concentration [mol/m$^3$]")
plt.title("Profil de la concentration de sel dans un pilier en \nbéton poreux en fonction de la position radiale")
plt.legend()  
plt.grid()
plt.savefig("concentration.png", dpi=300,bbox_inches='tight')
plt.show()

# graphique de l'erreur entre solution numérique et analytique
plt.plot(vec_r,abs(C_act_1-C_anal),label='Erreur cas 1')
plt.plot(vec_r,abs(C_act_2-C_anal),label='Erreur cas 2')
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Position radiale [m]")
plt.ylabel("Erreur")
plt.title("Évolution de l'erreur de la solution numérique par rapport\nà la solution analytique en fonction de la position radiale")
plt.legend()
plt.grid()
plt.savefig("erreur.png", dpi=300,bbox_inches='tight')
plt.show()





