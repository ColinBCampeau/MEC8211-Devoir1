"""

OUTIL NUMÉRIQUE DE SIMULATION DE LA DIFFUSION DU SEL DANS UN PILIER EN BÉTON POREUX

                            >FICHIER PRINCIPAL<

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

delta_t = prm.delta_t

vec_t = np.arange(0,prm.t_fin+delta_t,delta_t)

R = prm.R

vec_r = prm.vec_r
    

# Création des coefficients

B = np.zeros(N)
C = np.zeros(N)

A = -prm.D*delta_t/delta_r**2    # Coefficient devant C_i-1 (scalaire)
B[1:] = 1 + prm.D*delta_t/(vec_r[1:]*delta_r) + 2*prm.D*delta_t/delta_r**2 + prm.k*delta_t  # Coefficient devant C_i (vecteur)
C[1:] = -(prm.D*delta_t/(vec_r[1:]*delta_r)+prm.D*delta_t/delta_r**2)  # Coefficient devant C_i+1 (vecteur)

# Génération du vecteur concentration actuelle cas 1 obtenue de manière numérique
C_act_1, Cr0_1 = f_diff(A,B,C,1,"num",prm)

C_mms_1, Cr0_mms_1 = f_diff(A,B,C,1,"mms",prm)
    
    
#%%========================= CAS 2 ==========================%%#

# Création des coefficients

D = np.zeros(N)
F = np.zeros(N)

D[1:] = -prm.D*delta_t/delta_r**2+prm.D*delta_t/(2*vec_r[1:]*delta_r)    # Coefficient devant C_i-1 (vecteur)
E = 1 + 2*prm.D*delta_t/delta_r**2 + prm.k*delta_t  # Coefficient devant C_i (scalaire)
F[1:] = -(prm.D*delta_t/(2*vec_r[1:]*delta_r)+prm.D*delta_t/delta_r**2)  # Coefficient devant C_i+1 (vecteur)

# Génération du vecteur concentration actuelle cas 2 obtenue de manière numérique
C_act_2, Cr0_2 = f_diff(D,E,F,2,"num",prm)

C_mms_2, Cr0_mms_2 = f_diff(D,E,F,2,"mms",prm)

#%%========================= SOLUTION COMSOL ==========================%%#

df = pd.read_csv('Comsol_profil_81_1e5.csv', delimiter=";")
data = df["c"].to_numpy()
position = df["R"].to_numpy()
fct_comsol = sp.interpolate.interp1d(position,data,kind='quadratic')
sol_comsol = fct_comsol(vec_r)
plt.plot(vec_r,sol_comsol,label='Comsol',color='orange') 

dft = pd.read_csv('Sol_Comsol_temporel_1e5.csv', delimiter=";")
data_t = dft["Concentration"].to_numpy()
temps = dft["Time"].to_numpy()
fct_comsol_temporel = sp.interpolate.interp1d(temps,data_t,kind='quadratic')
sol_comsol_temporel = fct_comsol_temporel(vec_t)




#%%================= VÉRIFICATION DES RÉSULTATS NUMÉRIQUES ==================%%#

   
L1 = f_L1(C_act_2, sol_comsol) # Calcul de l'erreur L1 - Comsol
    
L2 = f_L2(C_act_2, sol_comsol) # Calcul de l'erreur L2 - Comsol
    
Linf = f_Linf(C_act_2, sol_comsol) # Calcul de l'erreur Linf - Comsol
    
# affichage des résultats
    
print('Nombre de points :',prm.N)
print('\u0394h = ',R/(prm.N-1))
print('\u0394t = ' + str("{:.1e}".format(prm.delta_t)))
print('Erreurs Comsol')
print('Erreur L1 : ',L1)
print('Erreur L2 : ',L2)
print('Erreur Linf : ',Linf)

    
# graphique profil de concentration analytique vs numérique

plt.scatter(vec_r,C_act_2,label='Numérique - ' +str(N)+' éléments')
plt.xlabel("Position radiale [m]")
plt.ylabel("Concentration [mol/m$^3$]")
plt.title("Profil de la concentration de sel en fonction de la position radiale \naprès " + str("{:.1e}".format(prm.t_fin)) + ' s ')
plt.legend()  
plt.grid()
plt.savefig("concentration_h.png", dpi=300,bbox_inches='tight')
plt.show()


plt.plot(vec_t,Cr0_2,label='Numérique - \u0394t = '+ str("{:.1e}".format(prm.delta_t)))
plt.plot(vec_t,sol_comsol_temporel,label='Comsol') 
plt.xlabel("Temps [s]")
plt.ylabel("Concentration [mol/m$^3$]")
plt.title("Profil de la concentration de sel en fonction du temps \nà R = 0 m")
plt.legend()  
plt.grid()
plt.savefig("concentration_t.png", dpi=300,bbox_inches='tight')


L2_t = f_L2(Cr0_2, sol_comsol_temporel)
print('Erreur L2 (temporel) :', L2_t) 






