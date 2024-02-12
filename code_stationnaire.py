# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

# Paramètres

R = 0.5 # Rayon
N = 5 # Nombre de points
delta_r = R/(N-1) # Intervalle spatial
delta_t = 1 # Intervalle temporel

class prm:
    R = 0.5 # Rayon
    D = 10e-10 # Coefficient de diffusion
    Ce = 12 # Concentration de l'eau
    k = 4e-9 # Coefficnent de réaction
    S = 8e-9 # Coefficient terme source constant

# Création du vecteur de position
vec_r = np.zeros(N)
for i in range(len(vec_r)):
    vec_r[i] = delta_r*i
    
    
 # Cas 1

# Création des coefficients

B = np.zeros(N)
C = np.zeros(N)

A = prm.D*delta_t/delta_r**2    # Coefficient devant C_i-1 (scalaire)
B[1:] = 0 - (prm.D*delta_t/(vec_r[1:]*delta_r) + 2*prm.D*delta_t/delta_r**2) #+ k*delta_t  # Coefficient devant C_i (vecteur)
C[1:] = (prm.D*delta_t/(vec_r[1:]*delta_r)+prm.D*delta_t/delta_r**2)  # Coefficient devant C_i+1 (vecteur)


# Initialisation des vecteurs de concentration
C_ini = np.ones(N)*prm.S
C_act = np.zeros(N)
mat_C = np.zeros([N,N])
crit = 1


# Condition de Neumann à i = 0

mat_C[0,0] = 1
mat_C[0,1] = -1
#mat_C[0,2] = -1
C_ini[0] = 0

# Condition de Dirichlet à i = N

mat_C[-1,-1] = 1
C_ini[-1] = prm.Ce


# Algorithme de différences finies

b = C_ini.copy()
c = 0


for i in range(1,len(C_ini)-1):
    
    mat_C[i,i-1] = A
    mat_C[i,i] = B[i]
    mat_C[i,i+1] = C[i]
    
    b[i] = C_ini[i] 
    
# Résolution du système matriciel
C_act = np.linalg.solve(mat_C,b)
    
    
 # Cas 2

# Création des coefficients

D = np.zeros(N)
F = np.zeros(N)

D[1:] = prm.D*delta_t/delta_r**2-prm.D*delta_t/(2*vec_r[1:]*delta_r)    # Coefficient devant C_i-1 (vecteur)
E = 0 - 2*prm.D*delta_t/delta_r**2 #+ k*delta_t  # Coefficient devant C_i (scalaire)
F[1:] = (prm.D*delta_t/(2*vec_r[1:]*delta_r)+prm.D*delta_t/delta_r**2)  # Coefficient devant C_i+1 (vecteur)

# Initialisation des vecteurs de concentration

C_ini_2 = np.ones(N)*prm.S
C_act_2 = np.zeros(N)
mat_C_2 = np.zeros([N,N])
crit = 1


# Condition de Neumann à i = 0

mat_C_2[0,0] = -3
mat_C_2[0,1] = 4
mat_C_2[0,2] = -1
C_ini_2[0] = 0

# Condition de Dirichlet à i = N

mat_C_2[-1,-1] = 1
C_ini_2[-1] = prm.Ce


# Algorithme de différences finies

b = C_ini_2.copy()
c = 0

for i in range(1,len(C_ini_2)-1):
    
    mat_C_2[i,i-1] = D[i]
    mat_C_2[i,i] = E
    mat_C_2[i,i+1] = F[i]
    
    b[i] = C_ini_2[i] 
    
# Résolution du système matriciel
C_act_2 = np.linalg.solve(mat_C_2,b)
  

# Établissement de la solution analytique

def sol_analytique(x,prm):
    
    C = 0.25*prm.S/prm.D*prm.R**2*(x**2/R**2-1)+prm.Ce
    
    return C


# Affichage des graphiques et des erreurs
    
C_anal = sol_analytique(vec_r,prm)
plt.scatter(vec_r,C_anal,label='Cas analytique',color='g')   
plt.plot(vec_r,C_act,label='Cas 1')
plt.plot(vec_r,C_act_2,label='Cas 2')
plt.legend()  
plt.xlabel("Position radiale [m]")
plt.ylabel("Concentration [mol/m$^3$]")
plt.title("Profil de la concentration de sel dans un pilier en \nbéton poreux en fonction de la position radiale")
plt.show()


L1_1 = np.sum(abs(C_act-C_anal))/(len(C_act))
L1_2 = np.sum(abs(C_act_2-C_anal))/(len(C_act_2))

L2_1 = (np.sum((C_act-C_anal)**2)/(len(C_act)))**0.5
L2_2 = (np.sum((C_act_2-C_anal)**2)/(len(C_act_2)))**0.5

Linf_1 = np.max(abs(C_act-C_anal))
Linf_2 = np.max(abs(C_act_2-C_anal))


print('nb points :', N)
print('Cas 1')
print('Erreur L1 : ',L1_1)
print('Erreur L2 : ',L2_1)
print('Erreur Linf : ',Linf_1)
      
print('Cas 2')
print('Erreur L1 : ',L1_2)
print('Erreur L2 : ',L2_2)
print('Erreur Linf : ',Linf_2)







