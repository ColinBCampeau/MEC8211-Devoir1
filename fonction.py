"""

OUTIL NUMÉRIQUE DE SIMULATION DE LA DIFFUSION DU SEL DANS UN PILIER EN BÉTON POREUX
                            >FICHIER FONCTION<

AUTEUR: EROJ MOHAMMAD ISHOQ, COLIN BISSONNETTE-CAMPEAU, TRISTAN SÉGUIN
CRÉATION: 04 FÉVRIER 2024
MISE À JOURS: 11 FÉVRIER 2024

"""
#%%===================== IMPORTATION DES MODULES ==========================%%#

import numpy as np

from classe import *

#%%===================== FONCTION SOLUTION ANALYTIQUE ==========================%%#
def sol_analytique(x,prm):
    
    """
    Fonction qui permet de calculer la solution analytique du problème de diffusion
    
    Entrées:
        
        x -> vecteur de point de discrétisation
        
        prm -> paramètres
        
    Sortie:
        
        C -> vecteur des valeurs de concentration à chaque point
    
    """
    
    
    C = 0.25*prm.S/prm.D*prm.R**2*(x**2/prm.R**2-1)+prm.Ce
    
    return C

#%%================= FONCTION MÉTHODE DIFFERENCES FINIS ====================%%#
def f_diff(A, B, C, cas, prm):
    
    """
    Fonction qui permet de calculer la solution numérique du problème de diffusion
    en utilisant la méthode des différences finis
    
    Entrées:
        
        A -> Coefficient devant C_i-1 (scalaire)
        
        B -> Coefficient devant C_i (vecteur)
        
        C -> Coefficient devant C_i+1 (vecteur)
        
        cas -> 1 ou 2
        
        prm -> paramètres
        
    Sortie:
        
        C -> vecteur des valeurs de concentration à chaque point
    
    """
    
    N = prm.N
    
    delta_t = prm.delta_t
    
    # Initialisation des vecteurs de concentration
    C_ini = np.ones(N)
    C_act = np.zeros(N)
    mat_C = np.zeros([N,N])
    crit = 1


    # Condition de Neumann à i = 0

    mat_C[0,0] = -3
    mat_C[0,1] = 4
    mat_C[0,2] = -1
    C_ini[0] = 0

    # Condition de Dirichlet à i = N

    mat_C[-1,-1] = 1
    C_ini[-1] = prm.Ce


    # Algorithme de différences finies

    b = C_ini.copy()
    c = 0

    while crit >= 0.001:
        
        for i in range(1,len(C_ini)-1):
            
            if cas ==1:
            
                mat_C[i,i-1] = A
                mat_C[i,i] = B[i]
                mat_C[i,i+1] = C[i]
            
            elif cas ==2:
                
                mat_C[i,i-1] = A[i]
                mat_C[i,i] = B
                mat_C[i,i+1] = C[i]
            
            b[i] = C_ini[i] - prm.S*delta_t
                
        # Résolution du système matriciel
        C_act = np.linalg.solve(mat_C,b)
        
        # Calcul du critère d'arrêt
        crit = np.linalg.norm(C_act-C_ini)
        #print(crit)
        # Actualisation de la solution (t+1 --> t)
        C_ini = C_act.copy()
        
        c += 1
        
    return C_act


        
#%%=========================== FONCTION ERREUR L1 ==========================%%#
def f_L1(C_act, C_anal):
    
    """
    Fonction qui permet de calculer l'erreur L1
    
    Entrés:
        
        C_act -> vecteur de la concentration obtenue de manière numérique
        
        C_anal -> vecteur de la concentration obtenue de manière analytique
    
    Sortie:
        
        L1 -> erreur L1
    
    """
    
    L1 = np.sum(abs(C_act-C_anal))/(len(C_act))
    
    return L1

#%%=========================== FONCTION ERREUR L2 ==========================%%#

def f_L2(C_act, C_anal):
    
    """
    Fonction qui permet de calculer l'erreur L2
    
    Entrés:
        
        C_act -> vecteur de la concentration obtenue de manière numérique
        
        C_anal -> vecteur de la concentration obtenue de manière analytique
    
    Sortie:
        
        L2 -> erreur L2
    
    """
    
    L2 = (np.sum((C_act-C_anal)**2)/(len(C_act)))**0.5
    
    return L2
    
#%%========================= FONCTION ERREUR Linf ==========================%%#
def f_Linf(C_act, C_anal):
    
    """
    Fonction qui permet de calculer l'erreur Linf
    
    Entrés:
        
        C_act -> vecteur de la concentration obtenue de manière numérique
        
        C_anal -> vecteur de la concentration obtenue de manière analytique
    
    Sortie:
        
        Linf -> erreur Linf
    
    """
    
    L_inf = np.max(abs(C_act-C_anal))
    
    
    return L_inf