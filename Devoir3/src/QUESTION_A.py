# -*- coding: utf-8 -*-

# Code

import matplotlib.pyplot as plt
import numpy as np

# Données provenant des simulations Matlab

dx = np.array([4e-6,2e-6,1e-6,5e-7])
k = np.array([22.8513,20.2503,19.726,19.5593])
r = dx[0]/dx[1]


# Graphique de k vs dx

plt.figure(figsize=(8, 6))
plt.scatter(dx, k, marker='o', color='b', label='Données numériques obtenues')

plt.title('Variation de la perméabilité k en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02) 

plt.xlabel('Taille de maille $Δx$ (m)', fontsize=12, fontweight='bold')  
plt.ylabel('Perméabilité k ($\mu m^2$)', fontsize=12, fontweight='bold')

plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)

plt.xscale('log')
plt.yscale('log')

plt.tick_params(width=2, which='both', direction='in', top=True, right=True, length=6)
plt.savefig('dev3_unum.png',dpi=300)


# Calcul de l'ordre observé

p_hat = np.log((k[-3]-k[-2])/(k[-2]-k[-1]))/np.log(r)
print('Ordre observé = ', p_hat)

p_f = 2
print('Ordre formel = ',p_f)

# Calcul de l'erreur
 
erreur_p = abs(p_hat-p_f)/p_f
print('Erreur = ', erreur_p)


# Calcul du GCI

if erreur_p <= 0.1:
    F_s = 1.25
    p = p_f
    
else:
    F_s = 3
    p = min(max(0.5,p_hat),p_f)
    
    
GCI = F_s*abs(k[-2]-k[-1])/(r**p-1)
print('GCI = ', GCI)


# Calcul de l'intervalle de la solution numérique et de l'incertitude numérique

f_h = k[-1]
f_num_min = f_h - GCI
f_num_max = f_h + GCI
print('intervalle f_num : [', f_num_min, ';', f_num_max, ']')

u_num = GCI/2
print('u_num = ', u_num)
    