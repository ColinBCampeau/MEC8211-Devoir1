# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:03:56 2024

@author: co_bc
"""
import numpy as np
import matplotlib.pyplot as plt

# Convergence spatiale

n = np.array([2,5,10,20,40,80])
h = 0.5/n
vec_l2 = np.array([0.4267334,0.08151597,0.02288,0.0058953,0.001480299,0.0003631455])

# Ajuster une loi de puissance à toutes les valeurs (en utilisant np.polyfit avec logarithmes)
coefficients = np.polyfit(np.log(h[2:]), np.log(vec_l2[2:]), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(h[-1])

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
plt.savefig('Convergence_spatiale_comsol.png',dpi=200)
plt.show()

coefficients_x = coefficients


# Convergence temporelle

delta_t = np.array([1e5,5e5,1e6,2e6,5e6,1e7,1e8])
vec_l2_t = np.array([0.0006732,0.00186467,0.003429,0.00655844,0.01577,0.030529,0.22692])

# Ajuster une loi de puissance à toutes les valeurs (en utilisant np.polyfit avec logarithmes)
coefficients = np.polyfit(np.log(delta_t[2:-2]), np.log(vec_l2_t[2:-2]), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(delta_t[2])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(delta_t, vec_l2_t, marker='o', color='b', label='Données numériques obtenues')
plt.plot(delta_t, fit_function(delta_t), linestyle='--', color='r', label='Régression en loi de puissance')

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
plt.savefig('Convergence_temporelle_comsol.png',dpi=200)
plt.show()

