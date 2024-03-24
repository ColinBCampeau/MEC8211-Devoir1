%% MONTE CARLO AVEC ECHANTILLONNAGE HYPERCUBE LATIN (LHS)

% Définir le nombre d'échantillons pour LHS
num_samples = 100;

% Définir la moyenne et l'écart type pour la porosité pour LHS
mean_porosity = 0.900; % Moyenne de la porosité
std_porosity = 7.50e-3; % Écart type de la porosité

% Réaliser l'échantillonnage hypercube latin pour la porosité
porosity_samples = lhsnorm(mean_porosity, std_porosity^2, num_samples);

% Préallouer le tableau pour les résultats de perméabilité
permeability_results = zeros(num_samples, 1);

% Définir les paramètres fixes pour la simulation
seed = 101; % Remplacer par la valeur de seed appropriée
mean_diameter = 12.5; % Diamètre moyen des fibres en microns
std_diameter = 2.85; % Écart type du diamètre des fibres en microns
NX = 100; % Remplacer par la taille latérale appropriée du domaine en cellules de grille
dx = 2e-6; % Taille de la grille en mètres
deltaP = 0.1; % Chute de pression en Pascal
filename = 'fiber_mat.tiff'; % Nom du fichier pour l'image de la structure des fibres

% Boucler sur les échantillons LHS et exécuter les simulations
for i = 1:num_samples
    % Extraire l'échantillon de porosité actuel
    current_porosity = porosity_samples(i);
    
    % Générer la structure des fibres pour l'échantillon de porosité actuel
    [d_equivalent] = Generate_sample(seed, filename, mean_diameter, std_diameter, current_porosity, NX, dx);
    
    % Exécuter la simulation LBM pour obtenir la perméabilité
    permeability_results(i) = LBM(filename, NX, deltaP, dx, d_equivalent);
end

%% GENERATION DES RESULTATS

close all;

% Filtrer les résultats non positifs
permeability_results = permeability_results(permeability_results > 0);

% Ajuster la distribution log-normale aux données
log_permeability_results = fitdist(permeability_results, 'Lognormal');

% Extraire les paramètres
mu_log = log_permeability_results.mu;    % Moyenne du logarithme des données
sigma_log = log_permeability_results.sigma; % Écart type du logarithme des données

% Générer une gamme de valeurs pour la perméabilité
permeability_range = linspace(min(permeability_results), max(permeability_results), 1000);

% Calculer la CDF pour la distribution log-normale
cdf_values = logncdf(permeability_range, mu_log, sigma_log);

% Tracer la CDF
figure;
plot(permeability_range, cdf_values, 'LineWidth', 2);
xlabel('Perméabilité (um^2)');
ylabel('CDF');
title('CDF de la perméabilité');
grid on;

% Calcul de l'intervalle pour u
u_low = exp(mu_log) - exp(mu_log)/exp(sigma_log);
u_high = exp(mu_log)*exp(sigma_log) - exp(mu_log);

% Afficher l'intervalle pour u
fprintf('\nu_low: %f\n', u_low);
fprintf('\nu_high %f\n', u_high);

% Calculer la plage de valeurs pour la porosité
x_range = linspace(min(porosity_samples), max(porosity_samples), 1000);

% Calculer la fonction de densité de probabilité (PDF)
pdf_values = normpdf(x_range, mean_porosity, std_porosity);

% Afficher la courbe de la PDF
figure;
plot(x_range, pdf_values, 'LineWidth', 2);
xlabel('Porosité');
ylabel('Densité de probabilité');
title('Courbe de distribution de la porosité (Échantillonnage LHS)');


