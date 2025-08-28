#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 19:56:53 2025

@author: adminprincipal
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import seaborn as sns

# Charger le fichier ODS
file_path = "data.ods"  # Vérifie que le chemin est correct
df_sheet2 = pd.read_excel(file_path, engine="odf", sheet_name="Sheet2", skiprows=2)

# Supprimer les colonnes entièrement vides
df_sheet2 = df_sheet2.dropna(axis=1, how='all')

# Afficher quelques informations pour vérification
print("Colonnes disponibles :", df_sheet2.columns)
print(df_sheet2.head())

# Convertir les données en float et recentrer en soustrayant 0.017
df_cleaned_sheet2 = df_sheet2.iloc[1:].reset_index(drop=True).astype(float, errors='ignore') - 0.017

# Vérification du nombre de colonnes (attendu : 13)
num_columns = df_cleaned_sheet2.shape[1]
if num_columns != 13:
    print(f"Attention : nombre de colonnes détecté = {num_columns}, attendu = 13.")

# Extraire le temps (colonne 0) et les positions (colonnes 1-12 réparties en 6 masses)
time = df_cleaned_sheet2.iloc[:, 0]
positions_sheet2 = []
for i in range(6):
    try:
        x = df_cleaned_sheet2.iloc[:, 1 + 2 * i]
        y = df_cleaned_sheet2.iloc[:, 2 + 2 * i]
        positions_sheet2.append((x, y))
    except IndexError:
        print(f"Problème avec la masse {i+1}, colonnes hors limites.")

# -----------------------------
# -----------------------------
# -----------------------------
# Partie 1 : Graphiques
# -----------------------------
# -----------------------------
# -----------------------------

# ----------------------------
# Figure 1 : Graphique global
# ----------------------------

fig_main, ax_main = plt.subplots(figsize=(8, 6))
fig_main.canvas.manager.set_window_title("Évolution des 6 masses - Graphique Global")
colors = ['b', 'g', 'r', 'c', 'm', 'y']
lines_main = []
markers_main = []

# Tracer toutes les trajectoires avec opacité réduite et créer des marqueurs vides
for i, (x, y) in enumerate(positions_sheet2, start=1):
    ax_main.plot(x, y, color=colors[i - 1], alpha=0.3, label=f'Masse {i}')
    marker, = ax_main.plot([], [], 'o', color=colors[i - 1], markersize=10)
    markers_main.append(marker)

ax_main.set_xlabel('Position X (m)')
ax_main.set_ylabel('Position Y (m)')
ax_main.set_title('Trajectoires des masses avec suivi temporel - Global')
ax_main.legend(bbox_to_anchor=(0, 1), loc='upper left')
ax_main.grid(True)

# Ajout du slider pour le graphique global
ax_slider_main = fig_main.add_axes([0.2, 0.02, 0.6, 0.03])
slider_main = Slider(ax_slider_main, 'Temps', 0, len(time)-1, valinit=0, valfmt='%0.0f (ds)')

def update_main(val):
    t_index = int(slider_main.val)
    for i, (x, y) in enumerate(positions_sheet2):
        if t_index < len(x):  # Pour éviter un dépassement d'indice
            markers_main[i].set_data(x.iloc[t_index], y.iloc[t_index])
    fig_main.canvas.draw_idle()

slider_main.on_changed(update_main)

# --------------------------------------
# Figure 2 : Graphiques individuels
# --------------------------------------

fig_ind, axes_ind = plt.subplots(2, 3, figsize=(12, 8))
fig_ind.canvas.manager.set_window_title("Suivi temporel des 6 masses - Individuel")
lines_ind = []
markers_ind = []
for i, (x, y) in enumerate(positions_sheet2, start=1):
    row, col = divmod(i - 1, 3)  # Disposition en grille 2x3
    ax = axes_ind[row, col]
    ax.plot(x, y, color=colors[i - 1], alpha=0.3)
    marker, = ax.plot([], [], 'o', color=colors[i - 1], markersize=10)
    markers_ind.append(marker)
    ax.set_xlabel('Position X (m)')
    ax.set_ylabel('Position Y (m)')
    ax.set_title(f'Trajectoire Masse {i}')
    ax.grid(True)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Laisser un peu d'espace en bas pour le slider

# Ajout du slider pour les graphiques individuels
ax_slider_ind = fig_ind.add_axes([0.2, 0.01, 0.6, 0.03])
slider_ind = Slider(ax_slider_ind, 'Temps', 0, len(time)-1, valinit=0, valfmt='%0.0f (ds)')

def update_ind(val):
    t_index = int(slider_ind.val)
    for i, (x, y) in enumerate(positions_sheet2):
        if t_index < len(x):
            markers_ind[i].set_data(x.iloc[t_index], y.iloc[t_index])
    fig_ind.canvas.draw_idle()

slider_ind.on_changed(update_ind)

# ----------------------------
# Afficher les deux figures
# ----------------------------

plt.show()

# -----------------------------
# Calcul des vitesses
# -----------------------------

time_array = time.values.astype(float)

velocities_x = []
velocities_y = []
speeds = []  # Module de la vitesse

for idx, (x, y) in enumerate(positions_sheet2, start=1):
    x_array = x.values.astype(float)
    y_array = y.values.astype(float)
    
    vx = np.gradient(x_array, time_array)
    vy = np.gradient(y_array, time_array)
    # Utilisation de sqrt(vx^2 + vy^2). Si vous préférez (vx^2+vy^2)^2, modifiez ici.
    v = np.sqrt(vx**2 + vy**2)
    
    velocities_x.append(vx)
    velocities_y.append(vy)
    speeds.append(v)
    
    print(f"Masse {idx} : v_x[0:5] = {vx[:5]}, v_y[0:5] = {vy[:5]}, vitesse[0:5] = {v[:5]}")

# -----------------------------
# Figure dynamique : Trajectoires avec ligne colorée et curseur temps
# -----------------------------

fig_dynamic, axes_dynamic = plt.subplots(2, 3, figsize=(14, 10))
fig_dynamic.canvas.manager.set_window_title("Trajectoires colorées par vitesse - Dynamique")
markers_dynamic = []  # Pour les marqueurs indiquant la position courante

# Pour chaque masse, on crée une ligne colorée via LineCollection
for i, ((x, y), v) in enumerate(zip(positions_sheet2, speeds), start=1):
    row, col = divmod(i - 1, 3)
    ax = axes_dynamic[row, col]
    
    # Récupération des données en numpy arrays
    x_array = x.values.astype(float)
    y_array = y.values.astype(float)
    # Construction des segments : une séquence de points reliés
    points = np.array([x_array, y_array]).T
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    # Pour colorer chaque segment, on prend la vitesse moyenne sur le segment
    v_seg = (v[:-1] + v[1:]) / 2
    
    # Création de la LineCollection avec la colormap 'viridis'
    norm = plt.Normalize(vmin=v_seg.min(), vmax=v_seg.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(v_seg)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    
    # Paramétrage de l'axe
    ax.set_xlim(x_array.min(), x_array.max())
    ax.set_ylim(y_array.min(), y_array.max())
    ax.set_xlabel('Position X (m)')
    ax.set_ylabel('Position Y (m)')
    ax.set_title(f'Masse {i}')
    ax.grid(True)
    
    # Ajout d'une colorbar pour chaque sous-graphe
    cbar = fig_dynamic.colorbar(lc, ax=ax)
    cbar.set_label('Vitesse (m/s)')
    
    # Ajout d'un marqueur pour la position courante (initialement à l'indice 0)
    marker, = ax.plot(x_array[0], y_array[0], 'o', color='black', markersize=8, markeredgecolor='white')
    markers_dynamic.append(marker)

plt.tight_layout(rect=[0, 0.08, 1, 0.95])

# Ajout d'un slider pour contrôler le temps dans la figure dynamique
ax_slider_dyn = fig_dynamic.add_axes([0.15, 0.02, 0.7, 0.03])
slider_dyn = Slider(ax_slider_dyn, 'Temps', 0, len(time)-1, valinit=0, valfmt='%0.0f (ds)')

def update_dynamic(val):
    t_index = int(slider_dyn.val)
    for i, ((x, y), marker) in enumerate(zip(positions_sheet2, markers_dynamic)):
        if t_index < len(x):
            marker.set_data(x.iloc[t_index], y.iloc[t_index])
    fig_dynamic.canvas.draw_idle()

slider_dyn.on_changed(update_dynamic)

plt.show()

# -----------------------------
# Fonction de regroupement (clustering) des points
# -----------------------------

def cluster_points(x_points, y_points, threshold=0.05):
    
   # Regroupe les points dont la distance mutuelle est inférieure à threshold (en m) et renvoie une liste de centres de clusters.
    
    clusters = []
    for x, y in zip(x_points, y_points):
        # Vérifier si le point est déjà proche d'un centre existant
        if any(np.hypot(x - cx, y - cy) < threshold for cx, cy in clusters):
            continue
        clusters.append((x, y))
    return np.array(clusters) if clusters else np.empty((0, 2))

# -----------------------------
# Graphique unique : Trajectoires de toutes les masses avec vitesse en fonction des positions (X, Y)
# et ajout des points rouges regroupés dans la zone considérée
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 8))
fig.canvas.manager.set_window_title("Vitesses en fonction des positions X et Y - Toutes masses")
markers = []  # Pour le suivi de la position/speed instantanée de chaque masse

# Tracer les trajectoires avec segments colorés selon la vitesse
for i, ((x, y), v) in enumerate(zip(positions_sheet2, speeds), start=1):
    x_array = x.values.astype(float)
    y_array = y.values.astype(float)
    
    # Construction des segments reliant les points
    points = np.array([x_array, y_array]).T
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    # Calcul de la vitesse moyenne sur chaque segment
    v_seg = (v[:-1] + v[1:]) / 2
    # Normalisation pour la colormap
    norm = plt.Normalize(vmin=v_seg.min(), vmax=v_seg.max())
    
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(v_seg)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    
    # Ajout d'un marqueur indiquant la position courante (initialement indice 0)
    # On ne donne pas de label ici pour ne pas l'inclure dans la légende.
    marker, = ax.plot(x_array[0], y_array[0], 'o', color='black',
                      markersize=8, markeredgecolor='white')
    markers.append(marker)
    
    # -----------------------------
    # Détermination des points rouges répondant aux conditions
    # Conditions : vitesse < 0.25 m/s, x entre 0.05 et 0.95 m, y entre 0.05 et 0.95 m
    # -----------------------------
    condition = (v < 0.15) & (x_array >= 0.08) & (x_array <= 0.92) & (y_array >= 0.08) & (y_array <= 0.92)
    if np.any(condition):
        x_candidates = x_array[condition]
        y_candidates = y_array[condition]
        # Regrouper les points rouges proches (à moins de 5 cm)
        clusters = cluster_points(x_candidates, y_candidates, threshold=0.05)
        if clusters.size > 0:
            ax.scatter(clusters[:, 0], clusters[:, 1],
                       color='red', s=100, zorder=10)

# Création d'un proxy pour la légende des collisions (point rouge)
collision_marker = mlines.Line2D([], [], marker='o', color='red', linestyle='None', markersize=10,
                                   label='Colission Atome/Atome')

# On crée la légende en n'affichant que le proxy de collision
ax.legend(handles=[collision_marker], loc='upper right')

ax.set_xlabel('Position X (m)')
ax.set_ylabel('Position Y (m)')
ax.set_title("Trajectoires de toutes les masses colorées par vitesse")
ax.grid(True)

# Définir des limites englobant toutes les trajectoires
all_x = np.concatenate([x.values.astype(float) for (x, _) in positions_sheet2])
all_y = np.concatenate([y.values.astype(float) for (_, y) in positions_sheet2])
ax.set_xlim(all_x.min(), all_x.max())
ax.set_ylim(all_y.min(), all_y.max())

# Ajout d'une colorbar globale basée sur l'ensemble des vitesses
all_v_seg = np.concatenate([ ((v[:-1]+v[1:]) / 2) for v in speeds ])
norm_global = plt.Normalize(vmin=all_v_seg.min(), vmax=all_v_seg.max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm_global)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Vitesse (m/s)')

# -----------------------------
# Curseur temporel pour la mise à jour des positions instantanées
# -----------------------------

ax_slider = fig.add_axes([0.2, 0.02, 0.6, 0.03])
slider = Slider(ax_slider, 'Temps', 0, len(time)-1, valinit=0, valfmt='%0.0f (ds)')

def update(val):
    t_index = int(slider.val)
    for (x, y), marker in zip(positions_sheet2, markers):
        if t_index < len(x):
            marker.set_data(x.iloc[t_index], y.iloc[t_index])
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()

# -----------------------------
# -----------------------------
# -----------------------------
# Partie 2 : Densité de probabilités
# -----------------------------
# -----------------------------
# -----------------------------

# -----------------------------
# Graphique global
# -----------------------------

# Regrouper toutes les positions des masses
all_x = np.concatenate([x.values.astype(float) for (x, _) in positions_sheet2])
all_y = np.concatenate([y.values.astype(float) for (_, y) in positions_sheet2])

# Estimation de densité par noyau gaussien (KDE)
kde = gaussian_kde(np.vstack([all_x, all_y]))
xmin, xmax = all_x.min(), all_x.max()
ymin, ymax = all_y.min(), all_y.max()

# Création d'une grille de points pour la visualisation
x_grid, y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
pos = np.vstack([x_grid.ravel(), y_grid.ravel()])

# Calcul des densités sur la grille
density = kde(pos).reshape(100, 100)

# Ajouter la contribution des disques à la densité
for x, y in zip(all_x, all_y):
    mask = (x_grid - x)**2 + (y_grid - y)**2 <= 0.03614**2
    density[mask] += 1  # Augmenter la densité dans les zones occupées

# Affichage de la heatmap
fig_density, ax_density = plt.subplots(figsize=(8, 6), constrained_layout=True)
cmap = ax_density.imshow(density.T, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='inferno', alpha=0.75, norm=plt.matplotlib.colors.PowerNorm(gamma=0.5))

# Configuration des axes
ax_density.set_xlabel("Position X (m)")
ax_density.set_ylabel("Position Y (m)")
ax_density.set_title("Répartition de probabilité lissée avec colormap")
fig_density.colorbar(cmap, label="Densité de probabilité")

plt.show()

# -----------------------------
# Graphique individuel 
# -----------------------------

# Regrouper toutes les positions des masses
all_x = np.concatenate([x.values.astype(float) for (x, _) in positions_sheet2])
all_y = np.concatenate([y.values.astype(float) for (_, y) in positions_sheet2])

# Estimation de densité par noyau gaussien (KDE)
kde = gaussian_kde(np.vstack([all_x, all_y]))
xmin, xmax = all_x.min(), all_x.max()
ymin, ymax = all_y.min(), all_y.max()

# Création d'une grille de points pour la visualisation
x_grid, y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
pos = np.vstack([x_grid.ravel(), y_grid.ravel()])

# Calcul des densités sur la grille
density = kde(pos).reshape(100, 100)

# Ajouter la contribution des disques à la densité
for x, y in zip(all_x, all_y):
    mask = (x_grid - x)**2 + (y_grid - y)**2 <= 0.03614**2
    density[mask] += 1  # Augmenter la densité dans les zones occupées
max_time = time.max()

# Création de la figure avec sous-graphiques
fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
fig.suptitle(f"Répartition de probabilité par masse (Temps max: {max_time:.2f}s)")
for i, ax in enumerate(axes.flat):
    if i < 6:
        x_mass = positions_sheet2[i][0].values.astype(float)
        y_mass = positions_sheet2[i][1].values.astype(float)
        kde_mass = gaussian_kde(np.vstack([x_mass, y_mass]))
        density_mass = kde_mass(pos).reshape(100, 100)
        for x, y in zip(x_mass, y_mass):
            mask = (x_grid - x)**2 + (y_grid - y)**2 <= 0.03614**2
            density_mass[mask] += 1
        
        cmap = ax.imshow(density_mass.T, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='inferno', alpha=0.75, norm=plt.matplotlib.colors.PowerNorm(gamma=0.5))
        ax.set_title(f"Masse {i+1}")
        ax.set_xlabel("Position X (m)")
        ax.set_ylabel("Position Y (m)")
        fig.colorbar(cmap, ax=ax, label="Densité de probabilité", shrink=0.7)

plt.show()
print(f"Temps maximal : {max_time}")

#Pour les deux graphiques de densité de probabilités, ce sont les mêmes réferentiels, la comparaison est donc présente
#Nous pouvons donc affirmer que l'augmentation de N permet de lisser la répartition
#A faire si le temps : mettre un curseur pour rajouter une masse etc 

# -----------------------------
# -----------------------------
# -----------------------------
# Partie 4 : Probabilité de présence à droite/gauche d'une ligne
# -----------------------------
# -----------------------------
# -----------------------------

# Créer une figure avec deux sous-graphiques (1: visualisation spatiale, 2: distribution de probabilité)
fig_distrib, (ax_spatial, ax_prob) = plt.subplots(1, 2, figsize=(14, 6))
fig_distrib.canvas.manager.set_window_title("Probabilité de présence selon l'axe X")

# Regrouper toutes les positions
all_x = np.concatenate([x.values.astype(float) for (x, _) in positions_sheet2])
all_y = np.concatenate([y.values.astype(float) for (_, y) in positions_sheet2])

# Valeurs limites pour X
x_min, x_max = all_x.min(), all_x.max()
total_points = len(all_x)

# Position initiale de la ligne verticale
initial_x_pos = (x_min + x_max) / 2

# Tracer les positions dans le graphique spatial
ax_spatial.scatter(all_x, all_y, s=5, c='blue', alpha=0.3)
line = ax_spatial.axvline(x=initial_x_pos, color='r', linestyle='-', linewidth=2)
ax_spatial.set_xlabel('Position X (m)')
ax_spatial.set_ylabel('Position Y (m)')
ax_spatial.set_title('Positions des masses avec ligne de séparation')
ax_spatial.grid(True)

# Fonction pour calculer et afficher la distribution de probabilité
def update_probability_distribution(x_pos):
    # Calculer les probabilités
    points_left = np.sum(all_x < x_pos)
    points_right = total_points - points_left
    prob_left = points_left / total_points
    prob_right = points_right / total_points
    
    # Effacer le graphique précédent
    ax_prob.clear()
    
    # Créer un simple graphique en barres
    bars = ax_prob.bar(['Gauche', 'Droite'], [prob_left, prob_right], color=['skyblue', 'lightgreen'])
    
    # Ajouter les valeurs numériques sur les barres
    for bar in bars:
        height = bar.get_height()
        ax_prob.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom')
    
    # Configurer l'axe
    ax_prob.set_ylim(0, 1)
    ax_prob.set_title(f'Probabilité de présence (x={x_pos:.2f} m)')
    ax_prob.set_ylabel('Probabilité')
    
    # Ajouter aussi l'information textuelle
    stats_text = f"Position de la ligne: {x_pos:.2f} m\n"
    stats_text += f"Points à gauche: {points_left} ({prob_left:.2%})\n"
    stats_text += f"Points à droite: {points_right} ({prob_right:.2%})"
    ax_prob.text(0.5, 0.5, stats_text, transform=ax_prob.transAxes,
              verticalalignment='center', horizontalalignment='center',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig_distrib.canvas.draw_idle()

# Initialiser avec la position initiale
update_probability_distribution(initial_x_pos)

# Créer un slider pour contrôler la position de la ligne
ax_slider_x = fig_distrib.add_axes([0.2, 0.02, 0.6, 0.03])
slider_x_pos = Slider(ax_slider_x, 'Position X', x_min, x_max, valinit=initial_x_pos, valfmt='%.2f m')

# Fonction de mise à jour lors du déplacement du slider
def update_x_pos(val):
    line.set_xdata([val, val])
    update_probability_distribution(val)

slider_x_pos.on_changed(update_x_pos)

# Ajout d'une fonctionnalité pour cliquer directement sur le graphique
def onclick(event):
    if event.inaxes == ax_spatial:
        val = event.xdata
        if x_min <= val <= x_max:
            slider_x_pos.set_val(val)

fig_distrib.canvas.mpl_connect('button_press_event', onclick)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# -----------------------------
# Version améliorée: Distribution en fonction de x
# -----------------------------

# Créer une nouvelle figure pour l'analyse détaillée de la distribution le long de l'axe X
fig_distrib_detailed, (ax_hist, ax_cumul) = plt.subplots(2, 1, figsize=(10, 8))
fig_distrib_detailed.canvas.manager.set_window_title("Distribution détaillée selon l'axe X")

# Histogramme des positions en X
bins = np.linspace(x_min, x_max, 40)  # 40 intervalles entre min et max
counts, edges, _ = ax_hist.hist(all_x, bins=bins, color='skyblue', alpha=0.7, edgecolor='black')
ax_hist.set_title("Distribution des positions selon l'axe X")
ax_hist.set_xlabel("Position X (m)")
ax_hist.set_ylabel("Nombre de points")
ax_hist.grid(True, alpha=0.3)

# Ligne verticale pour la position X sélectionnée
line_hist = ax_hist.axvline(x=initial_x_pos, color='r', linestyle='-', linewidth=2)

# Graphique de probabilité cumulée
cumul_values = np.cumsum(counts) / total_points
bin_centers = 0.5 * (edges[1:] + edges[:-1])
ax_cumul.plot(bin_centers, cumul_values, 'b-', linewidth=2)
ax_cumul.set_title("Probabilité cumulée selon l'axe X")
ax_cumul.set_xlabel("Position X (m)")
ax_cumul.set_ylabel("Probabilité cumulée")
ax_cumul.set_ylim(0, 1)
ax_cumul.grid(True, alpha=0.3)

# Lignes horizontale et verticale pour la position X sélectionnée
line_cumul_v = ax_cumul.axvline(x=initial_x_pos, color='r', linestyle='-', linewidth=2)
line_cumul_h = ax_cumul.axhline(y=np.interp(initial_x_pos, bin_centers, cumul_values), 
                               color='r', linestyle='--', linewidth=1)

# Texte pour afficher la valeur de probabilité cumulée
prob_text = ax_cumul.text(0.98, 0.95, f"P(X ≤ {initial_x_pos:.2f}) = {np.interp(initial_x_pos, bin_centers, cumul_values):.2%}",
                         transform=ax_cumul.transAxes, ha='right', va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Fonction de mise à jour pour la deuxième figure
def update_detailed_view(val):
    line_hist.set_xdata([val, val])
    line_cumul_v.set_xdata([val, val])
    
    # Mettre à jour la ligne horizontale à la probabilité cumulée correspondante
    cum_prob = np.interp(val, bin_centers, cumul_values)
    line_cumul_h.set_ydata([cum_prob, cum_prob])
    
    # Mettre à jour le texte
    prob_text.set_text(f"P(X ≤ {val:.2f}) = {cum_prob:.2%}")
    
    fig_distrib_detailed.canvas.draw_idle()

# Créer un slider pour la deuxième figure
ax_slider_x_detailed = fig_distrib_detailed.add_axes([0.2, 0.02, 0.6, 0.03])
slider_x_detailed = Slider(ax_slider_x_detailed, 'Position X', x_min, x_max, valinit=initial_x_pos, valfmt='%.2f m')
slider_x_detailed.on_changed(update_detailed_view)

# Synchroniser les deux sliders (optionnel)
def update_both_sliders(val):
    slider_x_pos.set_val(val)
    slider_x_detailed.set_val(val)

slider_x_pos.on_changed(lambda val: slider_x_detailed.set_val(val))
slider_x_detailed.on_changed(lambda val: slider_x_pos.set_val(val))

# Fonction pour cliquer sur les graphiques de la deuxième figure
def onclick_detailed(event):
    if event.inaxes in [ax_hist, ax_cumul]:
        val = event.xdata
        if x_min <= val <= x_max:
            update_both_sliders(val)

fig_distrib_detailed.canvas.mpl_connect('button_press_event', onclick_detailed)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# -----------------------------
# -----------------------------
# -----------------------------
# Partie 5 : Probabilité de présence au-dessus/au-dessous d'une ligne horizontale
# -----------------------------
# -----------------------------
# -----------------------------

# Créer une figure avec deux sous-graphiques (1: visualisation spatiale, 2: distribution de probabilité)
fig_distrib_y, (ax_spatial_y, ax_prob_y) = plt.subplots(1, 2, figsize=(14, 6))
fig_distrib_y.canvas.manager.set_window_title("Probabilité de présence selon l'axe Y")

# Valeurs limites pour Y
y_min, y_max = all_y.min(), all_y.max()
total_points = len(all_y)

# Position initiale de la ligne horizontale
initial_y_pos = (y_min + y_max) / 2

# Tracer les positions dans le graphique spatial
ax_spatial_y.scatter(all_x, all_y, s=5, c='blue', alpha=0.3)
line_y = ax_spatial_y.axhline(y=initial_y_pos, color='r', linestyle='-', linewidth=2)
ax_spatial_y.set_xlabel('Position X (m)')
ax_spatial_y.set_ylabel('Position Y (m)')
ax_spatial_y.set_title('Positions des masses avec ligne de séparation horizontale')
ax_spatial_y.grid(True)

# Fonction pour calculer et afficher la distribution de probabilité selon Y
def update_probability_distribution_y(y_pos):
    # Calculer les probabilités
    points_below = np.sum(all_y < y_pos)
    points_above = total_points - points_below
    prob_below = points_below / total_points
    prob_above = points_above / total_points
    
    # Effacer le graphique précédent
    ax_prob_y.clear()
    
    # Créer un simple graphique en barres
    bars = ax_prob_y.bar(['Au-dessous', 'Au-dessus'], [prob_below, prob_above], color=['orange', 'green'])
    
    # Ajouter les valeurs numériques sur les barres
    for bar in bars:
        height = bar.get_height()
        ax_prob_y.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom')
    
    # Configurer l'axe
    ax_prob_y.set_ylim(0, 1)
    ax_prob_y.set_title(f'Probabilité de présence (y={y_pos:.2f} m)')
    ax_prob_y.set_ylabel('Probabilité')
    
    # Ajouter aussi l'information textuelle
    stats_text = f"Position de la ligne: {y_pos:.2f} m\n"
    stats_text += f"Points au-dessous: {points_below} ({prob_below:.2%})\n"
    stats_text += f"Points au-dessus: {points_above} ({prob_above:.2%})"
    ax_prob_y.text(0.5, 0.5, stats_text, transform=ax_prob_y.transAxes,
              verticalalignment='center', horizontalalignment='center',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig_distrib_y.canvas.draw_idle()

# Initialiser avec la position initiale
update_probability_distribution_y(initial_y_pos)

# Créer un slider pour contrôler la position de la ligne horizontale
ax_slider_y = fig_distrib_y.add_axes([0.2, 0.02, 0.6, 0.03])
slider_y_pos = Slider(ax_slider_y, 'Position Y', y_min, y_max, valinit=initial_y_pos, valfmt='%.2f m')

# Fonction de mise à jour lors du déplacement du slider
def update_y_pos(val):
    line_y.set_ydata([val, val])
    update_probability_distribution_y(val)

slider_y_pos.on_changed(update_y_pos)

# Ajout d'une fonctionnalité pour cliquer directement sur le graphique
def onclick_y(event):
    if event.inaxes == ax_spatial_y:
        val = event.ydata
        if y_min <= val <= y_max:
            slider_y_pos.set_val(val)

fig_distrib_y.canvas.mpl_connect('button_press_event', onclick_y)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# -----------------------------
# Version améliorée: Distribution en fonction de y
# -----------------------------

# Créer une nouvelle figure pour l'analyse détaillée de la distribution le long de l'axe Y
fig_distrib_detailed_y, (ax_hist_y, ax_cumul_y) = plt.subplots(2, 1, figsize=(10, 8))
fig_distrib_detailed_y.canvas.manager.set_window_title("Distribution détaillée selon l'axe Y")

# Histogramme des positions en Y
bins_y = np.linspace(y_min, y_max, 40)  # 40 intervalles entre min et max
counts_y, edges_y, _ = ax_hist_y.hist(all_y, bins=bins_y, color='orange', alpha=0.7, edgecolor='black')
ax_hist_y.set_title("Distribution des positions selon l'axe Y")
ax_hist_y.set_xlabel("Position Y (m)")
ax_hist_y.set_ylabel("Nombre de points")
ax_hist_y.grid(True, alpha=0.3)

# Ligne horizontale pour la position Y sélectionnée
line_hist_y = ax_hist_y.axvline(x=initial_y_pos, color='r', linestyle='-', linewidth=2)

# Graphique de probabilité cumulée
cumul_values_y = np.cumsum(counts_y) / total_points
bin_centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])
ax_cumul_y.plot(bin_centers_y, cumul_values_y, 'b-', linewidth=2)
ax_cumul_y.set_title("Probabilité cumulée selon l'axe Y")
ax_cumul_y.set_xlabel("Position Y (m)")
ax_cumul_y.set_ylabel("Probabilité cumulée")
ax_cumul_y.set_ylim(0, 1)
ax_cumul_y.grid(True, alpha=0.3)

# Lignes horizontale et verticale pour la position Y sélectionnée
line_cumul_v_y = ax_cumul_y.axvline(x=initial_y_pos, color='r', linestyle='-', linewidth=2)
line_cumul_h_y = ax_cumul_y.axhline(y=np.interp(initial_y_pos, bin_centers_y, cumul_values_y), 
                               color='r', linestyle='--', linewidth=1)

# Texte pour afficher la valeur de probabilité cumulée
prob_text_y = ax_cumul_y.text(0.98, 0.95, f"P(Y ≤ {initial_y_pos:.2f}) = {np.interp(initial_y_pos, bin_centers_y, cumul_values_y):.2%}",
                         transform=ax_cumul_y.transAxes, ha='right', va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Fonction de mise à jour pour la deuxième figure (axe Y)
def update_detailed_view_y(val):
    line_hist_y.set_xdata([val, val])
    line_cumul_v_y.set_xdata([val, val])
    
    # Mettre à jour la ligne horizontale à la probabilité cumulée correspondante
    cum_prob_y = np.interp(val, bin_centers_y, cumul_values_y)
    line_cumul_h_y.set_ydata([cum_prob_y, cum_prob_y])
    
    # Mettre à jour le texte
    prob_text_y.set_text(f"P(Y ≤ {val:.2f}) = {cum_prob_y:.2%}")
    
    fig_distrib_detailed_y.canvas.draw_idle()

# Créer un slider pour la deuxième figure (axe Y)
ax_slider_y_detailed = fig_distrib_detailed_y.add_axes([0.2, 0.02, 0.6, 0.03])
slider_y_detailed = Slider(ax_slider_y_detailed, 'Position Y', y_min, y_max, valinit=initial_y_pos, valfmt='%.2f m')
slider_y_detailed.on_changed(update_detailed_view_y)

# Synchroniser les deux sliders pour l'axe Y (optionnel)
def update_both_sliders_y(val):
    slider_y_pos.set_val(val)
    slider_y_detailed.set_val(val)

slider_y_pos.on_changed(lambda val: slider_y_detailed.set_val(val))
slider_y_detailed.on_changed(lambda val: slider_y_pos.set_val(val))

# Fonction pour cliquer sur les graphiques de la deuxième figure (axe Y)
def onclick_detailed_y(event):
    if event.inaxes in [ax_hist_y, ax_cumul_y]:
        val = event.xdata
        if y_min <= val <= y_max:
            update_both_sliders_y(val)

fig_distrib_detailed_y.canvas.mpl_connect('button_press_event', onclick_detailed_y)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# -----------------------------
# Partie 6 : Probabilité de présence dans un rectangle - Version finale optimisée
# -----------------------------

# Créer une figure avec une disposition soignée
fig_optimized = plt.figure(figsize=(16, 9))
fig_optimized.canvas.manager.set_window_title("Probabilité de présence dans un rectangle - Disposition optimisée")

# Définir précisément les zones de la figure
ax_main = plt.axes([0.08, 0.15, 0.55, 0.7])          # Zone principale (60% largeur, 70% hauteur)
ax_prob = plt.axes([0.75, 0.30, 0.22, 0.35])         # Graphique probabilité (haut droite)
ax_slider_x = plt.axes([0.08, 0.05, 0.55, 0.03])     # Slider X position (bas)
ax_slider_dx = plt.axes([0.08, 0.01, 0.55, 0.03])    # Slider ΔX (bas)
ax_slider_y = plt.axes([0.69, 0.15, 0.03, 0.7])      # Slider Y position (droite)
ax_slider_dy = plt.axes([0.63, 0.15, 0.03, 0.7])     # Slider ΔY (gauche)

# Valeurs limites et initiales
x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()
total_points = len(all_x)
initial_x = (x_min + x_max) / 2
initial_dx = (x_max - x_min) / 10
initial_y = (y_min + y_max) / 2
initial_dy = (y_max - y_min) / 10

# Configuration de la zone principale
scatter = ax_main.scatter(all_x, all_y, s=5, c='blue', alpha=0.3)
ax_main.set_xlabel('Position X (mètres)', fontsize=12)
ax_main.set_ylabel('Position Y (mètres)', fontsize=12)
ax_main.set_title('Zone de mesure des positions', pad=15, fontsize=14)
ax_main.grid(True, alpha=0.3)

# Rectangle et variables globales
rect = plt.Rectangle((initial_x, initial_y), initial_dx, initial_dy,
                    fill=False, color='red', linewidth=2, linestyle='--')
ax_main.add_patch(rect)
current_x, current_dx, current_y, current_dy = initial_x, initial_dx, initial_y, initial_dy

# Fonction de mise à jour optimisée
def update_optimized():
    global current_x, current_dx, current_y, current_dy
    
    # Mise à jour des variables depuis les curseurs
    current_x = slider_x.val
    current_dx = slider_dx.val
    current_y = slider_y.val
    current_dy = slider_dy.val
    
    # Gestion des limites
    if current_x + current_dx > x_max:
        current_dx = x_max - current_x
        slider_dx.set_val(current_dx)
    if current_y + current_dy > y_max:
        current_dy = y_max - current_y
        slider_dy.set_val(current_dy)
    
    # Calcul des probabilités
    condition = ((all_x >= current_x) & (all_x <= current_x + current_dx) &
                (all_y >= current_y) & (all_y <= current_y + current_dy))
    points_in_rect = np.sum(condition)
    prob_in_rect = points_in_rect / total_points
    
    # Mise à jour visuelle
    rect.set_xy((current_x, current_y))
    rect.set_width(current_dx)
    rect.set_height(current_dy)
    
    ax_prob.clear()
    bars = ax_prob.bar(['Dans la zone', 'Hors zone'], 
                      [prob_in_rect, 1-prob_in_rect], 
                      color=['#2ca02c', '#d62728'])
    
    for bar in bars:
        height = bar.get_height()
        ax_prob.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom')
    
    ax_prob.set_ylim(0, 1.1)
    ax_prob.set_title('Probabilité de présence', fontsize=12)
    ax_prob.set_ylabel('Probabilité', fontsize=10)
    ax_prob.grid(axis='y', alpha=0.2)
    
    info_text = (f"Zone: X ∈ [{current_x:.3f}, {current_x+current_dx:.3f}] m\n"
                f"Y ∈ [{current_y:.3f}, {current_y+current_dy:.3f}] m\n"
                f"Points: {points_in_rect}/{total_points}")
    ax_prob.text(0.5, -0.3, info_text, transform=ax_prob.transAxes,
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    fig_optimized.canvas.draw_idle()

# Configuration des sliders
slider_x = Slider(ax_slider_x, 'Centre X (m)', x_min, x_max, valinit=initial_x, valfmt='%.3f', color='#1f77b4')
slider_dx = Slider(ax_slider_dx, 'Largeur ΔX (m)', 0.001, x_max-x_min, valinit=initial_dx, valfmt='%.3f', color='#1f77b4')
slider_y = Slider(ax_slider_y, 'Centre Y (m)', y_min, y_max, valinit=initial_y, valfmt='%.3f', orientation='vertical', color='#ff7f0e')
slider_dy = Slider(ax_slider_dy, 'Hauteur ΔY (m)', 0.001, y_max-y_min, valinit=initial_dy, valfmt='%.3f', orientation='vertical', color='#ff7f0e')

# Connexion des événements
slider_x.on_changed(lambda val: update_optimized())
slider_dx.on_changed(lambda val: update_optimized())
slider_y.on_changed(lambda val: update_optimized())
slider_dy.on_changed(lambda val: update_optimized())

# Interaction au clic
def onclick(event):
    if event.inaxes == ax_main:
        slider_x.set_val(max(x_min, min(event.xdata, x_max)))
        slider_y.set_val(max(y_min, min(event.ydata, y_max)))
        update_optimized()

fig_optimized.canvas.mpl_connect('button_press_event', onclick)

# Initialisation
update_optimized()

plt.tight_layout()
plt.show()