# -*- coding: utf-8 -*-
"""
Ondes de choc autour d'une ogive (Modèle Axisymétriaue)- Analyse complète
Créé le : 12 Décembre 2024
Auteur : MAFFIA Edoardo
"""

import numpy as np
from scipy.integrate import odeint
from IPython.display import Image
from scipy.optimize import fsolve
from pylab import *
import matplotlib.pyplot as plt
import math

def draw_cone_tube_horizontal(cone_length, cone_radius, tube_length, 
                              mach_angle_plan=None, 
                              mach_angle_axi=None,
                              angle_mach_amont=None, 
                              angle_mach_aval=None):
    """
    Dessine une représentation 2D (en vue de coupe) d'un cône horizontal connecté à un tube.
    
    Paramètres obligatoires :
    -------------------------
    - cone_length : Longueur du cône
    - cone_radius : Rayon du cône à sa base
    - tube_length : Longueur du tube

    Paramètres optionnels :
    -----------------------
    - mach_angle_plan       : Angle de l'onde de Mach en degrés dans un modle plan (si spécifié, dessine les lignes d'onde de choc)
    - mach_angle_axi        : Angle de l'onde de Mach en degrés dans un modle axisym (si spécifié, dessine les lignes d'onde de choc)
    """
    
    # --------------------------------------------------------------------------------
    # 1) Définition des coordonnées du cône et du tube pour le tracé
    # --------------------------------------------------------------------------------
    # Le cône est défini par trois points (vu de profil) :
    #   - Sommet à x=0, y=0
    #   - Extrémité supérieure à x=cone_length, y=cone_radius
    #   - Extrémité inférieure à x=cone_length, y=-cone_radius
    cone_x = [0, cone_length, cone_length]
    cone_y = [0, cone_radius, -cone_radius]

    # Le tube est un rectangle (vu de profil) relié à la base du cône :
    #   - (cone_length, -cone_radius)
    #   - (cone_length + tube_length, -cone_radius)
    #   - (cone_length + tube_length, cone_radius)
    #   - (cone_length, cone_radius)
    tube_x = [cone_length, cone_length + tube_length, 
              cone_length + tube_length, cone_length]
    tube_y = [-cone_radius, -cone_radius, cone_radius, cone_radius]

    # --------------------------------------------------------------------------------
    # 2) Création de la figure et tracé des surfaces (cône + tube)
    # --------------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    # On remplit la zone correspondant au cône
    plt.fill(cone_x, cone_y, color='gray', edgecolor='black', label='Cône + Tube')
    # On remplit la zone correspondant au tube
    plt.fill(tube_x, tube_y, color='gray', edgecolor='black')
    
    # --------------------------------------------------------------------------------
    # 3) Ajout de l’axe de symétrie (ligne pointillée)
    # --------------------------------------------------------------------------------
    symmetry_line_x = [-cone_length * 0.2, cone_length + tube_length]
    symmetry_line_y = [0, 0]
    plt.plot(symmetry_line_x, symmetry_line_y, color='black', 
             linestyle='--', label='Axe de symétrie')
    
    
    # --------------------------------------------------------------------------------
    # 5) Calcul et affichage de l’angle du cône (delta) entre l’axe et la paroi
    # --------------------------------------------------------------------------------
    cone_angle_rad = np.arctan(cone_radius / cone_length)  # angle en radians
    cone_angle_deg = np.degrees(cone_angle_rad)            # angle en degrés
    
    # Arc de cercle pour représenter l’angle du cône
    angle_circle_radius = cone_length / 1.3
    theta = np.linspace(0, cone_angle_rad, 100)
    circle_x = angle_circle_radius * np.cos(theta)
    circle_y = angle_circle_radius * np.sin(theta)
    plt.plot(circle_x, circle_y, color='black')
    
    # Position du texte pour l'angle (delta)
    plt.text(circle_x[40] + 5, circle_y[40], 
             r'$\delta$ = ' + str(np.round(cone_angle_deg, 2)) + '°', 
             color='black', fontsize=10)

    # --------------------------------------------------------------------------------
    # 6) Tracé de l'onde de Mach plan (mach_angle plan) si spécifiée
    # --------------------------------------------------------------------------------
    if mach_angle_plan is not None:
        mach_angle_rad = np.radians(mach_angle_plan)
        
        # Lignes de l'onde de choc : on dessine deux droites symétriques par rapport à l'axe
        mach_line_x = [0, cone_length + tube_length]
        mach_line_y_top = [0, (cone_length + tube_length) * np.tan(mach_angle_rad)]
        mach_line_y_bottom = [0, -(cone_length + tube_length) * np.tan(mach_angle_rad)]
        
        plt.plot(mach_line_x, mach_line_y_top, color='red', linestyle='--', label='Onde de choc modèle plan')
        plt.plot(mach_line_x, mach_line_y_bottom, color='red', linestyle='--')
        
        # Petit arc de cercle pour représenter l'angle (beta)
        circle_center_x = 0
        circle_center_y = 0
        circle_radius = cone_length / 1.5
        theta = np.linspace(0, mach_angle_rad, 100)
        circle_x = circle_center_x + circle_radius * np.cos(theta)
        circle_y = circle_center_y + circle_radius * np.sin(theta)
        plt.plot(circle_x, circle_y, color='red', linestyle='-', linewidth=2)
        
        # Position du texte pour l'angle de Mach (beta)
        plt.text(circle_x[70] + 10, circle_y[70], 
                 r'$\beta_{plan}$ = ' + str(np.round(mach_angle_plan, 2)) + '°', 
                 color='red', fontsize=12)
        
        # --------------------------------------------------------------------------------
        # 7) Tracé de l'onde de Mach axisym (mach_angle axisym) si spécifiée
        # --------------------------------------------------------------------------------
    if mach_angle_axi is not None:
        mach_angle_rad = np.radians(mach_angle_axi)
    
        # Lignes de l'onde de choc : on dessine deux droites symétriques par rapport à l'axe
        mach_line_x = [0, cone_length + tube_length]
        mach_line_y_top = [0, (cone_length + tube_length) * np.tan(mach_angle_rad)]
        mach_line_y_bottom = [0, -(cone_length + tube_length) * np.tan(mach_angle_rad)]
        
        plt.plot(mach_line_x, mach_line_y_top, color='blue', linestyle='--', label='Onde de choc modèle plan')
        plt.plot(mach_line_x, mach_line_y_bottom, color='blue', linestyle='--')
        
        # Petit arc de cercle pour représenter l'angle (beta)
        circle_center_x = 0
        circle_center_y = 0
        circle_radius = cone_length / 6
        theta = np.linspace(0, mach_angle_rad, 100)
        circle_x = circle_center_x + circle_radius * np.cos(theta)
        circle_y = circle_center_y + circle_radius * np.sin(theta)
        plt.plot(circle_x, circle_y, color='blue', linestyle='-', linewidth=2)
        
        # Position du texte pour l'angle de Mach (beta)
        plt.text(circle_x[70] + 10, circle_y[70], 
                 r'$\beta_{axisym}$ = ' + str(np.round(mach_angle_axi, 2)) + '°', 
                 color='blue', fontsize=12)
    # --------------------------------------------------------------------------------
    # 7) Tracé des lignes de Mach issues du coin convexe (angle_mach_amont / angle_mach_aval)
    # --------------------------------------------------------------------------------
    if (angle_mach_amont is not None) and (angle_mach_aval is not None):
        # Conversion degrés -> radians
        # On considère un "coin" à x=cone_length. 
        # Coin supérieur à y=cone_radius, coin inférieur à y=-cone_radius.
        angle_mach_aval_rad = np.radians(angle_mach_aval) 
        # Pour la ligne "amont", on ajoute l’angle du cône pour bien aligner avec la paroi
        angle_mach_amont_rad = np.radians(angle_mach_amont) + cone_angle_rad

        # Coin convexe supérieur (base du cône en haut)
        convex_corner_x_top = cone_length
        convex_corner_y_top = cone_radius

        # Droite aval (bleue) qui sort du coin supérieur
        extra_line1_x_top = [convex_corner_x_top, convex_corner_x_top + 600]
        extra_line1_y_top = [
            convex_corner_y_top,
            convex_corner_y_top + 600 * np.tan(angle_mach_aval_rad)
        ]
        plt.plot(extra_line1_x_top, extra_line1_y_top, color='blue', linestyle='--', 
                 label='Ligne de Mach aval')

        # Droite amont (verte) qui sort du coin supérieur
        extra_line2_x_top = [convex_corner_x_top, convex_corner_x_top + 600]
        extra_line2_y_top = [
            convex_corner_y_top,
            convex_corner_y_top + 600 * np.tan(angle_mach_amont_rad)
        ]
        plt.plot(extra_line2_x_top, extra_line2_y_top, color='green', linestyle='--', 
                 label='Ligne de Mach amont')

        # Coin convexe inférieur (base du cône en bas)
        convex_corner_x_bottom = cone_length
        convex_corner_y_bottom = -cone_radius

        # Droite aval (bleue) qui sort du coin inférieur
        extra_line1_x_bottom = [convex_corner_x_bottom, convex_corner_x_bottom + 600]
        extra_line1_y_bottom = [
            convex_corner_y_bottom,
            convex_corner_y_bottom + 600 * np.tan(-angle_mach_aval_rad)
        ]
        plt.plot(extra_line1_x_bottom, extra_line1_y_bottom, 
                 color='blue', linestyle='--')

        # Droite amont (verte) qui sort du coin inférieur
        extra_line2_x_bottom = [convex_corner_x_bottom, convex_corner_x_bottom + 600]
        extra_line2_y_bottom = [
            convex_corner_y_bottom,
            convex_corner_y_bottom + 600 * np.tan(-angle_mach_amont_rad)
        ]
        plt.plot(extra_line2_x_bottom, extra_line2_y_bottom, 
                 color='green', linestyle='--')

        # --------------------------------------------------------------------------------
        # 7a) Affichage de l’angle entre la ligne de Mach aval et l’axe horizontal (mu_2)
        # --------------------------------------------------------------------------------
        # On mesure l’angle entre la ligne aval (angle_mach_aval_rad) et l’horizontale (0 rad).
        angle_with_tube_rad = abs(angle_mach_aval_rad - 0)
        angle_with_tube_deg = np.degrees(angle_with_tube_rad)
        
        # Petit arc pour représenter l’angle mu_2
        angle_circle_radius = cone_length / 7
        theta = np.linspace(0, angle_mach_aval_rad, 100)
        circle_x = convex_corner_x_top + angle_circle_radius * np.cos(theta)
        circle_y = convex_corner_y_top + angle_circle_radius * np.sin(theta)
        plt.plot(circle_x, circle_y, color='blue')
        
        # Texte pour mu_2
        plt.text(circle_x[60] + 5, circle_y[60], 
                 r'$\mu_2$ = ' + str(np.round(angle_with_tube_deg, 2)) + '°', 
                 color='blue', fontsize=12)

        # --------------------------------------------------------------------------------
        # 7b) Affichage de l’angle entre la ligne de Mach amont et la prolongation du cône (mu_1)
        # --------------------------------------------------------------------------------
        # On définit la même logique mais pour l'angle amont (qui inclut l'angle du cône)
        prolongation_angle_rad = cone_angle_rad
        angle_between_rad = abs(angle_mach_amont_rad - prolongation_angle_rad)
        angle_between_deg = np.degrees(angle_between_rad)
        
        # Petit arc pour représenter l’angle mu_1
        angle_circle_radius = cone_length / 2
        theta = np.linspace(prolongation_angle_rad, angle_mach_amont_rad, 100)
        circle_x = convex_corner_x_top + angle_circle_radius * np.cos(theta)
        circle_y = convex_corner_y_top + angle_circle_radius * np.sin(theta)
        plt.plot(circle_x, circle_y, color='green')
        
        # Texte pour mu_1
        plt.text(circle_x[50] + 5, circle_y[50], 
                 r'$\mu_1$ = ' + str(np.round(angle_mach_amont, 2)) + '°', 
                 color='green', fontsize=12)
        
        angle_mach_aval_rad = np.radians(angle_mach_aval)
        angle_mach_amont_rad = np.radians(angle_mach_amont) + np.arctan(cone_radius / cone_length)

        angle_between_rad = abs(angle_mach_amont_rad - angle_mach_aval_rad)
        angle_between_deg = np.degrees(angle_between_rad)

        # Affichage de l'angle
        convex_corner_x_top = cone_length
        convex_corner_y_top = cone_radius
        angle_circle_radius = cone_length / 1.5
        theta = np.linspace(angle_mach_aval_rad, angle_mach_amont_rad, 100)
        circle_x = convex_corner_x_top + angle_circle_radius * np.cos(theta)
        circle_y = convex_corner_y_top + angle_circle_radius * np.sin(theta)
        plt.plot(circle_x, circle_y, color='purple', linestyle='-')
        plt.text(circle_x[99], circle_y[99]+10, r'$\gamma$=' + str(np.round(angle_between_deg, 2))) 
        
        # --------------------------------------------------------------------------------
        # 4) Tracé de la prolongation du cône (lignes fines en pointillé)
        # --------------------------------------------------------------------------------
        cone_extension_x_top = [0, cone_length * 5]
        cone_extension_y_top = [0, cone_radius * 5]
        
        cone_extension_x_bottom = [0, cone_length * 5]
        cone_extension_y_bottom = [0, -cone_radius * 5]
        
        plt.plot(cone_extension_x_top, cone_extension_y_top, color='black', 
                 linewidth=0.7, linestyle='--', label="Prolongement du cône")
        plt.plot(cone_extension_x_bottom, cone_extension_y_bottom, color='black', 
                 linewidth=0.7, linestyle='--')
    
    # --------------------------------------------------------------------------------
    # 8) Paramètres d'affichage final
    # --------------------------------------------------------------------------------
    plt.axis('equal')  # Même échelle sur X et Y
    plt.xlim(-cone_length * 0.2, cone_length + tube_length)  
    plt.ylim(-cone_radius * 2, cone_radius * 2)  
    plt.title("Onde de Mach & Détente de Prandtl-Meyer")
    plt.xlabel("Longueur (mm)")
    plt.ylabel("Hauteur (mm)")
    plt.grid(True)
    plt.legend()
    plt.show()


gam = 1.4              # Exposant isentropique (ratio des chaleurs spécifiques)

h = 405                # Longueur du cône (mm)
r = 98/2               # Rayon du cône (mm) -> 98 mm de diamètre, donc 49 mm de rayon
l = 700                # Longueur du tube (mm)

Ma1 = 1.33
P1 = 90178.8
T1 = 13.5 + 273.15

delta = np.arcsin(r/h) * 180.0 / np.pi

def temp_to_sos(T):
    # Speed of sound in dry air given temperature in K
    return 20.05 * T**0.5


def taylor_maccoll(y, theta, gamma=1.4):
    # Taylor-Maccoll function
    # Source: https://www.grc.nasa.gov/www/k-12/airplane/coneflow.html
    v_r, v_theta = y
    dydt = [
        v_theta,
        (v_theta ** 2 * v_r - (gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) * (2 * v_r + v_theta / np.tan(theta))) / ((gamma - 1) / 2 * (1 - v_r ** 2 - v_theta ** 2) - v_theta ** 2) 
    ]
    return dydt


def oblique_shock(theta, Ma, T, p, rho, gamma=1.4):
    """
    Computes the weak oblique shock resulting from supersonic
    flow impinging on a wedge in 2 dimensional flow.
    
    Inputs:
     - theta is the angle of the wedge in radians.
     - Ma, T, p, and rho are the Mach number, temperature (K),
       pressure (Pa), and density (kg/m^3) of the flow.
     - gamma is the ratio of specific heats. Defaults
       to air's typical value of 1.4.
    
    Returns:
     - shock angle in radians
     - resultant flow direction in radians
     - respectively, Mach number, temperature, pressure, density,
       and velocity components downstream of shock.
    
    Source: https://www.grc.nasa.gov/WWW/K-12/airplane/oblique.html
    """
    x = np.tan(theta)
    for B in np.arange(1, 500) * np.pi/1000:
        r = 2 / np.tan(B) * (Ma**2 * np.sin(B)**2 - 1) / (Ma**2 * (gamma + np.cos(2 * B)) + 2)
        if r > x:
            break
    cot_a = np.tan(B) * ((gamma + 1) * Ma ** 2 / (2 * (Ma ** 2 * np.sin(B) ** 2 - 1)) - 1)
    a = np.arctan(1 / cot_a)

    Ma2 = 1 / np.sin(B - theta) * np.sqrt((1 + (gamma - 1)/2 * Ma**2 * np.sin(B)**2) / (gamma * Ma**2 * np.sin(B)**2 - (gamma - 1)/2))

    h = Ma ** 2 * np.sin(B) ** 2
    T2 = T * (2 * gamma * h - (gamma - 1)) * ((gamma - 1) * h + 2) / ((gamma + 1) ** 2 * h)
    p2 = p * (2 * gamma * h - (gamma - 1)) / (gamma + 1)
    rho2 = rho * ((gamma + 1) * h) / ((gamma - 1) * h + 2)

    v2 = Ma2 * temp_to_sos(T2)
    v_x = v2 * np.cos(a)
    v_y = v2 * np.sin(a)
    return B, a, Ma2, T2, p2, rho2, v_x, v_y


def cone_shock(cone_angle, Ma, T, p, rho):
    """
    Computes properties of the conical oblique shock resulting
    from supersonic flow impinging on a cone in 3 dimensional flow.
    Inputs:
     - cone_angle is the half-angle of the 3D cone in radians.
     - Ma, T, p, and rho are the Mach number, temperature (K),
       pressure (Pa), and density (kg/m^3) of the flow.
    Returns:
     - shock angle in radians
     - flow redirection amount in radians
     - respectively, Mach number, temperature, pressure, density,
       and velocity components downstream of shock.
    Source: https://www.grc.nasa.gov/www/k-12/airplane/coneflow.html
    """

    wedge_angles = np.linspace(cone_angle, 0, 300)

    for wedge_angle in wedge_angles:
        B, a, Ma2, T2, p2, rho2, v_x, v_y = oblique_shock(wedge_angle, Ma, T, p, rho)
        v_theta = v_y * np.cos(B) - v_x * np.sin(B)
        v_r = v_y * np.sin(B) + v_x * np.cos(B)
        y0 = [v_r, v_theta]
        thetas = np.linspace(B, cone_angle, 2000)

        sol = odeint(taylor_maccoll, y0, thetas)
        if sol[-1, 1] < 0:
            return B, a, Ma2, T2, p2, rho2, v_x, v_y



print("Condition en Amont:")
print("Ma1 = ", Ma1)
print("P1 = ", P1,"Pa")
print("T1 = ", T1,"K")

print()

result_plan = oblique_shock(6.95 * np.pi / 180, Ma1, T1, P1, 0.1)

print('COIN (2D):')
print('Angle de Mach: β = ', result_plan[0] * 180/np.pi)
print('Ma2 =', result_plan[2])
print('P2 =', result_plan[4],"Pa")
print('T2 =', result_plan[3],"K")

print()

result_axisym = cone_shock(6.95 * np.pi / 180, Ma1, 286.65, 90178, 0.1)

print('CÔNE (3D):')
print('Angle de Mach: β = ', result_axisym[0] * 180/np.pi)
print('Ma2 =', result_axisym[2])
print('P2 =', result_axisym[4],"Pa")
print('T2 =', result_axisym[3],"K")


print()

draw_cone_tube_horizontal(cone_length=h, cone_radius=r, tube_length=l, 
                              mach_angle_plan=result_plan[0] * 180/np.pi, 
                              mach_angle_axi=result_axisym[0] * 180/np.pi)

# Note the 3D case is weaker, because the airflow has an extra dimension to move in

#%%
# ------------------------------------------------------------------------------
# Test pour voir si la fonction de Prandl-Meyer est applicable ou non?
# ------------------------------------------------------------------------------

import math
from scipy.optimize import fsolve


# ------------------------------------------------------------------------------
# Fonction de Prandtl-Meyer
# ------------------------------------------------------------------------------
def prandl_meyer(gam, M):
    """
    Calcule la fonction de Prandtl-Meyer ν(M) (en radians) pour un écoulement supersonique.
    gam : rapport des chaleurs spécifiques (ex. 1.4 pour l'air)
    M   : nombre de Mach

    Formule :
        ν(M) = sqrt((gam + 1)/(gam - 1)) * arctan( sqrt(((gam - 1)/(gam + 1)) * (M^2 - 1)) )
               - arctan( sqrt(M^2 - 1) )
    Retourne : ν(M) en radians
    """
    nu = np.sqrt((gam + 1)/(gam - 1)) * np.arctan(np.sqrt(((gam - 1)/(gam + 1)) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1))
    return nu

Ma2 = result_axisym[2]
beta = result_axisym[0] * 180/np.pi

# ------------------------------------------------------------------------------
# 1) Calcul de l'angle ν(Ma2) en degrés
# ------------------------------------------------------------------------------
nu2 = prandl_meyer(gam, Ma2)  # en radians
nu2_deg = np.degrees(nu2)     # conversion en degrés

print("Onde de détente")

print("ν(Ma2)  = ", nu2_deg, "°")

# ------------------------------------------------------------------------------
# 2) Calcul de nu3 = delta + ν(Ma2)
#    Ici, on ajoute l'angle delta (déviation supplémentaire) à l'angle ν(Ma2).
#    ATTENTION : delta doit être en degrés ; on additionne donc en degrés.
# ------------------------------------------------------------------------------
nu3 = delta + nu2_deg
print("ν(Ma3)  = ", nu3, "°")

# ------------------------------------------------------------------------------
# 3) Définition de la fonction f(Ma3) = ν(Ma3) - ν3
#    On veut f(Ma3) = 0  =>  ν(Ma3) = ν3
#    => solve for Ma3
# ------------------------------------------------------------------------------
def f(Ma3, gam, nu3_deg):
    """
    Évalue la différence : ν(Ma3) - nu3.
    Ma3       : nombre de Mach inconnu
    gam       : gamma
    nu3_deg   : valeur cible de la fonction Prandtl-Meyer (en degrés)

    Retourne f(Ma3) = ν(Ma3) (en degrés) - nu3_deg
    """
    # ν(Ma3) en radians
    nu_Ma3 = prandl_meyer(gam, Ma3)
    # conversion en degrés
    nu_Ma3_deg = nu_Ma3 * 180/np.pi

    return nu_Ma3_deg - nu3_deg

# ------------------------------------------------------------------------------
# 4) Résolution numérique via fsolve pour trouver Ma3
#    - Initial guess : 1.2 (doit être >1 pour écoulement supersonique)
#    Utilise la foction fsolve de la bibliothèque SciPy
#    (méthode basée sur des techniques dérivées de Newton-Raphson)
# ------------------------------------------------------------------------------
Ma3_guess = 1.1
Ma3_solution = fsolve(f, Ma3_guess, args=(gam, nu3))[0]  # on prend le premier élément du tableau fsolve

print("Ma3     = ", Ma3_solution)

# ------------------------------------------------------------------------------
# 5) Angles de Mach (µ1, µ2) = arcsin(1 / M) en radians, puis conversion en degrés
# ------------------------------------------------------------------------------
mu1_rad = np.arcsin(1.0 / Ma2)
mu2_rad = np.arcsin(1.0 / Ma3_solution)

mu1_deg = mu1_rad * 180/np.pi
mu2_deg = mu2_rad * 180/np.pi

print("µ1      = ", mu1_deg, "°")
print("µ2      = ", mu2_deg, "°")

draw_cone_tube_horizontal(
    cone_length=h, 
    cone_radius=r, 
    tube_length=l, 
    mach_angle_axi=beta,
    angle_mach_amont=mu1_deg,
    angle_mach_aval=mu2_deg)
print(f"\n")