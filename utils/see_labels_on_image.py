"""Visualiser les labels YOLO sur une image donnée afin de tester si les annotations sont correctes (après l'ensemble des conversions)."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import ast

# --- PARAMÈTRES ---
img_name = "d1_02_04_2020__IMG_1083"
image_path = "datasets/deepdarts_d1_yolo/images/train/" + img_name + ".jpg"
image_size = 800

# Format YOLO (converti depuis le brut par le script 'convert_labels_to_yolo_format.py') : [[classe, x_centre, y_centre, largeur, hauteur], ...]
labels_brut = "[[0, 0.527560763888889, 0.19943576388888892, 0.025, 0.025], [0, 0.42339409722222227, 0.4338107638888889, 0.025, 0.025]]" # Prendre les valeurs depuis deepdarts_d1_yolo/labels.pkl pour l'image correspondante
# ------------------

if not os.path.exists(image_path):
    print("Erreur : Image introuvable.")
    exit()

# Transformation de la chaîne en liste Python
annotations = ast.literal_eval(labels_brut)

# Charger l'image
img = Image.open(image_path)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)

# Parcourir chaque annotation YOLO
for ann in annotations:
    cls, x_c, y_c, w, h = ann
    
    # 1. Convertir les valeurs normalisées en pixels
    x_center_px = x_c * image_size
    y_center_px = y_c * image_size
    width_px = w * image_size
    height_px = h * image_size
    
    # 2. Calculer le coin supérieur gauche pour la Bounding Box
    x_min_px = x_center_px - (width_px / 2)
    y_min_px = y_center_px - (height_px / 2)
    
    # Définir la couleur selon la classe (0 = Fléchette, >0 = Calibrage si vous en avez laissé)
    couleur = 'red' if cls == 0 else 'green'
    nom_label = 'Fléchette' if cls == 0 else f'Calibrage {int(cls)}'
    
    # 3. Dessiner le point d'impact central (Croix)
    ax.plot(x_center_px, y_center_px, marker='X', color=couleur, markersize=8, label=nom_label)
    
    # 4. Dessiner la Bounding Box
    boite = patches.Rectangle(
        (x_min_px, y_min_px), width_px, height_px,
        linewidth=2, edgecolor=couleur, facecolor='none'
    )
    ax.add_patch(boite)

# Gestion de la légende (pour éviter de répéter "Fléchette" à chaque fois)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
if by_label:
    plt.legend(by_label.values(), by_label.keys())

plt.title("Visualisation YOLO - Bounding Boxes (2.5%)")
plt.show()
