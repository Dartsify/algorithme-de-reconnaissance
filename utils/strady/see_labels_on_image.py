"""Script pour afficher une image d'un dataset (strady_yolo_pose ou strady_yolo) avec les labels YOLO correspondants (bounding boxes et keypoints) afin de vérifier si les annotations sont correctes après l'ensemble de traitements.

Il faut choisir le dataset (strady_yolo_pose ou strady_yolo), le split (train, val ou test) et le nom de l'image (sans extension) à visualiser."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

# ---
dataset_dir = Path("datasets/strady_yolo") # ou strady_yolo
split = "train" # 'train', 'val' ou 'test'
img_name = "cam2_060526_162901_jpg.rf.1d7a4c3f82519e4f4116a3ebf2f7bf45" # Nom de l'image SANS l'extension

image_path = dataset_dir / "images" / split / f"{img_name}.jpg"
label_path = dataset_dir / "labels" / split / f"{img_name}.txt"
# ---

if not image_path.exists():
    print(f"Erreur : Image introuvable -> {image_path}")
    exit()
if not label_path.exists():
    print(f"Erreur : Fichier label introuvable -> {label_path}")
    exit()

# Charger l'image pour récupérer ses dimensions réelles
img = Image.open(image_path)
img_width, img_height = img.size

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)

# Lire les annotations YOLO depuis le fichier texte
with open(label_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) >= 5:
        cls = int(parts[0])
        x_c, y_c, w, h = map(float, parts[1:5])
        
        # 1. Convertir BBox en pixels
        x_center_px = x_c * img_width
        y_center_px = y_c * img_height
        width_px = w * img_width
        height_px = h * img_height
        
        x_min_px = x_center_px - (width_px / 2)
        y_min_px = y_center_px - (height_px / 2)
        
        couleur = 'red' if cls == 0 else 'green'
        nom_label = 'Fléchette' if cls == 0 else f'Classe {cls}'
        
        # 2. Dessiner la Bounding Box
        boite = patches.Rectangle(
            (x_min_px, y_min_px), width_px, height_px,
            linewidth=2, edgecolor=couleur, facecolor='none', label=f"BBox {nom_label}"
        )
        ax.add_patch(boite)
        
        # 3. Dessiner les Keypoints si on est en format YOLO Pose (plus de 5 colonnes)
        if len(parts) > 5:
            keypoints = parts[5:]
            for i in range(0, len(keypoints), 3):
                kx = float(keypoints[i])
                ky = float(keypoints[i+1])
                # Le 3ème élément est la visibilité, on ne l'utilise pas pour le tracé ici
                
                # Ne pas dessiner si le point est marqué comme absent (0, 0)
                if kx > 0 and ky > 0:
                    kx_px = kx * img_width
                    ky_px = ky * img_height
                    ax.plot(kx_px, ky_px, marker='X', color='blue', markersize=8, label='Pointe')

# Gestion de la légende pour éviter les doublons
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
if by_label:
    plt.legend(by_label.values(), by_label.keys())

plt.title(f"Visualisation YOLO - {img_name}")
plt.axis('off')
plt.show()
