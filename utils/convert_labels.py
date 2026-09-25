"""Script pour convertir les labels du fichier PKL en fichiers .txt au format YOLO. Chaque image aura un fichier .txt correspondant contenant les coordonnées des boîtes englobantes et la classe associée. Les fichiers .txt seront créés dans les dossiers "labels/train" et "labels/val" selon l'emplacement de l'image (train ou val)."""

import pandas as pd
import os
from pathlib import Path

# --- CONFIGURATION DES CHEMINS ---
base_dir = "datasets/deepdarts_d1_yolo"
pkl_file = os.path.join(base_dir, "labels.pkl")

# Dossiers d'images existants
img_train_dir = os.path.join(base_dir, "images/train")
img_val_dir = os.path.join(base_dir, "images/val")

# Dossiers de labels à créer
lbl_train_dir = os.path.join(base_dir, "labels/train")
lbl_val_dir = os.path.join(base_dir, "labels/val")
# ---------------------------------

# Création des dossiers de destination
os.makedirs(lbl_train_dir, exist_ok=True)
os.makedirs(lbl_val_dir, exist_ok=True)

df = pd.read_pickle(pkl_file)

count_train, count_val, count_not_found = 0, 0, 0

for index, row in df.iterrows():
    img_name = row['img_name'] 
    boxes = row['labels']
    
    # Vérification de l'emplacement de l'image
    if os.path.exists(os.path.join(img_train_dir, img_name)):
        output_dir = lbl_train_dir
        count_train += 1
    elif os.path.exists(os.path.join(img_val_dir, img_name)):
        output_dir = lbl_val_dir
        count_val += 1
    else:
        print(f"⚠️ Image introuvable : {img_name}")
        count_not_found += 1
        continue # On passe à la suivante
    
    txt_filename = Path(img_name).stem + ".txt"
    txt_filepath = os.path.join(output_dir, txt_filename)
    
    with open(txt_filepath, 'w') as txt_file:
        if len(boxes) > 0:
            for box in boxes:
                class_id = int(box[0])
                x_center = float(box[1])
                y_center = float(box[2])
                width = float(box[3])
                height = float(box[4])
                txt_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print(f"Conversion terminée !")
print(f"Labels Train : {count_train}")
print(f"Labels Val   : {count_val}")
if count_not_found > 0:
    print(f"Attention : {count_not_found} images du PKL n'étaient ni dans train ni dans val.")
