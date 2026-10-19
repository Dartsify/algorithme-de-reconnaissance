"""Code qui rogne et redimensionne les images et les annotations afin de :
 - convenir à ce qu'on avait définit pour DeepDarts (800x800) ;
 - dégager ce qui sert à rien sur l'image.

Il s'occupe également de la création du dossier pour YOLO Pose et de la réécriture des fichiers texte (labels) en conséquence.
"""

import cv2
import os
import numpy as np
from pathlib import Path

# ---
input_dataset_path = Path("datasets/strady")
output_dataset_path = Path("datasets/strady_yolo_pose")  # Chemin vers le dataset de sortie

# Nombre de pixels à rogner de chaque côté
CROP_TOP = 0    
CROP_BOTTOM = 0
# -> On ne rogne pas en haut et en bas pour garder la totalité des fléchettes au maximum
CROP_LEFT = 150  
CROP_RIGHT = 150 
TARGET_SIZE = (800, 800) # Comme les images issues de DeepDarts, on veut que les images soient carrées et de taille 800x800
# ---

splits = ['train', 'valid', 'test']

for split in splits:
    in_images_dir = input_dataset_path / split / "images"
    in_labels_dir = input_dataset_path / split / "labels"
    
    if not in_images_dir.exists() or not in_labels_dir.exists():
        continue
    
    # Renommer 'valid' en 'val' pour la nouvelle arborescence YOLO
    out_split = 'val' if split == 'valid' else split
    
    # Création des nouveaux dossiers
    out_images_dir = output_dataset_path / "images" / out_split
    out_labels_dir = output_dataset_path / "labels" / out_split
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
        
    for img_path in in_images_dir.glob("*.jpg"): # Ajuster l'extension si besoin (.png)
        # 1. Charger et rogner l'image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h_orig, w_orig = img.shape[:2]
        
        # Sécurité : ignorer les images qui n'ont pas la résolution attendue (ex: 640x480)
        if (w_orig, h_orig) != (1280, 720):
            print(f"[SKIP] {img_path.name} ignorée : résolution {w_orig}x{h_orig} != 1280x720")
            continue
        
        new_w = w_orig - CROP_LEFT - CROP_RIGHT
        new_h = h_orig - CROP_TOP - CROP_BOTTOM
        
        cropped_img = img[CROP_TOP:h_orig-CROP_BOTTOM, CROP_LEFT:w_orig-CROP_RIGHT]
        
        # 2. Redimensionner avec Letterbox (bandes grises) et sauvegarder dans le NOUVEAU dossier
        target_w, target_h = TARGET_SIZE
        scale = min(target_w / new_w, target_h / new_h)
        resized_w = int(round(new_w * scale))
        resized_h = int(round(new_h * scale))
        pad_left = (target_w - resized_w) // 2
        pad_top = (target_h - resized_h) // 2
        
        resized_crop = cv2.resize(cropped_img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        
        resized_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        resized_img[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized_crop
        
        out_img_path = out_images_dir / img_path.name
        cv2.imwrite(str(out_img_path), resized_img)
        
        # 3. Mettre à jour le fichier texte (Label)
        in_label_path = in_labels_dir / (img_path.stem + ".txt")
        out_label_path = out_labels_dir / (img_path.stem + ".txt")
        
        if in_label_path.exists():
            with open(in_label_path, 'r') as file:
                lines = file.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = parts[0]
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])
                    
                    # Relatif -> Pixels absolus
                    abs_x = x_center_norm * w_orig
                    abs_y = y_center_norm * h_orig
                    abs_w = width_norm * w_orig
                    abs_h = height_norm * h_orig
                    
                    # Appliquer le décalage du rognage et borner la boîte aux limites de l'image rognée
                    x1 = max(0.0, (abs_x - abs_w / 2) - CROP_LEFT)
                    y1 = max(0.0, (abs_y - abs_h / 2) - CROP_TOP)
                    x2 = min(float(new_w), (abs_x + abs_w / 2) - CROP_LEFT)
                    y2 = min(float(new_h), (abs_y + abs_h / 2) - CROP_TOP)
                    
                    # Sécurité : vérifier que la box est toujours valide dans l'image rognée
                    if x2 > x1 and y2 > y1:
                        new_abs_x = (x1 + x2) / 2
                        new_abs_y = (y1 + y2) / 2
                        new_abs_w = x2 - x1
                        new_abs_h = y2 - y1
                        
                        # Repasser en relatif dans le canvas final (800x800 avec bandes grises)
                        new_x_norm = (new_abs_x * scale + pad_left) / target_w
                        new_y_norm = (new_abs_y * scale + pad_top) / target_h
                        new_w_norm = (new_abs_w * scale) / target_w
                        new_h_norm = (new_abs_h * scale) / target_h

                        new_line = f"{cls_id} {new_x_norm:.6f} {new_y_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}"
                        
                        if len(parts) > 5:
                            keypoints = parts[5:]
                            # Les points sont formatés en px, py, visibilité (groupes de 3)
                            for i in range(0, len(keypoints), 3):
                                kx_norm = float(keypoints[i])
                                ky_norm = float(keypoints[i+1])
                                v = keypoints[i+2] if i+2 < len(keypoints) else "0"
                                
                                # Si le point n'est pas annoté (0, 0, 0)
                                if kx_norm == 0 and ky_norm == 0:
                                    new_line += f" 0.000000 0.000000 0"
                                    continue
                                
                                # Décalage du rognage pour le point
                                new_abs_kx = (kx_norm * w_orig) - CROP_LEFT
                                new_abs_ky = (ky_norm * h_orig) - CROP_TOP
                                
                                # Si le point se retrouve hors de la nouvelle image rognée, on l'annule
                                if not (0 <= new_abs_kx <= new_w and 0 <= new_abs_ky <= new_h):
                                    new_kx_norm, new_ky_norm, v = 0.0, 0.0, "0"
                                else:
                                    new_kx_norm = (new_abs_kx * scale + pad_left) / target_w
                                    new_ky_norm = (new_abs_ky * scale + pad_top) / target_h
                                    
                                new_line += f" {new_kx_norm:.6f} {new_ky_norm:.6f} {v}"

                        new_lines.append(new_line + "\n")
            
            # Écrire le nouveau fichier texte dans le NOUVEAU dossier
            with open(out_label_path, 'w') as file:
                file.writelines(new_lines)

in_yaml_path = input_dataset_path / "data.yaml"
out_yaml_path = output_dataset_path / "data.yaml"

if in_yaml_path.exists():
    with open(in_yaml_path, 'r') as file:
        yaml_lines = file.readlines()
        
    new_yaml_lines = ["# Fichier généré automatiquement par utils/strady/crop_and_resize.py sur base du fichier data.yaml extrait du dataset Roboflow\n"]
    for line in yaml_lines:
        # On arrête de lire si on tombe sur la section roboflow
        if line.startswith("roboflow:"):
            break
            
        # On met à jour les chemins pour correspondre à la nouvelle arborescence
        if line.startswith("train:"):
            new_yaml_lines.append("train: images/train\n")
        elif line.startswith("val:") or line.startswith("valid:"):
            new_yaml_lines.append("val: images/val\n")
        elif line.startswith("test:"):
            new_yaml_lines.append("test: images/test\n")
        else:
            new_yaml_lines.append(line)
            
    with open(out_yaml_path, 'w') as file:
        file.writelines(new_yaml_lines)
    print("Fichier data.yaml copié et mis à jour.")

print(f"Opération terminée. Le dataset final est prêt dans : {output_dataset_path.absolute()}")
