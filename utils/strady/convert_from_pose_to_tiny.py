"""Code qui convertit les labels de YOLO Pose en format YOLO Tiny -> dégager les bbox, garder uniquement le keypoint de la pointe."""

import shutil
import os
from pathlib import Path

# ---
input_dataset_path = Path("datasets/strady_yolo_pose")
output_dataset_path = Path("datasets/strady_yolo")

# Taille de la BBox DeepDarts : 20px sur 800px = 0.025
BBOX_SIZE = 0.025
splits = ['train', 'val', 'test']
# ---

for split in splits:
    in_images_dir = input_dataset_path / "images" / split
    in_labels_dir = input_dataset_path / "labels" / split
    
    if not in_images_dir.exists() or not in_labels_dir.exists():
        continue
    
    out_images_dir = output_dataset_path / "images" / split
    out_labels_dir = output_dataset_path / "labels" / split
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
        
    for img_path in in_images_dir.glob("*.jpg"):
        # 1. Copier l'image directement (déjà rognée et redimensionnée)
        out_img_path = out_images_dir / img_path.name
        shutil.copy(img_path, out_img_path)
        
        # 2. Traitement des Labels (Conversion DeepDarts)
        in_label_path = in_labels_dir / (img_path.stem + ".txt")
        out_label_path = out_labels_dir / (img_path.stem + ".txt")
        
        if in_label_path.exists():
            with open(in_label_path, 'r') as file:
                lines = file.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 5:
                    cls_id = parts[0]
                    
                    # Cible le premier keypoint (x=parts[5], y=parts[6]) = pointe
                    kx_norm = float(parts[5])
                    ky_norm = float(parts[6])
                    
                    if kx_norm == 0 and ky_norm == 0:
                        continue
                        
                    # Construction du label YOLO Classique centré sur la pointe
                    new_line = f"{cls_id} {kx_norm:.6f} {ky_norm:.6f} {BBOX_SIZE:.6f} {BBOX_SIZE:.6f}\n"
                    new_lines.append(new_line)
            
            with open(out_label_path, 'w') as file:
                file.writelines(new_lines)

# 3. Copier et adapter data.yaml
in_yaml_path = input_dataset_path / "data.yaml"
out_yaml_path = output_dataset_path / "data.yaml"

if in_yaml_path.exists():
    with open(in_yaml_path, 'r') as file:
        yaml_lines = file.readlines()
        
    new_yaml_lines = ["# Fichier généré automatiquement pour YOLO Classique (DeepDarts)\n"]
    
    for line in yaml_lines:
        # On supprime les paramètres exclusifs à YOLO Pose
        if line.startswith("kpt_shape:") or line.startswith("flip_idx:") or line.startswith("#"):
            continue
        new_yaml_lines.append(line)
            
    with open(out_yaml_path, 'w') as file:
        file.writelines(new_yaml_lines)

print(f"Conversion terminée. Dataset strady_yolo prêt dans : {output_dataset_path.absolute()}")
