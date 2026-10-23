"""Code qui rogne et redimensionne les images d'un dossier de test afin de :
 - convenir à ce qu'on avait défini pour DeepDarts (800x800) ;
 - dégager ce qui ne sert à rien sur l'image avant inférence avec YOLO Pose.
"""

import cv2
from pathlib import Path

# --- CONFIGURATION ---
input_dir = Path("img_tests") # À modifier avec le bon chemin désiré !!

# Nombre de pixels à rogner de chaque côté (même convention)
CROP_TOP = 0    
CROP_BOTTOM = 0
# -> On ne rogne pas en haut et en bas pour garder la totalité des fléchettes au maximum !! 
CROP_LEFT = 150  
CROP_RIGHT = 150 
TARGET_SIZE = (800, 800)
# ---------------------

# Création automatique du dossier <nom>_croppedAndResized au même niveau
output_dir = input_dir.with_name(f"{input_dir.name}_croppedAndResized")
output_dir.mkdir(parents=True, exist_ok=True)

valid_extensions = {".jpg", ".jpeg", ".png"}
count = 0

for img_path in input_dir.iterdir():
    if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
        # 1. Charger l'image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Impossible de lire l'image : {img_path.name}")
            continue
            
        h_orig, w_orig = img.shape[:2]
        
        # Vérification de sécurité pour éviter un crash si une image est déjà trop petite
        if w_orig <= (CROP_LEFT + CROP_RIGHT) or h_orig <= (CROP_TOP + CROP_BOTTOM):
            print(f"Ignorée (dimensions {w_orig}x{h_orig} trop petites pour le crop) : {img_path.name}")
            continue
            
        # 2. Rogner l'image
        cropped_img = img[CROP_TOP:h_orig-CROP_BOTTOM, CROP_LEFT:w_orig-CROP_RIGHT]
        
        # 3. Redimensionner et sauvegarder
        resized_img = cv2.resize(cropped_img, TARGET_SIZE)
        out_img_path = output_dir / img_path.name
        cv2.imwrite(str(out_img_path), resized_img)
        count += 1

print(f"Opération terminée : {count} image(s) traitée(s).")
print(f"Dossier de sortie : {output_dir.absolute()}")
