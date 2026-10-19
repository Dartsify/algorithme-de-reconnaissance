import cv2
import numpy as np
from pathlib import Path
import sys

# On importe la fonction de lecture depuis ton dossier Flux_cameras
RACINE = Path(__file__).resolve().parent.parent
sys.path.append(str(RACINE / "Flux_cameras"))
# from homographie import charger_homographies
from homographie import charger_homographies #type: ignore

DOSSIER_HOMO = Path(__file__).resolve().parent
FICHIER_MATRICES = DOSSIER_HOMO / "matrice_homographie.txt"

# Constantes exactes de ton backend
CENTER = (642, 356)
R_BULL_INNER = 8
R_BULL_OUTER = 24
R_TRIPLE_INNER = 139
R_TRIPLE_OUTER = 155
R_DOUBLE_INNER = 235
R_DOUBLE_OUTER = 251

def verifier_visuellement():
    matrices = charger_homographies(FICHIER_MATRICES)
    
    for cam_id in [1, 2, 3]:
        nom_img = f"cam{cam_id}_homographie.jpg"
        chemin_img = DOSSIER_HOMO / nom_img
        
        if not chemin_img.exists():
            print(f"[WARN] Image {nom_img} introuvable pour le test visuel.")
            continue
            
        img_inclinee = cv2.imread(str(chemin_img))
        
        # 1. On redresse toute l'image grâce à la matrice
        img_redressee = cv2.warpPerspective(img_inclinee, matrices[cam_id], (1280, 720))
        
        # 2. On dessine les anneaux mathématiques du backend par-dessus (en vert)
        for rayon in [R_BULL_INNER, R_BULL_OUTER, R_TRIPLE_INNER, R_TRIPLE_OUTER, R_DOUBLE_INNER, R_DOUBLE_OUTER]:
            cv2.circle(img_redressee, CENTER, rayon, (0, 255, 0), 2)
            
        # 3. On dessine une ligne rouge qui pointe vers le milieu du 20 (Midi / 0°)
        cv2.line(img_redressee, CENTER, (642, 356 - R_DOUBLE_OUTER), (0, 0, 255), 2)
        cv2.putText(img_redressee, "20", (630, 356 - R_DOUBLE_OUTER - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(f"Verification Camera {cam_id} (Appuyez sur une touche)", img_redressee)
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verifier_visuellement()