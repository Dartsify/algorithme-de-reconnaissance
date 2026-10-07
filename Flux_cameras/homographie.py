import re
from pathlib import Path
import numpy as np
import cv2

# Charge les matrices calibrées depuis un fichier texte (dans Homographie/matrice_homographie.txt).
def charger_homographies(fichier_calibration: Path) -> dict[int, np.ndarray]:
    
    if not fichier_calibration.exists():
        raise FileNotFoundError(
            f"\n[ERREUR CRITIQUE] Fichier d'homographie introuvable à l'emplacement : {fichier_calibration}\n"
            "-> Veuillez exécuter 'Homographie/outil_calibration_visuelle.py' pour générer ce fichier."
        )

    content = fichier_calibration.read_text(encoding="utf-8")
    float_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    matrices = {}

    for match in re.finditer(r"CAMERA\s+(\d+)\s*(\[\[.*?\]\])", content, re.S):
        camera_id = int(match.group(1))
        matrix_text = match.group(2)
        nombres = [float(v) for v in re.findall(float_pattern, matrix_text)]
        
        if len(nombres) == 9:
            matrices[camera_id] = np.array(nombres, dtype=np.float32).reshape(3, 3)

    if len(matrices) != 3:
        raise ValueError(
            f"\n[ERREUR CRITIQUE] Le fichier de calibration contient {len(matrices)} matrice(s) au lieu de 3.\n"
            "-> Veuillez refaire la calibration complète des 3 caméras."
        )

    print("[INFO] Matrices d'homographie chargées avec succès depuis le fichier.")
    return matrices


# Projette les coordonnées de la pointe vues par une caméra inclinée 
# pour obtenir la position (x, y) de face sur la cible parfaite.
def projeter_point_sur_cible(x_camera: float, y_camera: float, matrice_homographie: np.ndarray) -> tuple[float, float]:
    
    point_numpy = np.array([[[float(x_camera), float(y_camera)]]], dtype=np.float32)
    point_transforme = cv2.perspectiveTransform(point_numpy, matrice_homographie)
    
    x_cible = float(point_transforme[0, 0, 0])
    y_cible = float(point_transforme[0, 0, 1])
    
    return x_cible, y_cible




# --- ZONE DE TEST ISOLÉE ---
if __name__ == "__main__":
    print("--- TEST DE LA PROJECTION (HOMOGRAPHIE) ---")
    
    # Pointe vers le dossier Homographie à la racine du projet
    CHEMIN_FICHIER_CALIBRATION = Path(__file__).resolve().parent.parent / "Homographie" / "matrice_homographie"
    
    try:
        matrices = charger_homographies(CHEMIN_FICHIER_CALIBRATION)
        
        camera_test = 1
        pointe_x_brute = 640.0
        pointe_y_brute = 360.0 
        
        print(f"[INFO] Pointe brute sur Caméra {camera_test} : X={pointe_x_brute}, Y={pointe_y_brute}")
        x_final, y_final = projeter_point_sur_cible(pointe_x_brute, pointe_y_brute, matrices[camera_test])
        print(f"[SUCCÈS] Coordonnées réelles projetées sur la cible 2D : X={x_final:.2f}, Y={y_final:.2f}")

    except Exception as e:
        print(e)