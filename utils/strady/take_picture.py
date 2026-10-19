"""Script pour remplir le dossier strady_raw de nos images pour compléter le dataset de Strady."""

import cv2
import os
import time
from datetime import datetime

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Forcer le MJPG permet souvent de débloquer le 720p/1080p à haut framerate sur Windows
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Toujours vérifier que la caméra a bien accepté la résolution
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Résolution active : {actual_w}x{actual_h}")

if (actual_w, actual_h) != (1280, 720):
    raise RuntimeError("Attention : La caméra n'est pas en 1280x720 !")

# Configuration
base_dir = "datasets/strady_raw"
camera_indices = [0, 1, 2]

# Création du nom du sous-dossier (ex: "7_octobre_pm")
mois_fr = ["", "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre", "novembre", "decembre"]
now = datetime.now()
jour = now.day
mois = mois_fr[now.month]
periode = "am" if now.hour < 12 else "pm"

# La caméra 2 dans mon ordi est la caméra 3 physique -> à changer à chaque fois qu'on change d'ordi ou rebranche !!
cameras = {
    1: 2,
    2: 3,
    3: 1
}

session_folder = f"{jour}_{mois}_{periode}"
output_dir = os.path.join(base_dir, session_folder)

# Création des dossiers s'ils n'existent pas
os.makedirs(output_dir, exist_ok=True)

# Horodatage simplifié pour les fichiers (la date est déjà dans le dossier)
timestamp = now.strftime("%H%M%S")

for i, cam_idx in enumerate(camera_indices):
    print(f"[INFO] Initialisation de la caméra {cameras[cam_idx + 1]}...")
    cap = cv2.VideoCapture(cam_idx)
    
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la caméra index {cameras[cam_idx + 1]}.")
        continue
        
    # Laisser l'exposition s'ajuster
    time.sleep(1.0)
    
    # Vider le buffer
    for _ in range(5):
        cap.read()
        
    # Capture
    ret, frame = cap.read()
    
    if ret:
        filename = f"cam{cameras[cam_idx + 1]}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"[SUCCÈS] Sauvegardé : {filepath}")
    else:
        print(f"[ERREUR] Échec de la capture sur la caméra {cameras[cam_idx + 1]}.")
        
    cap.release()

print("\n[INFO] Fin du script.")