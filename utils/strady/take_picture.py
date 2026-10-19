"""Script pour remplir le dossier strady_raw de nos images pour compléter le dataset de Strady."""

import cv2
import os
import time
from datetime import datetime

# Configuration
base_dir = "datasets/strady_raw"
# Mapping index OpenCV -> numéro physique de la caméra.
# Vérifier les index sur cet ordinateur : ils peuvent varier selon le branchement.
camera_mapping = {
    1: 2,
    2: 3,
    3: 1,
}

# Création du nom du sous-dossier (ex: "7_octobre_pm")
mois_fr = ["", "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre", "novembre", "decembre"]
now = datetime.now()
jour = now.day
mois = mois_fr[now.month]
periode = "am" if now.hour < 12 else "pm"

session_folder = f"{jour}_{mois}_{periode}"
output_dir = os.path.join(base_dir, session_folder)

# Création des dossiers s'ils n'existent pas
os.makedirs(output_dir, exist_ok=True)

# Horodatage simplifié pour les fichiers (la date est déjà dans le dossier)
timestamp = now.strftime("%H%M%S")

for cam_idx, camera_number in camera_mapping.items():
    print(f"[INFO] Initialisation de la caméra {camera_number} (index OpenCV {cam_idx})...")
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la caméra {camera_number} (index OpenCV {cam_idx}).")
        continue
        
    # Laisser l'exposition s'ajuster
    time.sleep(1.0)
    
    # Vider le buffer
    for _ in range(5):
        cap.read()
        
    # Capture
    ret, frame = cap.read()
    
    if ret:
        filename = f"cam{camera_number}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"[SUCCÈS] Sauvegardé : {filepath}")
    else:
        print(f"[ERREUR] Échec de la capture sur la caméra {camera_number} (index OpenCV {cam_idx}).")
        
    cap.release()

print("\n[INFO] Fin du script.")
