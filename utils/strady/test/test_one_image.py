"""Script pour choisir une image et un modèle et tester ce modèle spécialement sur cette image, en affichant le résultat avec les prédictions tracées dessus."""

import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
# MODEL_PATH = "runs/pose/strady_final/strady_yolo_pose_run-1/weights/best.pt"  # best.pt 1
MODEL_PATH = "runs/pose/strady_final/strady_yolo_pose_run-1/weights/best.pt"  # best.pt 3
IMAGE_PATH = "datasets/strady_raw/23_octobre_am_croppedAndResized/cam1_105010.jpg" 
# ---------------------

# 1. Charger le modèle local et lire l'image d'origine
model = YOLO(MODEL_PATH)
original_img = cv2.imread(IMAGE_PATH)

if original_img is None:
    raise FileNotFoundError(f"Impossible de charger l'image : {IMAGE_PATH}")

# 2. Exécuter l'inférence sur l'image
results = model(original_img)

# 3. Générer l'image avec les prédictions (boxes + points) tracées dessus
annotated_img = results[0].plot()

# 4. Concaténer horizontalement (Originale à gauche, Annotée à droite)
combined_img = cv2.hconcat([original_img, annotated_img])

# 5. Afficher le résultat
cv2.imshow("Originale (gauche) vs Prediction (droite)", combined_img)

# Boucle infinie qui attend explicitement 'q' ou 'Echap' pour quitter (je veux pouvoir appuyer sur control afin de zoomer numériquement mon écran sur mon mac)
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == ord('q'):  # 27 correspond à la touche Échap
        break

cv2.destroyAllWindows()