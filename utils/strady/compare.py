import os
import glob
import shutil

# --- Chemins à adapter selon ton arborescence ---
labels_dir = "datasets/strady_yolo_pose/labels/test"          # Dossier de tes labels réels (Ground Truth)
preds_dir = "runs/pose/test_results-10/labels"       # Dossier des prédictions (généré avec save_txt=True)
images_dir = "datasets/strady_yolo_pose/images/test"          # Dossier de tes images
output_dir = "bad_predictions"             # Dossier de sortie pour l'inspection

os.makedirs(output_dir, exist_ok=True)

# Parcourir les images pour identifier les Faux Positifs sur les images vierges
img_files = glob.glob(f"{images_dir}/*.jpg") # Adapte l'extension si tu utilises des .png

erreurs_trouvees = 0

for img_path in img_files:
    filename = os.path.basename(img_path)
    txt_name = filename.replace('.jpg', '.txt').replace('.png', '.txt')
    
    label_file = os.path.join(labels_dir, txt_name)
    pred_file = os.path.join(preds_dir, txt_name)

    # Lire Ground Truth (si le fichier n'existe pas ou est vide = image vierge/background)
    gts = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            gts = [line for line in f.readlines() if line.strip()]

    # Lire Prédictions
    preds = []
    if os.path.exists(pred_file):
        with open(pred_file, 'r') as f:
            preds = [line for line in f.readlines() if line.strip()]

    # S'il y a une différence entre le nombre d'objets réels et prédits
    if len(preds) != len(gts):
        erreurs_trouvees += 1
        # Format du nom : diff_[Nb_Pred]P_[Nb_GT]GT_nom_image.jpg
        # Ex: diff_1P_0GT_image.jpg indique un faux positif sur une image vierge
        new_name = f"diff_{len(preds)}P_{len(gts)}GT_{filename}"
        shutil.copy(img_path, os.path.join(output_dir, new_name))

print(f"Terminé. {erreurs_trouvees} images avec de mauvaises prédictions copiées dans '{output_dir}'.")
