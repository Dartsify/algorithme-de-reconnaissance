import cv2
from pathlib import Path

# Charge une image, la recadre en carré pour YOLO et affiche le résultat.
def visualiser_recadrage(chemin_image: str | Path):
    
    img = cv2.imread(str(chemin_image))
    if img is None:
        print(f"[ERREUR] Impossible de trouver l'image : {chemin_image}")
        return

    hauteur_orig, largeur_orig = img.shape[:2]
    print(f"[INFO] Image originale : {largeur_orig}x{hauteur_orig} pixels")

    # 2. Définir les coordonnées de coupe (ROI - Region of Interest)
    y_min = int(hauteur_orig * 0.1)  # On coupe les x% du haut
    y_max = hauteur_orig              # On garde l'image jusqu'en bas
    
    # Calcul pour forcer un format carré strict (idéal pour YOLO)
    taille_carre = y_max - y_min
    centre_x = largeur_orig // 2
    
    x_min = centre_x - (taille_carre // 2)
    x_max = centre_x + (taille_carre // 2)

    # 3. Sécurité : S'assurer que le carré ne déborde pas de l'image originale
    x_min = max(0, x_min)
    x_max = min(largeur_orig, x_max)

    # 4. Exécuter le recadrage (Slicing Numpy )
    img_recadree = img[y_min:y_max, x_min:x_max]

    print(f"[INFO] Image recadrée : {img_recadree.shape[1]}x{img_recadree.shape[0]} pixels")

    # 5. Affichage visuel pour comparaison
    # On redimensionne l'affichage pour que ça rentre bien sur l'écran
    nom_fenetre = "Image Recadree (Prete pour YOLO)"
    cv2.imshow(nom_fenetre, cv2.resize(img_recadree, (800, 800)))
    
    while True:
        cv2.waitKey(100)
        if cv2.getWindowProperty(nom_fenetre, cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Définition du chemin relatif (dossier 'media' parallèle au script)
    chemin = Path(__file__).parent / "cam1_020526_124254.jpg"
    
    visualiser_recadrage(chemin)