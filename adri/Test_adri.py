import cv2
import numpy as np
from pathlib import Path

# Ce print permet de vérifier que le script s'est bien lancé
print("[INFO] Lancement du script...")

def visualiser_homographie(chemin_image: str | Path, matrice: np.ndarray):
    print(f"[INFO] Recherche de l'image : {chemin_image}")
    
    img = cv2.imread(str(chemin_image))
    if img is None:
        print(f"[ERREUR CRITIQUE] Impossible de trouver l'image !")
        print("Vérifie que l'image est bien dans le dossier 'adri'.")
        return

    hauteur, largeur = img.shape[:2]
    print(f"[INFO] Image trouvée ({largeur}x{hauteur}). Déformation en cours...")

    # C'est LA ligne magique qui tord l'image selon ta matrice
    img_corrigee = cv2.warpPerspective(img, matrice, (largeur, hauteur))

    # Affichage
    nom_fenetre_1 = "1 - Vue Camera (Inclinee)"
    nom_fenetre_2 = "2 - Vue Corrigee (De face)"
    
    # On force l'initialisation des fenêtres sur Windows
    cv2.namedWindow(nom_fenetre_1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(nom_fenetre_2, cv2.WINDOW_NORMAL)
    
    cv2.imshow(nom_fenetre_1, cv2.resize(img, (800, int(800 * hauteur / largeur))))
    cv2.imshow(nom_fenetre_2, cv2.resize(img_corrigee, (800, int(800 * hauteur / largeur))))

    print("[INFO] Fenêtres ouvertes. Clique sur la croix d'une fenêtre pour fermer.")

    # Boucle pour maintenir ouvert jusqu'au clic sur la croix
    while True:
        cv2.waitKey(100)
        if cv2.getWindowProperty(nom_fenetre_1, cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty(nom_fenetre_2, cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyAllWindows()
    print("[INFO] Fin du programme.")

if __name__ == "__main__":
    # La matrice de ta Caméra 1 que tu m'as fournie
    matrice_cam1 = np.array([
        [-1.64115188e+00,  2.34401619e+00,  1.64859287e+03],
        [-8.86771793e-02, -1.63297571e+00,  1.23236315e+03],
        [ 1.40486855e-04,  3.60742270e-03,  1.00000000e+00]
    ], dtype=np.float32)

    # Assure-toi que l'image cam1_040526_190621.jpg est bien dans le dossier "adri"
    chemin = Path(__file__).parent / "cam1_040526_190621.jpg"
    
    visualiser_homographie(chemin, matrice_cam1)