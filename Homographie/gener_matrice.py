import cv2
import numpy as np
from pathlib import Path

# Comment l'utiliser pour tes 3 caméras :
# Tu remplis les 20 noms et les 20 coordonnées parfaites (X, Y) dans le code une seule fois.

# Tu lances le script pour la Caméra 1, tu cliques tes 20 points, et tu notes/sauvegardes la matrice générée.

# Tu changes la variable CHEMIN_IMAGE_CAM pour pointer vers la photo de la Caméra 2, tu relances le script, et tu cliques les 20 mêmes points sur ce nouvel angle de vue.

# Tu répètes pour la Caméra 3.



# 1. PARAMÈTRES À CHANGER AVANT CHAQUE TEST
CAMERA_A_CALIBRER = 1  # Changer pour 2 puis 3 quand fais les autres
CHEMIN_IMAGE_CAM = "cam1_reference.jpg"  # La photo prise par cette caméra

# Fichier de sortie qui sera lu par pipeline
DOSSIER_ACTUEL = Path(__file__).resolve().parent
FICHIER_SORTIE = DOSSIER_ACTUEL / "matrice_homographie"

# 2. LES 20 POINTS PARFAITS (BASÉS SUR LE BACKEND)
NOMS_POINTS = [
    "Intersection 20/1 (Double Interne)", "Intersection 1/18 (Double Interne)",
    "Intersection 18/4 (Double Interne)", "Intersection 4/13 (Double Interne)",
    "Intersection 13/6 (Double Interne)", "Intersection 6/10 (Double Interne)",
    "Intersection 10/15 (Double Interne)", "Intersection 15/2 (Double Interne)",
    "Intersection 2/17 (Double Interne)", "Intersection 17/3 (Double Interne)",
    "Intersection 3/19 (Double Interne)", "Intersection 19/7 (Double Interne)",
    "Intersection 7/16 (Double Interne)", "Intersection 16/8 (Double Interne)",
    "Intersection 8/11 (Double Interne)", "Intersection 11/14 (Double Interne)",
    "Intersection 14/9 (Double Interne)", "Intersection 9/12 (Double Interne)",
    "Intersection 12/5 (Double Interne)", "Intersection 5/20 (Double Interne)"
]

pts_destination = np.array([
    [678.8, 123.9], [748.7, 146.6], [808.2, 190.0], [851.3, 249.2], [874.1, 319.3],
    [874.1, 392.7], [851.3, 462.8], [808.2, 522.0], [748.7, 565.4], [678.8, 588.1],
    [605.2, 588.1], [535.3, 565.4], [475.8, 522.0], [432.7, 462.8], [409.9, 392.7],
    [409.9, 319.3], [432.7, 249.2], [475.8, 190.0], [535.3, 146.6], [605.2, 123.9]
], dtype=np.float32)

# 3. INTERFACE DE CLIC ET CALCUL
points_cliques = []
img_affichage = None
index_point = 0


# Ajoute ou met à jour la matrice dans le fichier texte.
def sauvegarder_matrice(matrice):
    contenu = ""
    if FICHIER_SORTIE.exists():
        contenu = FICHIER_SORTIE.read_text(encoding="utf-8")
    
    # Formatage de la matrice
    matrice_str = np.array2string(matrice, separator=', ', formatter={'float_kind':lambda x: f"{x:e}"})
    bloc_matrice = f"CAMERA {CAMERA_A_CALIBRER}\n{matrice_str}\n\n"
    
    # Remplacement si la caméra existe déjà, sinon ajout
    if f"CAMERA {CAMERA_A_CALIBRER}" in contenu:
        import re
        contenu = re.sub(rf"CAMERA {CAMERA_A_CALIBRER}\n\[\[.*?\]\]\n\n", bloc_matrice, contenu, flags=re.DOTALL)
    else:
        if not contenu.endswith("\n\n") and contenu != "":
            contenu += "\n\n"
        contenu += bloc_matrice
        
    FICHIER_SORTIE.write_text(contenu, encoding="utf-8")
    print(f"\n[SUCCÈS] Matrice de la Caméra {CAMERA_A_CALIBRER} enregistrée dans {FICHIER_SORTIE.name}")



def clic_souris(event, x, y, flags, param):
    global points_cliques, img_affichage, index_point
    if event == cv2.EVENT_LBUTTONDOWN and index_point < 20:
        points_cliques.append([float(x), float(y)])
        cv2.circle(img_affichage, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img_affichage, str(index_point + 1), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        index_point += 1
        mettre_a_jour_affichage()



def mettre_a_jour_affichage():
    img_temp = img_affichage.copy()
    cv2.rectangle(img_temp, (0, 0), (img_temp.shape[1], 40), (0, 0, 0), -1)
    
    if index_point < 20:
        txt = f"Cam {CAMERA_A_CALIBRER} - Cliquez : {NOMS_POINTS[index_point]} ({index_point + 1}/20)"
        cv2.putText(img_temp, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(img_temp, "Termine ! Appuyez sur ENTREE pour valider ou R pour recommencer.", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Calibration", img_temp)



def main():
    global img_affichage, points_cliques, index_point
    chemin_img = DOSSIER_ACTUEL / CHEMIN_IMAGE_CAM
    
    if not chemin_img.exists():
        print(f"[ERREUR] Mettez une image nommée {CHEMIN_IMAGE_CAM} dans le dossier !")
        return

    img_originale = cv2.imread(str(chemin_img))
    img_affichage = img_originale.copy()

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", clic_souris)
    mettre_a_jour_affichage()

    while True:
        touche = cv2.waitKey(1) & 0xFF
        if touche == 27: # Echap
            break
        elif touche == ord('r'):
            points_cliques.clear()
            index_point = 0
            img_affichage = img_originale.copy()
            mettre_a_jour_affichage()
        elif touche == 13 and len(points_cliques) == 20: # Entree
            matrice, _ = cv2.findHomography(np.array(points_cliques, dtype=np.float32), pts_destination)
            sauvegarder_matrice(matrice)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()