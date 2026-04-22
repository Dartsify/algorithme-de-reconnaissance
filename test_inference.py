from fastai.learner import load_learner
import pathlib
import cv2
from matplotlib import pyplot as plt
import imutils
import numpy as np


def main():
    # Chemin de l'image à tester (télécharger même une image du dataset via le lien :https://drive.google.com/file/d/1GdktN4h6vyQu9NoCh68M54UAf0Nu01vQ/view?usp=sharing
    path_image = 'Dartsify/dataset/cam1_080426_215714.jpg'
    image = cv2.imread(path_image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {path_image}")

    # Charger le modèle entraîné (chemin d'accè du fichier)
    # 1. (plus rapide) Récupérer le fichier 'dartsify_ai.pkl' depuis le lien : https://drive.google.com/file/d/1GdktN4h6vyQu9NoCh68M54UAf0Nu01vQ/view?usp=sharing
    # 2. Ré-entraîner le modèle via le notebook dartsify_unet_fastai.ipynb et récupérer le fichier 'dartsify_ai.pkl' généré dans le dossier 'models'

    # """
    # --------------------------------------------------
    #  load_learner requires all your custom code 
    #  be in the exact same place as when exporting your Learner (the main script, or the module you imported it from).
    # --------------------------------------------------
    # """   
    # def get_x(dataframe_row):
    #     return f"Dartsify/dataset/{dataframe_row['ID']}"

    # def get_mask(dataframe_row):
    #     return f"Dartsify/masks/{dataframe_row['ID']}"

    # def accuracy_camvid(inp, targ):
    #   targ = targ.squeeze(1)
    #   mask = targ != void_code
    #   return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()

    # Windows compatibility for FastAI .pkl exported on Linux
    if pathlib.PosixPath is not pathlib.WindowsPath:
        pathlib.PosixPath = pathlib.WindowsPath

    model = load_learner('dartsify_ai_optuna_dicejaccard_20epochs.pkl')

    out = model.predict(path_image)

    # récupérer masque
    mask = out[0].numpy().astype(np.uint8)

    """
    Etape post-traitement de l'image de sortie du modèle pour trouver les coordonnées de la fléchette 
    et les afficher sur l'image d'origine
    """
    mask = cv2.convertScaleAbs(mask)

    # trouver les contours de la fléchette
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    output = image.copy()

    for c in contours:
        area = cv2.contourArea(c)
        print(f"Aire détectée : {area}") # Observe les petites valeurs vs les grandes
        if cv2.contourArea(c) < 50:
            continue

        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print("La fléchètte se trouve en :", (cX, cY))

        overlay = output.copy()

        color = (255, 255, 255)
        cv2.circle(overlay, (cX, cY), 25, color, -1)

        alpha = 0.2
        cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)

        cv2.circle(output, (cX, cY), 3, color, -1)

    plt.figure(figsize = (12, 12))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    main()