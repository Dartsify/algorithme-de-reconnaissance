from fastai.vision.all import *
import cv2
from matplotlib import pyplot as plt
import imutils


# Chemin de l'image à tester (télécharger même une image du dataset via le lien :https://drive.google.com/file/d/1GdktN4h6vyQu9NoCh68M54UAf0Nu01vQ/view?usp=sharing
path_image = 'image_test/cam2_20260415_113050_204682.jpg'
image = cv2.imread(path_image, cv2.IMREAD_COLOR)

# Charger le modèle entraîné (chemin d'accè du fichier)
# 1. (plus rapide) Récupérer le fichier 'dartsify_ai.pkl' depuis le lien : https://drive.google.com/file/d/1GdktN4h6vyQu9NoCh68M54UAf0Nu01vQ/view?usp=sharing
# 2. Ré-entraîner le modèle via le notebook dartsify_unet_fastai.ipynb et récupérer le fichier 'dartsify_ai.pkl' généré dans le dossier 'models'   
model = load_learner('dartsify_ai.pkl')

out = model.predict(path_image)


"""
Etape post-traitement de l'image de sortie du modèle pour trouver les coordonnées de la fléchette 
et les afficher sur l'image d'origine
"""
img = image[:, :, ::-1].copy() 
img_out = out[0].numpy()
img_out = cv2.convertScaleAbs(img_out)

contours = cv2.findContours(img_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for c in contours:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print("La fléchètte se trouve en :", (cX, cY))
    
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    overlay = img.copy()
    output = img.copy()
    inner_colour = (255, 255, 255)
    outer_colour = (255, 255, 255)
    cv2.circle(overlay, (cX, cY), 25, outer_colour, -1)
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)

    cv2.circle(output, (cX, cY), 25, outer_colour, 1, cv2.LINE_AA)
    cv2.circle(output, (cX, cY), 3, inner_colour, 1, cv2.LINE_AA)

plt.figure(figsize = (14, 14))
plt.imshow(output)
plt.show()