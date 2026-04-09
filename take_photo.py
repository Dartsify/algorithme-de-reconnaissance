import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from random import randint

message = ["Bien joué !!!!!!!", "Allez ! Encore quelques images et c'est bon", "Continue comme ça", "Excellent !!", "Oh le champion !", "Parfait", "Magnifique !", "Le modèle IA va en manger !", "Incroyable !!", "Fantastique !"]

DATASET_DIR = Path('dataset')
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Définir les caméras par son numéro d'index

# 3 -> cam 3
cam1 = cv2.VideoCapture(3)
if not cam1.isOpened():
	exit("Erreur: Impossible d'ouvrir la caméra1")

# 0 -> cam 1
cam2 = cv2.VideoCapture(0)
if not cam2.isOpened():
	exit("Erreur: Impossible d'ouvrir la caméra2")

# 1 -> cam 2
cam3 = cv2.VideoCapture(1)
if not cam3.isOpened():
	exit("Erreur: Impossible d'ouvrir la caméra3")

# Définir la résolution de chaque caméra (1280x720, format par défaut des caméras)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cam3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Boucle de capture et d'affichage du flux vidéo
while True:
	success1, frame1 = cam1.read()
	if not success1:
		print("Erreur: Impossible de lire le flux vidéo de la caméra1")
		break

	success2, frame2 = cam2.read()
	if not success2:
		print("Erreur: Impossible de lire le flux vidéo de la caméra2")
		break

	success3, frame3 = cam3.read()
	if not success3:
		print("Erreur: Impossible de lire le flux vidéo de la caméra3")
		break

	# Afficher le flux vidéo de chaque caméra dans une fenêtre, individuellement
	# cv2.imshow('Caméra 1', frame1)
	# cv2.imshow('Caméra 2', frame2)
	# cv2.imshow('Caméra 3', frame3)

	# Redimensionner les flux vidéo pour les afficher côte à côte 
	# (optionnel mais efficace pour mieux visualiser)
	frame_rz1 = cv2.resize(frame1, (500, 500))
	frame_rz2 = cv2.resize(frame2, (500, 500))
	frame_rz3 = cv2.resize(frame3, (500, 500))
	# Combiner horizontalement les trois flux vidéo
	combined_frame = np.hstack((frame_rz1, frame_rz2, frame_rz3))

	# Afficher le flux vidéo combiné dans une même fenêtre
	cv2.imshow('Vue d\'ensemble', combined_frame)

	key = cv2.waitKey(1) & 0xFF

	# Appuyer sur ESPACE pour enregistrer la capture
	if key == 32:  # 32 est le code ASCII pour ESPACE
		ts = datetime.now().strftime('%d%m%y_%H%M%S')
		ok1 = cv2.imwrite(str(DATASET_DIR / f'cam3_{ts}.jpg'), frame1)
		ok2 = cv2.imwrite(str(DATASET_DIR / f'cam1_{ts}.jpg'), frame2)
		ok3 = cv2.imwrite(str(DATASET_DIR / f'cam2_{ts}.jpg'), frame3)

		if not (ok1 and ok2 and ok3):
			print("Erreur: échec d'enregistrement (cv2.imwrite a renvoyé False).")
		else:
			print(f"{message[randint(0, len(message)-1)]}")
			print(f'Nombre d\'images dans le dataset: {len(list(DATASET_DIR.glob("*.jpg")))}')

	# Appuyer sur ESC pour arrêter la capture
	if key == 27:  # 27 est le code ASCII pour ESC
		break

# Release the capture and writer objects
cam1.release()
cam2.release()
cam3.release()
cv2.destroyAllWindows()