import os

# Désactive la recherche de transformations matérielles de MSMF
# Cela fait passer l'ouverture de 15 secondes à 0.5 seconde par caméra
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from random import randint
import concurrent.futures # pour le traitement en parallèle des caméras

message = ["Bien joué !!!!!!!", "Allez ! Encore quelques images et c'est bon", "Continue comme ça", "Excellent !!", "Oh le champion !", "Parfait", "Magnifique !", "Le modèle IA va en manger !", "Incroyable !!", "Fantastique !"]

DATASET_DIR = Path('Dartsify_mur/dataset_mur/')
DATASET_DIR.mkdir(parents=True, exist_ok=True)
HEIGHT, WIDTH = 720, 1280

def ouvrir_une_camera(camera_id):
	"""Tente d'ouvrir une seule caméra (Multithreadé)."""
	# cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) # rends très instable les flux vidéo (fps très nul)
	cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF) # on force explicitement l'api MSMF
	
	if cap.isOpened():
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
		return camera_id, cap
	else:
		print(f"Échec de la caméra {camera_id}.")
		cap.release()
		return camera_id, None


camera_index = [3, 0, 2] # Les index des 3 caméras
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
	# Lance les 3 ouvertures exactement au même moment
	resultats = list(executor.map(ouvrir_une_camera, camera_index))
	print("Résultats de l'ouverture des caméras :", resultats)

	cameras_ouvertes = {camera_id: cap for camera_id, cap in resultats if cap is not None}
	if len(cameras_ouvertes) != len(camera_index):
		for cap in cameras_ouvertes.values():
			cap.release()
		exit("Erreur: impossible d'ouvrir les 3 caméras en parallèle")


# Associer les identifiants physiques aux trois flux attendus
cam1 = cameras_ouvertes[camera_index[0]]  # Caméra 1
cam2 = cameras_ouvertes[camera_index[1]]  # Caméra 2
cam3 = cameras_ouvertes[camera_index[2]]  # Caméra 3

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
	cv2.imshow('Camera 1', frame1)
	cv2.imshow('Camera 2', frame2)
	cv2.imshow('Camera 3', frame3)

	# Redimensionner les flux vidéo pour les afficher côte à côte 
	# (optionnel mais efficace pour mieux visualiser)
	# frame_rz1 = cv2.resize(frame1, (WIDTH, HEIGHT))
	# frame_rz2 = cv2.resize(frame2, (WIDTH, HEIGHT))
	# frame_rz3 = cv2.resize(frame3, (WIDTH, HEIGHT))
	# # Combiner horizontalement les trois flux vidéo
	# combined_frame = np.hstack((frame_rz1, frame_rz2, frame_rz3))

	# # Afficher le flux vidéo combiné dans une même fenêtre
	# cv2.imshow('Vue d\'ensemble', combined_frame)

	key = cv2.waitKey(1) & 0xFF

	# Appuyer sur ESPACE pour enregistrer la capture
	if key == 32:  # 32 est le code ASCII pour ESPACE
		ts = datetime.now().strftime('%d%m%y_%H%M%S')
		ok1 = cv2.imwrite(str(DATASET_DIR / f'cam1_{ts}.jpg'), frame1)
		ok2 = cv2.imwrite(str(DATASET_DIR / f'cam2_{ts}.jpg'), frame2)
		ok3 = cv2.imwrite(str(DATASET_DIR / f'cam3_{ts}.jpg'), frame3)

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