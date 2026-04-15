import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from time import monotonic


IMAGE_DIR = Path('image_test/detect_dart/')
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres de détection du mouvement
MOTION_PIXEL_THRESHOLD = 25
MOTION_AREA_THRESHOLD = 4000
CAPTURE_DELAY_SECONDS = 1
CAPTURE_COOLDOWN_SECONDS = 1.0


def preparer_image_pour_difference(frame):
	"""Convertit une image en niveau de gris et la lisse pour réduire le bruit."""
	gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gris = cv2.GaussianBlur(gris, (21, 21), 0)
	return gris


def calculer_score_mouvement(image_precedente, image_actuelle):
	"""Retourne un score de mouvement basé sur la différence entre deux frames."""
	difference = cv2.absdiff(image_precedente, image_actuelle)
	_, difference_binaire = cv2.threshold(difference, MOTION_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)
	difference_binaire = cv2.dilate(difference_binaire, None, iterations=2)
	contours, _ = cv2.findContours(difference_binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return sum(cv2.contourArea(contour) for contour in contours)

# Définir les caméras par son numéro d'index

# INDICE 3 -> cam 1
cam1 = cv2.VideoCapture(3)
if not cam1.isOpened():
	exit("Erreur: Impossible d'ouvrir la caméra1")

# INDICE 1 -> cam 2
cam2 = cv2.VideoCapture(1)
if not cam2.isOpened():
	exit("Erreur: Impossible d'ouvrir la caméra2")

# # INDICE 0 -> cam 3
cam3 = cv2.VideoCapture(0)
if not cam3.isOpened():
	exit("Erreur: Impossible d'ouvrir la caméra3")

# Définir la résolution de chaque caméra (1280x720, format par défaut des caméras)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cam3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Lecture des premières images de référence pour initialiser la comparaison
success1, frame1 = cam1.read()
success2, frame2 = cam2.read()
success3, frame3 = cam3.read()

if not (success1 and success2 and success3):
	cam1.release()
	cam2.release()
	cam3.release()
	raise RuntimeError("Erreur: Impossible d'initialiser les images de référence.")

previous_gray1 = preparer_image_pour_difference(frame1)
previous_gray2 = preparer_image_pour_difference(frame2)
previous_gray3 = preparer_image_pour_difference(frame3)

derniere_capture = 0.0
capture_en_attente = False
instant_detection = 0.0

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

	# Préparer les images courantes pour la détection de différence
	current_gray1 = preparer_image_pour_difference(frame1)
	current_gray2 = preparer_image_pour_difference(frame2)
	current_gray3 = preparer_image_pour_difference(frame3)

	# Calculer un score de mouvement pour chaque caméra
	score1 = calculer_score_mouvement(previous_gray1, current_gray1)
	score2 = calculer_score_mouvement(previous_gray2, current_gray2)
	score3 = calculer_score_mouvement(previous_gray3, current_gray3)
	score_mouvement = max(score1, score2, score3)

	# Déclencher une capture automatique si un mouvement important est détecté.
	# La photo est prise 1 seconde plus tard pour laisser la fléchette se stabiliser.
	maintenant = monotonic()
	if not capture_en_attente and score_mouvement > MOTION_AREA_THRESHOLD and (maintenant - derniere_capture) > CAPTURE_COOLDOWN_SECONDS:
		capture_en_attente = True
		instant_detection = maintenant
		print("Fléchette détectée, capture prévue dans 1 seconde.")

	if capture_en_attente and (maintenant - instant_detection) >= CAPTURE_DELAY_SECONDS:
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
		ok1 = cv2.imwrite(str(IMAGE_DIR / f'cam1_{timestamp}.jpg'), frame1)
		ok2 = cv2.imwrite(str(IMAGE_DIR / f'cam2_{timestamp}.jpg'), frame2)
		ok3 = cv2.imwrite(str(IMAGE_DIR / f'cam3_{timestamp}.jpg'), frame3)

		if not (ok1 and ok2 and ok3):
			print("Erreur: échec d'enregistrement automatique (cv2.imwrite a renvoyé False).")
		else:
			print(f"Images enregistrées après stabilisation dans {IMAGE_DIR}.")

		capture_en_attente = False
		derniere_capture = maintenant

	# Afficher le flux vidéo de chaque caméra dans une fenêtre, individuellement
	# cv2.imshow('Camera 1', frame1)
	# cv2.imshow('Camera 2', frame2)
	# cv2.imshow('Camera 3', frame3)

	# Redimensionner les flux vidéo pour les afficher côte à côte (optionnel mais efficace pour mieux visualiser)
	# + Combiner horizontalement les trois flux vidéo
	frame_rz1 = cv2.resize(frame1, (500, 500))
	frame_rz2 = cv2.resize(frame2, (500, 500))
	frame_rz3 = cv2.resize(frame3, (500, 500))
	combined_frame = np.hstack((frame_rz1, frame_rz2, frame_rz3))

	# Afficher le flux vidéo combiné dans une même fenêtre
	cv2.imshow('Vue d\'ensemble', combined_frame)

	# Mettre à jour les images de référence pour la prochaine comparaison
	previous_gray1 = current_gray1
	previous_gray2 = current_gray2
	previous_gray3 = current_gray3

	key = cv2.waitKey(1) & 0xFF

	# Appuyer sur ESC pour arrêter la capture
	if key == 27:  # 27 est le code ASCII pour ESC
		break

# Release the capture and writer objects
cam1.release()
cam2.release()
cam3.release()
cv2.destroyAllWindows()