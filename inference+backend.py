import cv2
import pathlib
import numpy as np
from datetime import datetime
from pathlib import Path
from time import monotonic
from fastai.vision.all import *
from matplotlib import pyplot as plt
import imutils

IMAGE_DIR = Path('image_test/detect_dart/')
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres de détection du mouvement
MOTION_PIXEL_THRESHOLD = 25 # Seuil de différence de pixel pour considérer qu'il y a du mouvement
MOTION_AREA_THRESHOLD = 4000 # Seuil de surface de mouvement pour déclencher la capture (ajuster selon les tests)
CAPTURE_DELAY_SECONDS = 1 # Délai entre la détection du mouvement et la capture de l'image (pour laisser la fléchette se stabiliser)
CAPTURE_COOLDOWN_SECONDS = 1.0 # Délai minimum entre deux captures pour éviter les captures multiples dues à un même mouvement

if pathlib.PosixPath is not pathlib.WindowsPath:
	pathlib.PosixPath = pathlib.WindowsPath


# Charger le modèle entraîné (chemin d'accè du fichier)
dartsify_ai = load_learner('dartsify_ai_dicejaccard.pkl')


"""
------------------------------------------------------------------
DEBUT - Communication avec Backend (à implémenter dans un module externe) :
------------------------------------------------------------------
"""
import requests

# config
API_URL = "http://127.0.0.1:8000/throws/" # Remplacer par l'IP du backend en production
API_KEY = "MaCleSecretePourLePi_Dart123!" # La clé de mon fichier .env (je te transmettrai)
TARGET_ID = "000001" # L'identifiant physique de CETTE cible dans base de données

HEADERS = {
"X-API-Key": API_KEY
}

def send_dart_to_backend(x_impact, y_impact):
	"""
	Fonction à appeler à chaque fois que la caméra détecte une fléchette.
	"""
	payload = {
		"target_id": TARGET_ID,
		"x_position": float(x_impact), 
		"y_position": float(y_impact),
		"camera_id": 1 # [1,2,3] selon la caméra qui a détecté la fléchette (pour l'instant on peut juste envoyer 1)
	}

	print(f"Envoi de l'impact en ({payload['x_position']}, {payload['y_position']}) sur la cible {TARGET_ID}...")

	try:
		reponse = requests.post(API_URL, json=payload, headers=HEADERS)
		if reponse.status_code == 200:
			data = reponse.json()
			print(f"Fléchette enregistrée par le serveur ! (Multiplicateur x{data['multiplier']})")
		else:
			print(f"Erreur API : {reponse.text}")
	except requests.exceptions.RequestException as e:
		print(f"Erreur de connexion au serveur : {e}")

"""
------------------------------------------------------------------
FIN - Communication avec Backend
------------------------------------------------------------------
"""

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


def analyser_image_capturee(path_image):
	image = PILImage.create(path_image)
	if image is None:
		print(f"Erreur: impossible de lire {path_image}.")
		return

	after_item  = dartsify_ai.dls.after_item     # resize, ToTensor, etc.
	after_batch = dartsify_ai.dls.after_batch    # Normalize, IntToFloat, etc.

	x = after_item(image)              # apply item transforms
	x = after_batch(x[None])               # add batch dim

	dartsify_ai.model.eval()
	with torch.no_grad():
		out = dartsify_ai.model(x.to(dartsify_ai.dls.device))
		out = out.argmax(dim=1)

	mask = out[0].numpy()
	# garder seulement la classe "Point"
	mask = (mask == 1).astype(np.uint8) * 255
	img_out = cv2.convertScaleAbs(img_out)

	contours_info = cv2.findContours(img_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours_info)

	if not contours:
		print(f"Aucun contour détecté dans {path_image}.")
		return
	
	image_np = np.array(image) # Convertir PILImage en tableau NumPy (format RGB)
	image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # Convertir RGB en BGR pour OpenCV
	output = image_bgr.copy()

	for contour in contours:
		moments = cv2.moments(contour)
		if moments["m00"] == 0:
			continue

		cX = int(moments["m10"] / moments["m00"]) # coordonnée x de la pointe
		cY = int(moments["m01"] / moments["m00"]) # coordonnée y de la pointe
		print("[INFO] La pointe se trouve en:", (cX, cY))


		"""
		------------------------------------------------------------------
		Communication avec Backend (à implémenter dans un module externe) :
		------------------------------------------------------------------
		"""
		# send_dart_to_backend(x_impact=cX, y_impact=cY) # normalement envoyer l'homographie


		overlay = output.copy()

		color = (0, 255, 0)
		cv2.circle(overlay, (cX, cY), 25, color, -1)

		alpha = 0.2
		cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

		# Dessiner les cercles de détection
		cv2.circle(output, (cX, cY), 3, color, -1)

	plt.figure(figsize = (12, 12))
	plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
	plt.show()

# Définir les caméras par son numéro d'index
# -- Attention qu'ils peuvent changer si les branchements sont modifiés. --
# Si c'est pas bon , utilisez le code de take_photo.py pour trouver les bons indices de caméras sur votre machine.

# INDICE 3 -> cam 1
cam1 = cv2.VideoCapture(3)
if not cam1.isOpened():
	exit("Erreur: Impossible d'ouvrir la caméra1")

# INDICE 0 -> cam 2
cam2 = cv2.VideoCapture(0)
if not cam2.isOpened():
	exit("Erreur: Impossible d'ouvrir la caméra2")

# # INDICE 2 -> cam 3
cam3 = cv2.VideoCapture(2)
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
else:
    print("La partie peut commencer !")

previous_gray1 = preparer_image_pour_difference(frame1)
previous_gray2 = preparer_image_pour_difference(frame2)
previous_gray3 = preparer_image_pour_difference(frame3)

derniere_capture = 0.0
capture_en_attente = False
instant_detection = 0.0

try:
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
		if (
			not capture_en_attente
			and score_mouvement > MOTION_AREA_THRESHOLD
			and (maintenant - derniere_capture) > CAPTURE_COOLDOWN_SECONDS
		):
			capture_en_attente = True
			instant_detection = maintenant
			print("Fléchette détectée, capture prévue dans 1 seconde.")

		if capture_en_attente and (maintenant - instant_detection) >= CAPTURE_DELAY_SECONDS:
			timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
			path_image1 = str(IMAGE_DIR / f'cam1_{timestamp}.jpg')
			path_image2 = str(IMAGE_DIR / f'cam2_{timestamp}.jpg')
			path_image3 = str(IMAGE_DIR / f'cam3_{timestamp}.jpg')
			ok1 = cv2.imwrite(path_image1, frame1)
			ok2 = cv2.imwrite(path_image2, frame2)
			ok3 = cv2.imwrite(path_image3, frame3)

			if not (ok1 and ok2 and ok3):
				print("Erreur: échec d'enregistrement automatique (cv2.imwrite a renvoyé False).")
			else:
				print(f"Images enregistrées après stabilisation dans {IMAGE_DIR}.")
				print('Caméra 1')
				analyser_image_capturee(path_image1)
				# print('Caméra 2')
				# analyser_image_capturee(path_image2)
				# print('Caméra 3')
				# analyser_image_capturee(path_image3)

			capture_en_attente = False
			derniere_capture = maintenant

		# Mettre à jour les images de référence pour la prochaine comparaison
		previous_gray1 = current_gray1
		previous_gray2 = current_gray2
		previous_gray3 = current_gray3

		key = cv2.waitKey(1) & 0xFF

		# Appuyer sur ESC pour arrêter la capture
		if key == 27:  # 27 est le code ASCII pour ESC
			break
finally:
	# Release the capture and writer objects
	cam1.release()
	cam2.release()
	cam3.release()
	cv2.destroyAllWindows()