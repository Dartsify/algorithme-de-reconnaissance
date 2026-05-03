from __future__ import annotations

import os
# Désactive la recherche de transformations matérielles de MSMF
# Cela fait passer l'ouverture de 15 secondes à 0.5 seconde par caméra
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import pathlib
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
import keyboard
import concurrent.futures # pour le traitement en parallèle des caméras

import cv2
import numpy as np
import requests
import torch
from fastai.vision.all import PILImage, load_learner, Pipeline
from fastai.callback.wandb import WandbCallback
from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback

# -----------------------------------------------------------------------------
# Configuration générale
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "saved_images"
HOMOGRAPHY_FILE = PROJECT_ROOT / "matrice_homographie"
MODEL_PATH = PROJECT_ROOT / "dartsify_ai_final_radius15.pkl"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres de détection du mouvement
MOTION_PIXEL_THRESHOLD = 25
MOTION_AREA_THRESHOLD = 4000
CAPTURE_DELAY_SECONDS = 1.0
CAPTURE_COOLDOWN_SECONDS = 1.0

# Paramètres de post-traitement des masques
MASK_CLASS_INDEX = 1 # Classe Point
# rayon 15 : 600
# rayon 10 : 300
# rayon 5 : 70
MIN_DART_CONTOUR_AREA = 600 # Seuil d'aire pour filtrer les contours de fléchettes valides avec les fausses détections
MASK_MORPH_KERNEL_SIZE = 3 # Nettoyage des masques

# Paramètres backend
API_URL = os.getenv("DARTS_API_URL", "http://127.0.0.1:8000/throws/")
API_KEY = os.getenv("DARTS_API_KEY", "super_secret_key_for_raspberry_api_12345")
TARGET_ID = os.getenv("DARTS_TARGET_ID", "000001")

HEADERS = {"X-API-Key": API_KEY}


# FastAI peut charger un modèle exporté sous Linux sur Windows si l'on remplace
# PosixPath par WindowsPath avant l'appel à load_learner.
if pathlib.PosixPath is not pathlib.WindowsPath:
	pathlib.PosixPath = pathlib.WindowsPath


@dataclass
class CameraState:
	"""État mémorisé pour chaque caméra entre deux lancers."""

	previous_mask: np.ndarray | None = None
	previous_count: int = 0


@dataclass
class CameraDetection:
	"""Résultat brut d'une caméra pour un lancer donné."""

	camera_id: int
	mask: np.ndarray
	dart_count: int


def charger_matrice_par_defaut() -> dict[int, np.ndarray]:
	"""Retourne les matrices d'homographie connues pour les 3 caméras."""

	return {
		1: np.array(
			[
				[-1.71909584e00, 2.58377826e00, 1.72569816e03],
				[7.63326491e-02, -1.72803527e00, 1.10946414e03],
				[1.64742308e-04, 4.29686646e-03, 1.00000000e00],
			],
			dtype=np.float32,
		),
		2: np.array(
			[
				[1.06418902e00, 5.22050293e00, -8.90854791e02],
				[-1.57792881e00, 2.80472826e00, 9.64442615e02],
				[1.71439037e-04, 3.84979197e-03, 1.00000000e00],
			],
			dtype=np.float32,
		),
		3: np.array(
			[
				[6.03144827e-01, -1.42398095e-01, 6.41179807e02],
				[1.11326188e00, 2.21438999e00, -8.19137245e02],
				[-1.88153205e-04, 3.04541146e-03, 1.00000000e00],
			],
			dtype=np.float32,
		),
	}


def charger_homographies(path: Path) -> dict[int, np.ndarray]:
	"""Charge les homographies depuis le fichier de calibration si possible.

	Si le fichier n'est pas lisible, on retombe sur les valeurs codées en dur.
	"""

	default_matrices = charger_matrice_par_defaut()
	if not path.exists():
		print(f"[WARN] Fichier d'homographie introuvable : {path}")
		return default_matrices

	try:
		content = path.read_text(encoding="utf-8")
		float_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
		matrices: dict[int, np.ndarray] = {}

		for camera_match in re.finditer(r"CAMERA\s+(\d+)\s*(\[\[.*?\]\])", content, re.S):
			camera_id = int(camera_match.group(1))
			matrix_text = camera_match.group(2)
			numbers = [float(value) for value in re.findall(float_pattern, matrix_text)]
			if len(numbers) != 9:
				raise ValueError(
					f"La matrice d'homographie de la caméra {camera_id} ne contient pas 9 valeurs."
				)
			matrices[camera_id] = np.array(numbers, dtype=np.float32).reshape(3, 3)

		if len(matrices) != 3:
			raise ValueError("Toutes les matrices d'homographie n'ont pas pu être extraites.")

		return matrices
	except Exception as exc:
		print(f"[WARN] Impossible de charger les homographies depuis le fichier : {exc}")
		return default_matrices


def preparer_image_pour_difference(frame: np.ndarray) -> np.ndarray:
	"""Convertit une image en niveaux de gris puis la lisse pour réduire le bruit."""

	gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gris = cv2.GaussianBlur(gris, (21, 21), 0)
	return gris


def calculer_score_mouvement(image_precedente: np.ndarray, image_actuelle: np.ndarray) -> float:
	"""Retourne un score de mouvement basé sur la différence entre deux frames."""

	difference = cv2.absdiff(image_precedente, image_actuelle)
	_, difference_binaire = cv2.threshold(
		difference,
		MOTION_PIXEL_THRESHOLD,
		255,
		cv2.THRESH_BINARY,
	)
	kernel = np.ones((3, 3), dtype=np.uint8)
	difference_binaire = cv2.dilate(difference_binaire, kernel, iterations=2)
	contours, _ = cv2.findContours(difference_binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return float(sum(cv2.contourArea(contour) for contour in contours))


def charger_modele():
	"""Charge le modèle de segmentation entraîné."""
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
	
	learn = load_learner(MODEL_PATH)

	# NETTOYAGE CRITIQUE : On supprime les callbacks d'entraînement pour l'inférence
	if WandbCallback in [type(cb) for cb in learn.cbs]:
		learn.remove_cb(WandbCallback)

	if SaveModelCallback in [type(cb) for cb in learn.cbs]:
		learn.remove_cb(SaveModelCallback)

	if EarlyStoppingCallback in [type(cb) for cb in learn.cbs]:
		learn.remove_cb(EarlyStoppingCallback)

	# ON DÉSACTIVE LE CENTERCROP FANTÔME DE FASTAI
	# 1. On filtre la liste des transformations pour retirer les "Crop"
	transformations_sans_crop = [tfm for tfm in learn.dls.after_item.fs if 'Crop' not in type(tfm).__name__]

	# 2. On recrée le Pipeline avec les transformations restantes (ex: ToTensor)
	learn.dls.after_item = Pipeline(transformations_sans_crop)

	return learn


# def predire_masque_binaire(learner: any, frame_bgr: np.ndarray) -> np.ndarray:
# 	"""Retourne un masque binaire pour la classe correspondant à la pointe."""

# 	frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
# 	image = PILImage.create(frame_rgb)

# 	after_item = learner.dls.after_item
# 	after_batch = learner.dls.after_batch
# 	x = after_item(image)
# 	x = after_batch(x[None])

# 	learner.model.eval()
# 	with torch.no_grad():
# 		prediction = learner.model(x.to(learner.dls.device)).argmax(dim=1)[0]

# 	mask = (prediction.cpu().numpy() == MASK_CLASS_INDEX).astype(np.uint8) * 255
# 	# mask = (prediction.numpy() == MASK_CLASS_INDEX).astype(np.uint8) * 255

# 	# mask = cv2.convertScaleAbs(mask)
# 	# Nettoyage léger pour supprimer les petits pixels parasites.
# 	kernel = np.ones((MASK_MORPH_KERNEL_SIZE, MASK_MORPH_KERNEL_SIZE), np.uint8)
# 	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
# 	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
# 	return mask


def compter_flechettes_dans_masque(mask: np.ndarray) -> int:
	"""Compte les contours valides du masque, assimilés aux fléchettes détectées."""

	# cv2.RETR_TREE : tous les contours, y compris les trous internes
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #uniquement les contours externes
	contours_valides = [contour for contour in contours if cv2.contourArea(contour) >= MIN_DART_CONTOUR_AREA]
	return len(contours_valides)


def choisir_camera_detection(detections: list[CameraDetection]) -> CameraDetection:
	"""Choisit la caméra à traiter selon le nombre de fléchettes détectées.

	La caméra avec le score le plus élevé est privilégiée. En cas d'égalité,
	on sélectionne aléatoirement parmi les caméras ex aequo.
	"""

	if not detections:
		raise ValueError("Aucune détection fournie.")

	meilleur_score = max(detection.dart_count for detection in detections)
	choix = [detection for detection in detections if detection.dart_count == meilleur_score]
	return random.choice(choix)


def isoler_nouvelle_fleche(mask_actuel: np.ndarray, masque_precedent: np.ndarray | None) -> np.ndarray:
	"""Garde uniquement les pixels présents dans le masque actuel et absents du précédent."""

	if masque_precedent is None:
		return mask_actuel.copy()

	masque_inverse = cv2.bitwise_not(masque_precedent)
	masque_difference = cv2.bitwise_and(mask_actuel, masque_inverse)

	kernel = np.ones((MASK_MORPH_KERNEL_SIZE, MASK_MORPH_KERNEL_SIZE), np.uint8)
	masque_difference = cv2.morphologyEx(masque_difference, cv2.MORPH_OPEN, kernel, iterations=1)
	masque_difference = cv2.morphologyEx(masque_difference, cv2.MORPH_CLOSE, kernel, iterations=1)
	return masque_difference


def extraire_point_cible(mask: np.ndarray) -> tuple[int, int] | None:
	"""Récupère la position (x, y) du centre du plus grand contour du masque."""

	# cv2.RETR_TREE tous les contours, y compris les trous internes
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #uniquement les contours externes
	contours = [contour for contour in contours if cv2.contourArea(contour) >= MIN_DART_CONTOUR_AREA]
	if not contours:
		return None

	contour_principal = max(contours, key=cv2.contourArea)
	moments = cv2.moments(contour_principal)
	if moments["m00"] == 0:
		x, y, largeur, hauteur = cv2.boundingRect(contour_principal)
		return x + largeur // 2, y + hauteur // 2

	cx = int(moments["m10"] / moments["m00"])
	cy = int(moments["m01"] / moments["m00"])
	return cx, cy


def appliquer_homographie(point: tuple[int, int], matrice_homographie: np.ndarray) -> tuple[float, float]:
	"""Projette un point de l'image inclinée vers l'image de référence."""

	point_numpy = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
	point_transforme = cv2.perspectiveTransform(point_numpy, matrice_homographie)
	return float(point_transforme[0, 0, 0]), float(point_transforme[0, 0, 1])


def envoyer_point_au_backend(x_impact: float, y_impact: float, camera_id: int) -> None:
	"""Envoie la position corrigée au backend."""

	payload = {
		"target_id": TARGET_ID,
		"x_position": float(x_impact),
		"y_position": float(y_impact),
		"camera_id": camera_id,
	}

	try:
		response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=15)
		if response.status_code == 200:
			data = response.json()
			print(f"[INFO] Fléchette enregistrée par le serveur (multiplicateur x{data.get('multiplier', '?')}).")
		else:
			print(f"[ERREUR] Erreur API : {response.status_code} - {response.text}")
	except requests.exceptions.RequestException as exc:
		print(f"[ERREUR] Erreur de connexion au serveur : {exc}")


# def analyser_lancer(
# 	learner: any,
# 	homographies: dict[int, np.ndarray],
# 	frames: dict[int, np.ndarray],
# 	states: dict[int, CameraState],
# ) -> None:
# 	"""Analyse un lancer complet à partir des 3 images capturées simultanément."""
# 	# L'objectif plus tard sera de capturer les images en parallèle, mais pour l'instant on les traite séquentiellement.

# 	detections: list[CameraDetection] = []
# 	for camera_id, frame in frames.items():
# 		mask = predire_masque_binaire(learner, frame)
# 		count = compter_flechettes_dans_masque(mask)
# 		detections.append(CameraDetection(camera_id=camera_id, mask=mask, dart_count=count))
# 		print(f"[INFO] Caméra {camera_id} : {count} fléchette(s) détectée(s).")

# 	camera_choisie = choisir_camera_detection(detections)
# 	etat_camera = states[camera_choisie.camera_id]
# 	masque_nouveau = isoler_nouvelle_fleche(camera_choisie.mask, etat_camera.previous_mask)
# 	point_camera = extraire_point_cible(masque_nouveau)

# 	if point_camera is None:
# 		print(
# 			f"[ERREUR] Impossible d'extraire la position de la nouvelle fléchette "
# 			f"sur la caméra {camera_choisie.camera_id}."
# 		)
# 	else:
# 		point_corrige = appliquer_homographie(point_camera, homographies[camera_choisie.camera_id])
# 		print(f"[INFO] Point détecté par la caméra {camera_choisie.camera_id} : {point_camera}")
# 		print(f"[INFO] Point corrigé par homographie : ({point_corrige[0]:.2f}, {point_corrige[1]:.2f})")

# 		# print(f"[INFO] Envoi en cours de la position corrigée au serveur backend")
# 		# envoyer_point_au_backend(point_corrige[0], point_corrige[1], camera_choisie.camera_id)

# 	# On met à jour l'état de toutes les caméras pour le lancer suivant.
# 	for detection in detections:
# 		states[detection.camera_id].previous_mask = detection.mask.copy()
# 		states[detection.camera_id].previous_count = detection.dart_count

def analyser_lancer(
    learner,
    homographies: dict[int, np.ndarray],
    frames: dict[int, np.ndarray],
    states: dict[int, CameraState],
) -> None:
    """Analyse un lancer complet en envoyant les 3 images redimensionnées en MÊME TEMPS à l'IA."""

    # --- NOUVEAU : FACTEUR D'ÉCHELLE ---
    SCALE_FACTOR = 2
    NEW_WIDTH = 1280 // SCALE_FACTOR  # 640
    NEW_HEIGHT = 720 // SCALE_FACTOR  # 360

    def predire_masques_en_lot(learner, liste_frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Prend une liste de 3 images BGR, les envoie à FastAI d'un coup, et retourne 3 masques binaires."""
        liste_tenseurs = []
        for img in liste_frames_bgr:
            # --- NOUVEAU : REDIMENSIONNEMENT AVANT L'IA ---
            img_resized = cv2.resize(img, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_AREA)
            
            image_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            image_pil = PILImage.create(image_rgb)
            x = learner.dls.after_item(image_pil)
            liste_tenseurs.append(x)

        x_batch = torch.stack(liste_tenseurs)
        x_batch = learner.dls.after_batch(x_batch)

        learner.model.eval()
        with torch.no_grad():
            preds = learner.model(x_batch.to(learner.dls.device))

        masques_batch = preds.argmax(dim=1) 
        liste_masques_finaux = []
        
        # ATTENTION : Si le masque est plus petit, il faut aussi réduire la taille du kernel de nettoyage
        # et réduire temporairement MIN_DART_CONTOUR_AREA si vous l'utilisez plus bas.
        kernel_size = max(1, MASK_MORPH_KERNEL_SIZE // SCALE_FACTOR)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for i in range(len(liste_frames_bgr)):
            masque_tensor = masques_batch[i] 
            masque_numpy = (masque_tensor.cpu().numpy() == MASK_CLASS_INDEX).astype(np.uint8) * 255
            masque_numpy = cv2.morphologyEx(masque_numpy, cv2.MORPH_OPEN, kernel, iterations=1)
            masque_numpy = cv2.morphologyEx(masque_numpy, cv2.MORPH_CLOSE, kernel, iterations=1)
            liste_masques_finaux.append(masque_numpy)

        return liste_masques_finaux


    camera_ids = list(frames.keys())
    liste_images = [frames[cam_id] for cam_id in camera_ids]
    liste_masques = predire_masques_en_lot(learner, liste_images)

    detections: list[CameraDetection] = []
    
    # NOUVEAU : Adapter le seuil d'aire car l'image est 4x plus petite
    SEUIL_AIRE_REDIMENSIONNE = MIN_DART_CONTOUR_AREA // (SCALE_FACTOR ** 2)

    for i, camera_id in enumerate(camera_ids):
        mask = liste_masques[i]
        
        # On passe notre seuil adapté à la fonction de comptage (il faudra modifier votre fonction
        # compter_flechettes_dans_masque pour qu'elle accepte un paramètre de seuil optionnel)
        # count = compter_flechettes_dans_masque(mask, seuil=SEUIL_AIRE_REDIMENSIONNE)
        
        # Pour faire simple ici sans modifier votre autre fonction, faisons le comptage en ligne :
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = sum(1 for c in contours if cv2.contourArea(c) >= SEUIL_AIRE_REDIMENSIONNE)
        
        detections.append(CameraDetection(camera_id=camera_id, mask=mask, dart_count=count))
        print(f"[INFO] Caméra {camera_id} : {count} fléchette(s) détectée(s).")

    camera_choisie = choisir_camera_detection(detections)
    etat_camera = states[camera_choisie.camera_id]
    masque_nouveau = isoler_nouvelle_fleche(camera_choisie.mask, etat_camera.previous_mask)
    
    # On extrait le point sur l'image réduite
    # (Il faut aussi modifier extraire_point_cible pour utiliser SEUIL_AIRE_REDIMENSIONNE)
    contours_cibles, _ = cv2.findContours(masque_nouveau, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_valides = [c for c in contours_cibles if cv2.contourArea(c) >= SEUIL_AIRE_REDIMENSIONNE]
    
    point_camera = None
    if contours_valides:
        contour_principal = max(contours_valides, key=cv2.contourArea)
        moments = cv2.moments(contour_principal)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            point_camera = (cx, cy)

    if point_camera is None:
        print(f"[ERREUR] Impossible d'extraire la position sur la caméra {camera_choisie.camera_id}.")
    else:
        # --- NOUVEAU : REMISE À L'ÉCHELLE 1280x720 ---
        point_original = (point_camera[0] * SCALE_FACTOR, point_camera[1] * SCALE_FACTOR)
        
        point_corrige = appliquer_homographie(point_original, homographies[camera_choisie.camera_id])
        print(f"[INFO] Point détecté (échelle réduite) : {point_camera}")
        print(f"[INFO] Point recalculé (échelle 100%) : {point_original}")
        print(f"[INFO] Point corrigé par homographie : ({point_corrige[0]:.2f}, {point_corrige[1]:.2f})")

        print(f"[INFO] Envoi en cours de la position corrigée au serveur backend")
        envoyer_point_au_backend(point_corrige[0], point_corrige[1], camera_choisie.camera_id)

    # Mise à jour de l'état (les masques sauvegardés sont en 640x360, ce qui économise aussi de la RAM !)
    for detection in detections:
        states[detection.camera_id].previous_mask = detection.mask.copy()
        states[detection.camera_id].previous_count = detection.dart_count

def ouvrir_une_camera(camera_id):
	"""Tente d'ouvrir une seule caméra (Multithreadé)."""
	cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF) # on force explicitement l'api MSMF
	
	if cap.isOpened():
		# Force la résolution 720p
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		return camera_id, cap
	else:
		print(f"Échec de la caméra {camera_id}.")
		return camera_id, None

def ouvrir_cameras() -> dict[int, cv2.VideoCapture]:
	"""Ouvre les 3 caméras en parallèle"""
	cameras_ouvertes = {}
	camera_index = [3, 0, 2] # Les index des 3 caméras
	
	with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
		# Lance les 3 ouvertures exactement au même moment
		resultats = executor.map(ouvrir_une_camera, camera_index)
		
		for camera_index, cap in resultats:
			# cam 1 -> indice 3
			# cam 2 -> indice 0
			# cam 3 -> indice 2
			mapping = {3: 1, 0: 2, 2: 3} # Mapping des indices physiques vers les numéros des caméras tels que mis sur leur support
			num_camera = mapping.get(camera_index)
			if cap is not None:
				cameras_ouvertes[num_camera] = cap
			else:
				raise RuntimeError(f"[ERREUR] Impossible d'ouvrir la caméra {num_camera}.")
			
	return cameras_ouvertes


def lire_images_reference(cameras: dict[int, cv2.VideoCapture]) -> dict[int, np.ndarray]:
	"""Lit une image de référence pour chaque caméra au démarrage."""

	frames: dict[int, np.ndarray] = {}
	for camera_id, camera in cameras.items():
		success, frame = camera.read()
		if not success:
			raise RuntimeError(f"[ERREUR] Impossible d'initialiser l'image de référence de la caméra {camera_id}.")
		frames[camera_id] = frame
	return frames


def capturer_images_courantes(cameras: dict[int, cv2.VideoCapture]) -> dict[int, np.ndarray]:
	"""Lit une image sur chacune des 3 caméras."""

	frames: dict[int, np.ndarray] = {}
	for camera_id, camera in cameras.items():
		success, frame = camera.read()
		if not success:
			raise RuntimeError(f"[ERREUR] Impossible de lire le flux vidéo de la caméra {camera_id}.")
		frames[camera_id] = frame
	return frames


def main() -> None:
	"""Boucle principale du système d'auto-scoring."""

	learner = charger_modele()
	homographies = charger_homographies(HOMOGRAPHY_FILE)
	cameras = ouvrir_cameras()

	try:
		frames_reference = lire_images_reference(cameras)
		previous_gray = {
			camera_id: preparer_image_pour_difference(frame)
			for camera_id, frame in frames_reference.items()
		}
		camera_states = {camera_id: CameraState() for camera_id in cameras}

		print("La partie peut commencer !")
		derniere_capture = 0.0
		capture_en_attente = False
		instant_detection = 0.0
		
		nbLancer = 1 # compteur de lancer pour le nommage des images sauvegardées(à supprimer plus tard)

		while True:
			frames_actuelles = capturer_images_courantes(cameras)
			current_gray = {
				camera_id: preparer_image_pour_difference(frame)
				for camera_id, frame in frames_actuelles.items()
			}

			scores = {
				camera_id: calculer_score_mouvement(previous_gray[camera_id], current_gray[camera_id])
				for camera_id in cameras
			}
			score_mouvement = max(scores.values())

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
				# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
				frames_capturees = capturer_images_courantes(cameras)

				# Sauvegarde des images capturées pour DEBUG par lance par caméra
				for camera_id, frame in frames_capturees.items():
					chemin_image = IMAGE_DIR / f"cam{camera_id}_lancer{nbLancer}.jpg"
					if not cv2.imwrite(str(chemin_image), frame):
						print(f"[ERREUR] Échec de l'enregistrement de {chemin_image}.")
				nbLancer += 1

				# Analyse du lancer à partir des 3 images capturées
				# Images trop grandes 1280x720, à redimensionner plus tard pour accélérer l'inférence
				print("---Temps d'analyse : ~3 secondes---")
				analyser_lancer(learner, homographies, frames_capturees, camera_states)

				capture_en_attente = False
				derniere_capture = maintenant

				# SI OK, ON PASSE AU LANCER SUIVANT)
				print(f"\n--- En attente du lancer {nbLancer} ---")

			# Mise à jour des images de référence pour la prochaine différence de frame.
			previous_gray = current_gray

			if keyboard.is_pressed('space') or keyboard.is_pressed('esc'):
				print("Fin de la partie.")
				break
	finally:
		for camera in cameras.values():
			camera.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
