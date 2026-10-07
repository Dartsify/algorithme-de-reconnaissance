import cv2
import numpy as np
from prendre_photo_3vues import capturer_synchronise


# Convertit l'image en niveaux de gris et la floute légèrement.
# Cela supprime le "bruit" numérique du capteur qui pourrait déclencher de faux mouvements.
def preparer_image_mouvement(frame: np.ndarray) -> np.ndarray:
    
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris_flou = cv2.GaussianBlur(gris, (21, 21), 0)
    return gris_flou


# Prend une photo à vide au lancement du jeu pour servir de base à la détection de mouvement.
def initialiser_references(cameras: dict) -> dict[int, np.ndarray]:
    
    print("[INIT] Capture de la cible à vide...")
    frames_brutes = capturer_synchronise(cameras)
    
    references_preparees = {}
    for cam_id, frame in frames_brutes.items():
        references_preparees[cam_id] = preparer_image_mouvement(frame)
        
    print("[INIT] Images de référence mémorisées.")
    return references_preparees


# Compare l'image actuelle à la référence. Si la différence est assez grande,
# on considère qu'une fléchette (ou une main) est en mouvement.
def verifier_mouvement(ref_gray: np.ndarray, frame_actuelle: np.ndarray, seuil_pixel: int = 25, seuil_surface: int = 6000) -> bool:
    
    actuelle_gray = preparer_image_mouvement(frame_actuelle)
    
    # Différence absolue entre l'image vide et l'image actuelle
    difference = cv2.absdiff(ref_gray, actuelle_gray)
    
    # Binarisation : tout pixel ayant changé de plus de 'seuil_pixel' devient blanc (255)
    _, difference_binaire = cv2.threshold(difference, seuil_pixel, 255, cv2.THRESH_BINARY)
    
    # Nettoyage des petits parasites
    kernel = np.ones((3, 3), dtype=np.uint8)
    difference_binaire = cv2.dilate(difference_binaire, kernel, iterations=2)
    
    # Calcul de la surface totale en mouvement
    contours, _ = cv2.findContours(difference_binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    surface_totale = sum(cv2.contourArea(contour) for contour in contours)
    
    return surface_totale > seuil_surface