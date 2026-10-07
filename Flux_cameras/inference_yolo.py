import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from pathlib import Path


#ATTENTION, le score de confiance sera basé sur tout ce que YOLO nous donne comme confiance, il faudra un max d'infos de YOLO

# Remonte d'un cran (..) pour sortir de Flux_caméras et entrer dans models/
RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_MODELS = RACINE_PROJET / "models"
CHEMIN_MODELE_DEFAUT = DOSSIER_MODELS / "darts_yolo.onnx"

# On s'assure que le dossier existe déjà
DOSSIER_MODELS.mkdir(parents=True, exist_ok=True)




# --- STRUCTURES DE DONNÉES ---

# Représente une fléchette trouvée par YOLO sur une caméra.
@dataclass
class FlechetteDetectee:
    confiance_globale: float
    # Coordonnées sur l'image d'origine (1280x720)
    x_tip: float
    y_tip: float
    confiance_tip: float
    x_flight: float
    y_flight: float
    confiance_flight: float #tip et flight dans annotation yolo




# --- FONCTIONS PRINCIPALES ---

def charger_modele_yolo(chemin_modele: Path) -> ort.InferenceSession:
    
    if not chemin_modele.exists():
        raise FileNotFoundError(f"Modèle YOLO introuvable à : {chemin_modele}")
    
    print(f"[IA] Chargement du modèle YOLO : {chemin_modele.name}...")
    # 'CPUExecutionProvider' est le plus stable sur Raspberry. 
    session = ort.InferenceSession(str(chemin_modele), providers=['CPUExecutionProvider'])
    return session



# Prépare les images pour YOLO (Redimensionnement, RGB, Normalisation, Batching).
# Retourne le tenseur, l'ordre des caméras, et la taille d'origine pour la remise à l'échelle.
def pretraiter_images(frames: dict[int, cv2.typing.MatLike], yolo_size: int = 640) -> tuple[np.ndarray, list[int], tuple[int, int]]:
    
    input_batch = []
    camera_ids = list(frames.keys())
    taille_origine = None

    for cam_id in camera_ids:
        img = frames[cam_id]
        if taille_origine is None:
            taille_origine = (img.shape[1], img.shape[0]) # (Largeur, Hauteur)

        # 1. YOLO veut généralement des images carrées (ex: 640x640)
        img_resized = cv2.resize(img, (yolo_size, yolo_size), interpolation=cv2.INTER_LINEAR)
        
        # 2. Convertir de BGR (OpenCV) à RGB (YOLO)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Normaliser les pixels entre 0.0 et 1.0 (YOLO n'utilise pas la moyenne/écart-type d'ImageNet)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # 4. Transposer pour ONNX : (Hauteur, Largeur, Canaux) -> (Canaux, Hauteur, Largeur)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        input_batch.append(img_transposed)

    # Empiler pour créer le batch de forme (3, 3, 640, 640)
    batch_tensor = np.array(input_batch, dtype=np.float32)
    return batch_tensor, camera_ids, taille_origine


# Exécute l'inférence YOLO et décode les résultats.
def analyser_batch_yolo(session: ort.InferenceSession, frames: dict[int, cv2.typing.MatLike], seuil_confiance: float = 0.5) -> dict[int, list[FlechetteDetectee]]:
    
    # 1. Préparation
    yolo_size = 640 # Vérifie la taille d'entrée de modèle ONNX (généralement 640)
    batch_tensor, camera_ids, (largeur_orig, hauteur_orig) = pretraiter_images(frames, yolo_size)
    
    # Ratios pour remettre les coordonnées de 640x640 vers 1280x720
    ratio_x = largeur_orig / yolo_size
    ratio_y = hauteur_orig / yolo_size

    # 2. Inférence ONNX
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch_tensor})
    
    # YOLOv8 renvoie généralement un tenseur de forme (Batch, Features, Anchors).
    # Ex: (3, 14, 8400) -> 3 images, 8400 prédictions par image, 14 valeurs par prédiction.
    # Les 14 valeurs sont souvent : [cx, cy, w, h, conf, kpt1_x, kpt1_y, kpt1_conf, kpt2_x, kpt2_y, kpt2_conf, ...]
    predictions = outputs[0] 
    
    resultats_finaux = {cam_id: [] for cam_id in camera_ids}

    # 3. Post-traitement (À ADAPTER SELON LA SORTIE EXACTE DU MODÈLE)
    for i, cam_id in enumerate(camera_ids):
        pred_camera = predictions[i] # Forme : (Features, Anchors)
        pred_camera = np.transpose(pred_camera) # Forme : (Anchors, Features) pour boucler facilement
        
        # Pour une vraie implémentation, il faut appliquer un NMS (Non-Maximum Suppression)
        # Ici, on simule le filtrage basique :
        for detection in pred_camera:
            confiance_box = detection[4] # Index à vérifier selon l'export YOLO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            if confiance_box > seuil_confiance:
                # Indices hypothétiques pour YOLO-Pose (tip = kpt1, flight = kpt2)
                # Remise à l'échelle immédiate
                x_tip_yolo, y_tip_yolo, conf_tip = detection[5], detection[6], detection[7]
                x_flight_yolo, y_flight_yolo, conf_flight = detection[8], detection[9], detection[10]

                flechette = FlechetteDetectee(
                    confiance_globale=float(confiance_box),
                    x_tip=float(x_tip_yolo * ratio_x),
                    y_tip=float(y_tip_yolo * ratio_y),
                    confiance_tip=float(conf_tip),
                    x_flight=float(x_flight_yolo * ratio_x),
                    y_flight=float(y_flight_yolo * ratio_y),
                    confiance_flight=float(conf_flight)
                )
                resultats_finaux[cam_id].append(flechette)
                
        # TODO: Ajouter cv2.dnn.NMSBoxes ici s'il y a des boîtes en double (fréquent avec YOLO brut)

    return resultats_finaux




# --- ZONE DE TEST ISOLÉE ---

class FauxModeleYOLO:
    """Simule les sorties d'un modèle YOLO-Pose pour tester le code sans fichier .onnx."""
    
    def analyser_simulation(self, frames: dict) -> dict[int, list[FlechetteDetectee]]:
        return {
            # Caméra 1 : fléchette vue de profil (grande distance tip-flight = ~180 px)
            1: [
                FlechetteDetectee(
                    confiance_globale=0.92,
                    x_tip=640.0, y_tip=360.0, confiance_tip=0.95,
                    x_flight=640.0, y_flight=180.0, confiance_flight=0.90
                )
            ],
            # Caméra 2 : fléchette vue plus de face (distance plus courte = ~60 px)
            2: [
                FlechetteDetectee(
                    confiance_globale=0.88,
                    x_tip=635.0, y_tip=358.0, confiance_tip=0.89,
                    x_flight=635.0, y_flight=300.0, confiance_flight=0.85
                )
            ],
            # Caméra 3 : fléchette cachée / non détectée (liste vide)
            3: []
        }


if __name__ == "__main__":
    print("--- TEST DU FLUX YOLO EN MODE SIMULATION ---")
    
    # 1. On vérifie si un vrai modèle existe, sinon on bascule sur la simulation
    if CHEMIN_MODELE_DEFAUT.exists():
        print(f"[INFO] Vrai modèle détecté : {CHEMIN_MODELE_DEFAUT}")
        session = charger_modele_yolo(CHEMIN_MODELE_DEFAUT)
        # Ici on appellerait analyser_batch_yolo(session, fausses_images)
    else:
        print("[INFO] Aucun modèle ONNX trouvé dans models/. Lancement avec le simulateur...")
        fake_model = FauxModeleYOLO()
        
        # Données de test
        fausses_images = {1: None, 2: None, 3: None}
        resultats = fake_model.analyser_simulation(fausses_images)
        
        for cam_id, flechettes in resultats.items():
            print(f"\nCaméra {cam_id} : {len(flechettes)} fléchette(s) détectée(s)")
            for f in flechettes:
                distance_pixels = ((f.x_tip - f.x_flight)**2 + (f.y_tip - f.y_flight)**2)**0.5
                print(f"  - Pointe : ({f.x_tip}, {f.y_tip})")
                print(f"  - Empennage : ({f.x_flight}, {f.y_flight})")
                print(f"  - Longueur apparente : {distance_pixels:.1f} px (Confiance: {f.confiance_globale:.2f})")