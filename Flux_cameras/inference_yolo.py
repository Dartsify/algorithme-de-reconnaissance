import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from pathlib import Path


RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_MODELS = RACINE_PROJET / "models"
CHEMIN_MODELE_DEFAUT = DOSSIER_MODELS / "darts_yolo.onnx"

DOSSIER_MODELS.mkdir(parents=True, exist_ok=True)

# --- CONSTANTES DE ROGNAGE (identiques à la préparation du dataset) ---
# L'homographie est faite sur du 1280x720 : on rogne 150 px de chaque côté (il reste 980x720),
# on étire en 800x800 comme les images d'entraînement, puis on remet les coordonnées
# sur l'image d'origine 1280x720 en fin de traitement.
CROP_TOP = 0
CROP_BOTTOM = 0
CROP_LEFT = 150
CROP_RIGHT = 150

TAILLE_YOLO_DEFAUT = 800      # utilisée seulement si l'ONNX a une taille d'entrée dynamique
NMS_IOU = 0.45                # utilisé seulement pour les exports avec NMS externe (end2end=False)


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
    confiance_flight: float


# --- FONCTIONS PRINCIPALES ---

def charger_modele_yolo(chemin_modele: Path) -> ort.InferenceSession:

    if not chemin_modele.exists():
        raise FileNotFoundError(f"Modèle YOLO introuvable à : {chemin_modele}")

    print(f"[IA] Chargement du modèle YOLO : {chemin_modele.name}...")

    # Optimisations spécifiques pour le processeur ARM du Raspberry Pi
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(str(chemin_modele), sess_options=options, providers=['CPUExecutionProvider'])
    print(f"[IA] Entrée : {session.get_inputs()[0].shape} | Sortie : {session.get_outputs()[0].shape}")
    return session


def _taille_entree(session: ort.InferenceSession) -> int:
    """Lit la taille d'entrée (carrée) directement dans le modèle ONNX."""
    taille = session.get_inputs()[0].shape[2]
    return taille if isinstance(taille, int) else TAILLE_YOLO_DEFAUT


# Prépare les images pour YOLO (rognage, étirement en carré, RGB, normalisation, batching).
# Retourne le tenseur, l'ordre des caméras et, pour chaque caméra, les ratios (x, y)
# permettant de revenir de l'espace YOLO (800x800) à la zone rognée (980x720).
def pretraiter_images(frames: dict[int, cv2.typing.MatLike], taille: int = TAILLE_YOLO_DEFAUT
                      ) -> tuple[np.ndarray, list[int], dict[int, tuple[float, float]]]:
    camera_ids = list(frames.keys())
    tenseurs = []
    ratios = {}

    for cam_id in camera_ids:
        img = frames[cam_id]
        h_orig, w_orig = img.shape[:2]

        rogne = img[CROP_TOP: h_orig - CROP_BOTTOM, CROP_LEFT: w_orig - CROP_RIGHT]
        h_crop, w_crop = rogne.shape[:2]
        ratios[cam_id] = (w_crop / taille, h_crop / taille)

        redim = cv2.resize(rogne, (taille, taille), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(redim, cv2.COLOR_BGR2RGB)
        tenseurs.append(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)))

    batch_tensor = np.ascontiguousarray(np.stack(tenseurs), dtype=np.float32)
    return batch_tensor, camera_ids, ratios


def _inferer(session: ort.InferenceSession, batch_tensor: np.ndarray) -> np.ndarray:
    """Lance l'inférence. Si le modèle a été exporté en batch=1, on traite les images une par une."""
    entree = session.get_inputs()[0]
    batch_modele = entree.shape[0]

    if isinstance(batch_modele, int) and batch_modele != len(batch_tensor):
        if batch_modele == 1:
            sorties = [session.run(None, {entree.name: batch_tensor[i:i + 1]})[0] for i in range(len(batch_tensor))]
            return np.concatenate(sorties, axis=0)
        raise ValueError(f"Le modèle attend un batch de {batch_modele}, reçu {len(batch_tensor)}. "
                         "Réexporter avec batch=3 ou dynamic=True.")

    return session.run(None, {entree.name: batch_tensor})[0]


# --- DÉCODAGE DE LA SORTIE (deux formats possibles selon l'export) ---
# Les deux décodeurs renvoient un tableau (n, 7) :
#   [conf_boite, x_tip, y_tip, conf_tip, x_flight, y_flight, conf_flight]

def _normaliser(pred: np.ndarray, debut_keypoints: int) -> np.ndarray:
    conf = pred[:, 4]
    kp = pred[:, debut_keypoints:]
    if kp.shape[1] == 6:    # (x, y, visibilité) par keypoint
        return np.column_stack((conf, kp))
    if kp.shape[1] == 4:    # (x, y) par keypoint
        return np.column_stack((conf, kp[:, 0], kp[:, 1], conf, kp[:, 2], kp[:, 3], conf))
    raise ValueError(f"Format de sortie YOLO inattendu ({pred.shape[1]} colonnes). "
                     "Attendu : 2 keypoints (tip, flight).")


def _decoder_e2e(pred: np.ndarray, seuil: float) -> np.ndarray:
    """
    Export end2end=True (défaut YOLO26) : (300, 6 + keypoints), lignes [x1, y1, x2, y2, conf, classe, kpts...].
    Le NMS est déjà fait dans le modèle : il ne reste qu'à filtrer sur la confiance.
    """
    return _normaliser(pred[pred[:, 4] > seuil], 6)


def _decoder_brut(pred: np.ndarray, seuil: float) -> np.ndarray:
    """
    Export end2end=False : (4 + 1 + keypoints, 8400), lignes [cx, cy, w, h, conf, kpts...]. NMS à faire ici.
    """
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    pred = pred[pred[:, 4] > seuil]
    if len(pred) == 0:
        return np.empty((0, 7), dtype=np.float32)

    # NMSBoxes attend [x_haut_gauche, y_haut_gauche, largeur, hauteur]
    boxes = np.column_stack((pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2, pred[:, 2], pred[:, 3]))
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), pred[:, 4].tolist(), seuil, NMS_IOU)
    if len(indices) == 0:
        return np.empty((0, 7), dtype=np.float32)

    return _normaliser(pred[np.array(indices).flatten()], 5)


def analyser_batch_yolo(session: ort.InferenceSession, frames: dict[int, cv2.typing.MatLike],
                        seuil_confiance: float = 0.5) -> dict[int, list[FlechetteDetectee]]:

    taille = _taille_entree(session)
    batch_tensor, camera_ids, ratios = pretraiter_images(frames, taille)
    predictions = _inferer(session, batch_tensor)

    # (N, 300, 12) -> export end-to-end ; (N, 11, 8400) -> export brut
    est_e2e = predictions.shape[1] > predictions.shape[2]

    resultats_finaux = {cam_id: [] for cam_id in camera_ids}

    for i, cam_id in enumerate(camera_ids):
        lignes = _decoder_e2e(predictions[i], seuil_confiance) if est_e2e \
            else _decoder_brut(predictions[i], seuil_confiance)

        ratio_x, ratio_y = ratios[cam_id]

        for conf_box, x_tip, y_tip, conf_tip, x_flight, y_flight, conf_flight in lignes.tolist():
            # On remet à l'échelle de la zone rognée, puis on ajoute l'offset du crop
            # pour retomber sur les vraies coordonnées de l'image 1280x720
            resultats_finaux[cam_id].append(FlechetteDetectee(
                confiance_globale=conf_box,
                x_tip=x_tip * ratio_x + CROP_LEFT,
                y_tip=y_tip * ratio_y + CROP_TOP,
                confiance_tip=conf_tip,
                x_flight=x_flight * ratio_x + CROP_LEFT,
                y_flight=y_flight * ratio_y + CROP_TOP,
                confiance_flight=conf_flight,
            ))

        # Meilleure détection en premier
        resultats_finaux[cam_id].sort(key=lambda f: f.confiance_globale, reverse=True)

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

    if CHEMIN_MODELE_DEFAUT.exists():
        print(f"[INFO] Vrai modèle détecté : {CHEMIN_MODELE_DEFAUT}")
        session = charger_modele_yolo(CHEMIN_MODELE_DEFAUT)
    else:
        print("[INFO] Aucun modèle ONNX trouvé dans models/. Lancement avec le simulateur...")
        fake_model = FauxModeleYOLO()

        fausses_images = {1: None, 2: None, 3: None}
        resultats = fake_model.analyser_simulation(fausses_images)

        for cam_id, flechettes in resultats.items():
            print(f"\nCaméra {cam_id} : {len(flechettes)} fléchette(s) détectée(s)")
            for f in flechettes:
                distance_pixels = ((f.x_tip - f.x_flight)**2 + (f.y_tip - f.y_flight)**2)**0.5
                print(f"  - Pointe : ({f.x_tip}, {f.y_tip})")
                print(f"  - Empennage : ({f.x_flight}, {f.y_flight})")
                print(f"  - Longueur apparente : {distance_pixels:.1f} px (Confiance: {f.confiance_globale:.2f})")