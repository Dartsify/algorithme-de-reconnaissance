import cv2
import concurrent.futures
from time import monotonic

# Configuration des résolutions (doit correspondre à ce que les caméras supportent nativement)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# Tente d'ouvrir une seule caméra et configure ses paramètres de base.
def ouvrir_une_camera(camera_index: int, num_logique: int) -> tuple[int, cv2.VideoCapture | None]:
    
    # Pour un Raspberry Pi (Linux), on utilise par défaut le backend V4L2.
    # Si test sur Windows -> cv2.CAP_MSMF si besoin.
    cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        # Forcer la résolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # Désactiver l'autofocus et l'adaptation automatique de l'exposition.
        # CRUCIAL : Si l'exposition change quand le joueur retire ses fléchettes, 
        # la détection de mouvement (qui utilise absdiff) va s'affoler.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        print(f"[HW] Caméra matérielle {camera_index} ouverte et assignée à l'ID logique {num_logique}.")
        return num_logique, cap
    else:
        print(f"[ERREUR] Échec de l'ouverture de la caméra matérielle {camera_index}.")
        return num_logique, None


# Ouvre les 3 caméras en parallèle (Multithreading) pour éviter d'attendre l'initialisation séquentielle 
def initialiser_cameras() -> dict[int, cv2.VideoCapture]:
    
    cameras_ouvertes = {}
    
    # Mapping exact de ancien code david : {Index_physique_USB: ID_logique_Camera}
    # Modifie les clés si les ports USB du Raspberry Pi changent.
    mapping_cameras = {3: 1, 0: 2, 2: 3}
    
    print("[HW] Démarrage de l'initialisation des caméras en parallèle...")
    t_start = monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Lancement asynchrone des 3 ouvertures
        futures = [
            executor.submit(ouvrir_une_camera, index_physique, id_logique)
            for index_physique, id_logique in mapping_cameras.items()
        ]
        
        for future in concurrent.futures.as_completed(futures):
            id_logique, cap = future.result()
            if cap is not None:
                cameras_ouvertes[id_logique] = cap
            else:
                raise RuntimeError(f"Impossible de démarrer le système : Caméra {id_logique} manquante.")

    print(f"[HW] Toutes les caméras sont prêtes en {(monotonic() - t_start):.2f} secondes.")
    return cameras_ouvertes



# Capture une image sur chaque caméra de manière quasi-simultanée.
# Utilise grab() puis retrieve() pour minimiser le délai de capture entre les vues.
def capturer_synchronise(cameras: dict[int, cv2.VideoCapture]) -> dict[int, cv2.typing.MatLike]:
    
    frames = {}
    
    # ÉTAPE 1 : Figer les capteurs. Les 3 caméras prennent la photo au même moment physique.
    for camera in cameras.values():
        camera.grab()
        
    # ÉTAPE 2 : Rapatrier les données via le bus USB (Plus lent, mais l'image est déjà figée)
    for camera_id, camera in cameras.items():
        success, frame = camera.retrieve()
        if not success:
            # En cas d'erreur de lecture USB passagère (fréquent sur Raspberry Pi), 
            # il vaut mieux lever une exception qui sera rattrapée dans la boucle principale.
            raise RuntimeError(f"Perte de flux sur la caméra {camera_id}.")
        frames[camera_id] = frame
        
    return frames


# Ferme proprement les connexions USB.
def liberer_cameras(cameras: dict[int, cv2.VideoCapture]) -> None:

    for camera in cameras.values():
        camera.release()
    print("[HW] Caméras libérées.")





from pathlib import Path

# --- Zone de test isolée ---
if __name__ == "__main__":
    import numpy as np

    # 1. Obtenir le chemin du dossier où se trouve ce script (Flux_caméras)
    DOSSIER_ACTUEL = Path(__file__).resolve().parent
    
    # 2. Construire le chemin vers image
    CHEMIN_IMAGE = DOSSIER_ACTUEL / "cam1_020526_131010.jpg"

    class FausseCamera:
        def __init__(self, image_path):
            self.frame = cv2.imread(str(image_path)) # On convertit en string pour OpenCV
            if self.frame is None:
                self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(self.frame, "Image introuvable", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        def isOpened(self): return True
        def set(self, prop, value): pass
        def grab(self): pass
        def retrieve(self): return True, self.frame.copy()
        def release(self): pass

    print("--- DÉMARRAGE DU TEST EN MODE SIMULATION (SANS CAMÉRAS) ---")
    
    # 2. On remplace le dictionnaire de caméras matérielles par nos fausses caméras
    cams_simulees = {
        1: FausseCamera(CHEMIN_IMAGE), 
        2: FausseCamera(CHEMIN_IMAGE),
        3: FausseCamera(CHEMIN_IMAGE)
    }

    try:
        t_capture = monotonic()
        # 3. On appelle ta vraie fonction avec les fausses caméras
        images = capturer_synchronise(cams_simulees)
        print(f"Capture réussie en {(monotonic() - t_capture):.4f} secondes.")
        
        for cam_id, img in images.items():
            print(f"Caméra simulée {cam_id} : Résolution {img.shape}")
            # Afficher l'image pour vérifier que ça marche sous Windows
            cv2.imshow(f"Vue Camera {cam_id}", img)
        
        print("Appuyez sur une touche pour quitter...")
        cv2.waitKey(0)
            
    except Exception as e:
        print(f"Erreur lors du test : {e}")
    finally:
        liberer_cameras(cams_simulees)
        cv2.destroyAllWindows()