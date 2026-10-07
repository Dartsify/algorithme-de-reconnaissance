import math
from typing import Optional
from inference_yolo import FlechetteDetectee

# Seuil en pixels : si une fléchette est à moins de 20px d'une ancienne, c'est la même. 
# A CHANGER ET A TROUVE LE MEILLEUR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SEUIL_DISTANCE_PX = 20.0


# Calcule la distance en pixels entre deux points.
def calculer_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# Compare les fléchettes vues par YOLO avec celles des lancers précédents.
# Retourne uniquement les fléchettes qui sont "nouvelles".
def isoler_nouvelles_flechettes(anciennes_pointes: list[tuple[float, float]], 
                                detections_actuelles: list[FlechetteDetectee]) -> list[FlechetteDetectee]:
    
    nouvelles = []
    
    for yolo_flechette in detections_actuelles:
        est_nouvelle = True
        
        # On compare la pointe actuelle avec toutes les anciennes pointes mémorisées
        for anc_x, anc_y in anciennes_pointes:
            dist = calculer_distance(yolo_flechette.x_tip, yolo_flechette.y_tip, anc_x, anc_y)
            if dist < SEUIL_DISTANCE_PX:
                # C'est une fléchette déjà connue, on l'ignore
                est_nouvelle = False
                break
                
        if est_nouvelle:
            nouvelles.append(yolo_flechette)
            
    return nouvelles



# Choisit la caméra qui a la vue la plus dégagée sur la NOUVELLE fléchette.
# Basé sur la distance apparente entre (flight) et la pointe (tip).
def elire_meilleure_camera(nouvelles_par_camera: dict[int, list[FlechetteDetectee]]) -> tuple[Optional[int], Optional[FlechetteDetectee]]:
    
    meilleure_cam = None
    meilleure_flechette = None
    max_longueur = -1.0

    for cam_id, liste_flechettes in nouvelles_par_camera.items():
        if not liste_flechettes:
            continue
            
        # Si une caméra voit plusieurs nouvelles fléchettes (rare mais possible au lancer 1),
        # on prend la plus évidente.
        f = liste_flechettes[0] 
        
        # mon idee : plus la fléchette est longue sur l'image, 
        # plus la caméra est rasante et précise.
        longueur_apparente = calculer_distance(f.x_tip, f.y_tip, f.x_flight, f.y_flight)
        
        if longueur_apparente > max_longueur:
            max_longueur = longueur_apparente
            meilleure_cam = cam_id
            meilleure_flechette = f

    return meilleure_cam, meilleure_flechette






# --- ZONE DE TEST ISOLÉE ---
if __name__ == "__main__":
    print("--- TEST DE LA LOGIQUE DE COMPARAISON ---")
    
    # Imaginons que la caméra 1 avait déjà vu une fléchette au lancer précédent à la position (640, 360)
    anciennes_cam1 = [(640.0, 360.0)]
    
    # YOLO vient de tourner pour le Lancer 2.
    # La caméra 1 voit DEUX fléchettes : la vieille, et une nouvelle.
    detections_yolo_cam1 = [
        # La vieille fléchette (elle a légèrement bougé de 2 pixels, normal avec YOLO)
        FlechetteDetectee(0.95, 642.0, 361.0, 0.9, 642.0, 200.0, 0.9),
        # La TOUTE NOUVELLE fléchette plantée dans le 20 !
        FlechetteDetectee(0.92, 500.0, 250.0, 0.9, 500.0, 100.0, 0.9)
    ]
    
    # La caméra 2 voit aussi la nouvelle fléchette, mais sous un mauvais angle (plus courte)
    detections_yolo_cam2 = [
        FlechetteDetectee(0.85, 490.0, 240.0, 0.9, 490.0, 180.0, 0.9) # Distance flight-tip = 60px
    ]

    # 1. On isole la nouvelle fléchette pour la Caméra 1
    nouvelles_cam1 = isoler_nouvelles_flechettes(anciennes_cam1, detections_yolo_cam1)
    print(f"[FILTRE] Caméra 1 : {len(nouvelles_cam1)} nouvelle(s) fléchette(s) trouvée(s).")
    
    # 2. On isole pour la Caméra 2 (pas d'anciennes fléchettes pour l'exemple)
    nouvelles_cam2 = isoler_nouvelles_flechettes([], detections_yolo_cam2)
    
    # 3. L'élection
    candidats = {
        1: nouvelles_cam1,
        2: nouvelles_cam2,
        3: [] # La caméra 3 n'a rien vu
    }
    
    cam_elue, flechette_elue = elire_meilleure_camera(candidats)
    
    if cam_elue:
        longueur = calculer_distance(flechette_elue.x_tip, flechette_elue.y_tip, flechette_elue.x_flight, flechette_elue.y_flight)
        print(f"[VICTOIRE] La Caméra {cam_elue} remporte l'élection avec une longueur de {longueur:.1f} pixels !")
        print(f"[ACTION] Point à envoyer à l'homographie : ({flechette_elue.x_tip}, {flechette_elue.y_tip})")