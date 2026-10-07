import time
from pathlib import Path

from prendre_photo_3vues import initialiser_cameras, capturer_synchronise, liberer_cameras
from inference_yolo import charger_modele_yolo, analyser_batch_yolo
from comparaison_images import isoler_nouvelles_flechettes, elire_meilleure_camera
from homographie import charger_homographies, projeter_point_sur_cible
from image_de_reference import initialiser_references, verifier_mouvement
from envoi_backend import envoyer_point_au_backend  

# "J'ai lancé 3 fléchettes, je vais les rechercher, ma main passe devant" :

# Dès que le compteur lancers_enregistres atteint 3, le script sort de la boucle principale et entre dans la fonction attendre_retrait_flechettes().

# Cette fonction prend des photos en boucle et demande à YOLO : "Vois-tu encore des fléchettes ?"

# Ta main qui passe ne sera pas confondue avec un lancer, car l'IA ne cherche pas des "mouvements", elle cherche la forme d'une fléchette. Tant que tes 3 fléchettes (ou une partie) sont là, le système reste en pause. Quand la cible est vue totalement vide, le tour se réinitialise.

# "J'ai lancé 2 fléchettes seulement (car j'ai raté la cible ou j'ai fini mon tour), je vais les rechercher" :

# Le compteur lancers_enregistres est à 2. Tu t'avances vers la cible.

# Ton mouvement déclenche l'IA (Étape B).

# Ton bras passe devant. Soit YOLO voit 0 fléchette car ton corps cache tout, soit tu as déjà retiré une fléchette et il en voit 1.

# Le code arrive à l'Étape B.2 : if max_flechettes_vues < lancers_enregistres:

# Le système comprend instantanément : "Il y avait 2 fléchettes, l'IA en voit moins. C'est impossible qu'il ait fait un lancer valide, c'est donc qu'il retire ses fléchettes."

# Le système réinitialise silencieusement le compteur de lancers à 0 et se tient prêt pour le prochain tour.



#  CONFIGURATION DES CHEMINS 
DOSSIER_ACTUEL = Path(__file__).resolve().parent
RACINE_PROJET = DOSSIER_ACTUEL.parent
CHEMIN_MODELE = RACINE_PROJET / "models" / "darts_yolo.onnx"
CHEMIN_CALIBRATION = RACINE_PROJET / "Homographie" / "matrice_homographie"

#  CONSTANTES DE TEMPS 
CAPTURE_DELAY_SECONDS = 0.5    # Temps de stabilisation après détection de mouvement
CAPTURE_COOLDOWN_SECONDS = 1.0 # Temps mort après une analyse

MODE_DEBUG = True # À passer sur False en production !
DOSSIER_DEBUG = DOSSIER_ACTUEL / "debug_images"
DOSSIER_DEBUG.mkdir(exist_ok=True)


# Met le jeu en pause tant qu'il y a encore des fléchettes sur la cible.
# Utilise YOLO pour confirmer que la cible est bien vide (0 fléchette vue).
def attendre_retrait_flechettes(cameras: dict, session_yolo, lancers_enregistres: int):
    
    print(f"\n[PAUSE] Fin du tour ({lancers_enregistres}/3). Allez retirer vos fléchettes.")
    
    while True:
        time.sleep(1.0)
        frames = capturer_synchronise(cameras)
        resultats_yolo = analyser_batch_yolo(session_yolo, frames, seuil_confiance=0.4)
        max_flechettes_vues = max(len(flechettes) for flechettes in resultats_yolo.values())
        
        if max_flechettes_vues == 0:
            print("[INFO] La cible est vide. La partie reprend dans 2 secondes...")
            time.sleep(2.0)
            print("\n[INFO] Go ! C'est reparti !")
            print(" En attente du lancer 1 ")
            break


def main():
    print("=== DÉMARRAGE DU SYSTÈME DARTSIFY ===")
    
    session_yolo = charger_modele_yolo(CHEMIN_MODELE)
    matrices_homographie = charger_homographies(CHEMIN_CALIBRATION)
    
    cameras = initialiser_cameras()
    references_mouvement = initialiser_references(cameras)
    
    etat_anciennes_pointes = {1: [], 2: [], 3: []} 
    lancers_enregistres = 0
    derniere_capture = 0.0
    mouvement_detecte_a = 0.0
    en_attente_de_stabilisation = False

    print("\n[PRÊT] La partie peut commencer.")
    print(" En attente du lancer 1 ")

    try:
        while True:
            maintenant = time.monotonic()
            frames_actuelles = capturer_synchronise(cameras)
            
            #  ÉTAPE A : DÉTECTION 
            mouvement = verifier_mouvement(references_mouvement[1], frames_actuelles[1])
            
            if mouvement and not en_attente_de_stabilisation and (maintenant - derniere_capture > CAPTURE_COOLDOWN_SECONDS):
                print("[CAPTEUR] Fléchette détectée ! Capture prévue dans 1/2 seconde.")
                en_attente_de_stabilisation = True
                mouvement_detecte_a = maintenant

            #  ÉTAPE B : ANALYSE 
            if en_attente_de_stabilisation and (maintenant - mouvement_detecte_a >= CAPTURE_DELAY_SECONDS):
                t_analyse = time.monotonic()
                print(" En cours d'analyse ")
                
                resultats_yolo = analyser_batch_yolo(session_yolo, frames_actuelles)
                max_flechettes_vues = max((len(flechettes) for flechettes in resultats_yolo.values()), default=0)
                
                # Gestion du retrait anticipé (ex: le joueur retire après 2 lancers)
                if max_flechettes_vues < lancers_enregistres:
                    print(f"\n[INFO] Retrait des fléchettes détecté en cours de tour ({lancers_enregistres} -> {max_flechettes_vues}).")
                    etat_anciennes_pointes = {1: [], 2: [], 3: []}
                    lancers_enregistres = 0
                    en_attente_de_stabilisation = False
                    derniere_capture = maintenant
                    print("\n[INFO] Tour réinitialisé.")
                    print(" En attente du lancer 1 ")
                    continue
                
                # Isolation et Élection
                nouvelles_par_camera = {}
                for cam_id, detections in resultats_yolo.items():
                    nouvelles_par_camera[cam_id] = isoler_nouvelles_flechettes(etat_anciennes_pointes[cam_id], detections)
                
                cam_elue, flechette_gagnante = elire_meilleure_camera(nouvelles_par_camera)
                
                if cam_elue:
                    lancers_enregistres += 1
                    
                    # Homographie et Envoi
                    matrice = matrices_homographie[cam_elue]
                    x_cible, y_cible = projeter_point_sur_cible(flechette_gagnante.x_tip, flechette_gagnante.y_tip, matrice)
                    
                    print(f"[IA] Caméra élue : {cam_elue} | Confiance IA : {flechette_gagnante.confiance_globale:.2f}")
                    print(f"[RÉSULTAT] Coordonnées corrigées : X={x_cible:.1f}, Y={y_cible:.1f}")
                    print(f"[CHRONO] Analyse terminée en {time.monotonic() - t_analyse:.3f} secondes")
                    
                    envoyer_point_au_backend(x_cible, y_cible, cam_elue)
                    
                    
                    if MODE_DEBUG:
                        # On récupère l'image brute de la caméra gagnante
                        img_debug = frames_actuelles[cam_elue].copy()
                        
                        # On dessine la fléchette gagnante dessus
                        pt_pointe = (int(flechette_gagnante.x_tip), int(flechette_gagnante.y_tip))
                        pt_empenage = (int(flechette_gagnante.x_flight), int(flechette_gagnante.y_flight))
                        
                        import cv2
                        cv2.circle(img_debug, pt_pointe, 5, (0, 0, 255), -1) # Point rouge sur la pointe
                        cv2.line(img_debug, pt_pointe, pt_empenage, (0, 255, 0), 3) # Ligne verte vers l'empennage
                        
                        # Sauvegarde avec l'heure exacte
                        nom_fichier = DOSSIER_DEBUG / f"lancer_{lancers_enregistres}_{int(time.time())}_cam{cam_elue}.jpg"
                        cv2.imwrite(str(nom_fichier), img_debug)
                        print(f"[DEBUG] Image de contrôle sauvegardée : {nom_fichier.name}")
                    
                    # Mise en mémoire de la fléchette validée
                    for c_id, nv_liste in nouvelles_par_camera.items():
                        if nv_liste:
                            etat_anciennes_pointes[c_id].append((nv_liste[0].x_tip, nv_liste[0].y_tip))
                            
                    # Contrôle de l'état du tour
                    if lancers_enregistres >= 3:
                        attendre_retrait_flechettes(cameras, session_yolo, lancers_enregistres)
                        etat_anciennes_pointes = {1: [], 2: [], 3: []}
                        lancers_enregistres = 0
                        references_mouvement = initialiser_references(cameras)
                    else:
                        print(f"\n En attente du lancer {lancers_enregistres + 1} ")
                        
                else:
                    print("[IA] Faux déclenchement (mouvement sans fléchette valide).")
                    print(f"\n En attente du lancer {lancers_enregistres + 1} ")

                en_attente_de_stabilisation = False
                derniere_capture = time.monotonic()

    except KeyboardInterrupt:
        print("\n[INFO] Fin de la partie demandée (ESC / Ctrl+C).")
    except Exception as e:
        print(f"\n[ERREUR CRITIQUE] {e}")
    finally:
        liberer_cameras(cameras)
        print("=== EXTINCTION DU SYSTÈME ===")

if __name__ == "__main__":
    main()