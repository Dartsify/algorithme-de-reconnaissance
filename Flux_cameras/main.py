import time
from pathlib import Path
import cv2

from prendre_photo_3vues import initialiser_cameras, capturer_synchronise, liberer_cameras
from inference_yolo import charger_modele_yolo, analyser_batch_yolo
from comparaison_images import isoler_nouvelles_flechettes, elire_meilleure_camera
from homographie import charger_homographies, projeter_point_sur_cible
from image_de_reference import initialiser_references, verifier_mouvement, verifier_mouvement_3vues, actualiser_references
from envoi_backend import envoyer_point_au_backend  

# CONFIGURATION DES CHEMINS 
DOSSIER_ACTUEL = Path(__file__).resolve().parent
RACINE_PROJET = DOSSIER_ACTUEL.parent
CHEMIN_MODELE = RACINE_PROJET / "models" / "best_1.pt"  # À ajuster selon le modèle exporté (ONNX ou PyTorch)
CHEMIN_CALIBRATION = RACINE_PROJET / "Homographie" / "matrice_homographie"

# CONSTANTES DE TEMPS 
CAPTURE_DELAY_SECONDS = 0.5    # Temps de stabilisation après détection de mouvement
CAPTURE_COOLDOWN_SECONDS = 1.0 # Temps mort après une analyse

MODE_DEBUG = True # À passer sur False en production 
DOSSIER_DEBUG = DOSSIER_ACTUEL / "debug_images"
DOSSIER_DEBUG.mkdir(exist_ok=True)


def attendre_retrait_flechettes(cameras: dict, session_yolo, lancers_enregistres: int):
    """Met le jeu en pause tant qu'il y a encore des fléchettes sur la cible."""
    print(f"\n[PAUSE] Fin du tour ({lancers_enregistres}/3). Allez retirer vos fléchettes.")
    
    while True:
        time.sleep(1.0)
        frames = capturer_synchronise(cameras)
        resultats_yolo = analyser_batch_yolo(session_yolo, frames, seuil_confiance=0.4)
        max_flechettes_vues = max((len(flechettes) for flechettes in resultats_yolo.values()), default=0)
        
        if max_flechettes_vues == 0:
            print("[INFO] La cible est vide. La partie reprend dans 2 secondes...")
            time.sleep(2.0)
            print("\n[INFO] Go ! C'est reparti !")
            print("--- En attente du lancer 1 ---")
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
    print("--- En attente du lancer 1 ---")

    try:
        while True:
            maintenant = time.monotonic()
            frames_actuelles = capturer_synchronise(cameras)
            
            # ÉTAPE A : DÉTECTION 
            mouvement = verifier_mouvement_3vues(references_mouvement, frames_actuelles)
            
            if mouvement and not en_attente_de_stabilisation and (maintenant - derniere_capture > CAPTURE_COOLDOWN_SECONDS):
                print("[CAPTEUR] Fléchette détectée ! Capture prévue dans 1/2 seconde.")
                en_attente_de_stabilisation = True
                mouvement_detecte_a = maintenant

            # ÉTAPE B : ANALYSE 
            if en_attente_de_stabilisation and (maintenant - mouvement_detecte_a >= CAPTURE_DELAY_SECONDS):
                t_analyse = time.monotonic()
                print("--- En cours d'analyse ---")
                
                # On reprend une capture fraîche une fois la fléchette stabilisée
                frames_stabilisees = capturer_synchronise(cameras)
                resultats_yolo = analyser_batch_yolo(session_yolo, frames_stabilisees)
                max_flechettes_vues = max((len(flechettes) for flechettes in resultats_yolo.values()), default=0)
                
                # 1. Gestion du retrait anticipé (ex: le joueur retire après 1 ou 2 lancers)
                if lancers_enregistres > 0 and max_flechettes_vues < lancers_enregistres:
                    print(f"\n[INFO] Retrait anticipé détecté ({lancers_enregistres} -> {max_flechettes_vues}).")
                    attendre_retrait_flechettes(cameras, session_yolo, lancers_enregistres)
                    etat_anciennes_pointes = {1: [], 2: [], 3: []}
                    lancers_enregistres = 0
                    references_mouvement = actualiser_references(frames_stabilisees)
                    en_attente_de_stabilisation = False
                    derniere_capture = time.monotonic()
                    continue
                
                # 2. Isolation et Élection
                nouvelles_par_camera = {}
                for cam_id, detections in resultats_yolo.items():
                    nouvelles_par_camera[cam_id] = isoler_nouvelles_flechettes(etat_anciennes_pointes[cam_id], detections)
                
                cam_elue, flechette_gagnante = elire_meilleure_camera(nouvelles_par_camera)
                
                if cam_elue and flechette_gagnante:
                    lancers_enregistres += 1
                    
                    # Homographie et Envoi
                    matrice = matrices_homographie[cam_elue]
                    x_cible, y_cible = projeter_point_sur_cible(flechette_gagnante.x_tip, flechette_gagnante.y_tip, matrice)
                    
                    print(f"[IA] Caméra élue : {cam_elue} | Confiance IA : {flechette_gagnante.confiance_globale:.2f}")
                    print(f"[RÉSULTAT] Coordonnées corrigées : X={x_cible:.1f}, Y={y_cible:.1f}")
                    print(f"[CHRONO] Analyse terminée en {time.monotonic() - t_analyse:.3f} secondes")
                    
                    envoyer_point_au_backend(x_cible, y_cible, cam_elue)
                    
                    if MODE_DEBUG:
                        img_debug = frames_stabilisees[cam_elue].copy()
                        pt_pointe = (int(flechette_gagnante.x_tip), int(flechette_gagnante.y_tip))
                        pt_empenage = (int(flechette_gagnante.x_flight), int(flechette_gagnante.y_flight))
                        
                        cv2.circle(img_debug, pt_pointe, 5, (0, 0, 255), -1)
                        cv2.line(img_debug, pt_pointe, pt_empenage, (0, 255, 0), 3)
                        
                        nom_fichier = DOSSIER_DEBUG / f"lancer_{lancers_enregistres}_{int(time.time())}_cam{cam_elue}.jpg"
                        cv2.imwrite(str(nom_fichier), img_debug)
                        print(f"[DEBUG] Image de contrôle sauvegardée : {nom_fichier.name}")
                    
                    # Mise en mémoire de la fléchette validée
                    for c_id, nv_liste in nouvelles_par_camera.items():
                        if nv_liste:
                            etat_anciennes_pointes[c_id].append((nv_liste[0].x_tip, nv_liste[0].y_tip))
                            
                    # Contrôle de fin de tour (3 fléchettes)
                    if lancers_enregistres >= 3:
                        attendre_retrait_flechettes(cameras, session_yolo, lancers_enregistres)
                        etat_anciennes_pointes = {1: [], 2: [], 3: []}
                        lancers_enregistres = 0
                    else:
                        print(f"\n--- En attente du lancer {lancers_enregistres + 1} ---")
                        
                else:
                    print("[IA] Faux déclenchement (mouvement sans nouvelle fléchette valide).")
                    print(f"\n--- En attente du lancer {lancers_enregistres + 1} ---")

                # CRUCIAL : On met à jour la référence de mouvement pour intégrer la fléchette plantée
                references_mouvement = actualiser_references(frames_stabilisees)                
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