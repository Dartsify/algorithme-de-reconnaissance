import os
import requests
import threading
from time import monotonic

# Configuration via variables d'environnement (pratique pour Docker ou systemd sur le Raspberry)
API_URL = os.getenv("DARTS_API_URL", "http://127.0.0.1:8000/throws/")
API_KEY = os.getenv("DARTS_API_KEY", "super_secret_key_for_raspberry_api_12345")
TARGET_ID = os.getenv("DARTS_TARGET_ID", "000001")

HEADERS = {"X-API-Key": API_KEY}

def envoyer_point_au_backend(x_cible: float, y_cible: float, camera_id: int) -> None:
    """
    Envoie les coordonnées de l'impact au serveur FastAPI.
    L'exécution se fait dans un thread (en tâche de fond) pour ne pas bloquer les caméras.
    """
    payload = {
        "target_id": TARGET_ID,
        "x_position": round(float(x_cible), 1),
        "y_position": round(float(y_cible), 1),
        "camera_id": camera_id,
    }

    def _tache_envoi():
        t_debut = monotonic()
        try:
            # Le timeout est crucial : si le backend plante, on ne veut pas que le Raspberry attende dans le vide
            response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=3.0)
            t_fin = monotonic()
            
            if response.status_code == 200:
                data = response.json()
                print(f"[RÉSEAU] Envoi réussi en {(t_fin - t_debut):.3f}s. Réponse : {data}")
            else:
                print(f"[RÉSEAU - ERREUR] Code {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"[RÉSEAU - ÉCHEC] Serveur injoignable : {e}")

    # Lancement de la tâche. "daemon=True" garantit que le thread s'arrêtera si on quitte le programme principal.
    threading.Thread(target=_tache_envoi, daemon=True).start()