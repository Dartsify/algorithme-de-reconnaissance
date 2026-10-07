"""Script de test pour évaluer les différents modèles et l'enregistrer dans WanDB."""

import wandb
from ultralytics import YOLO
import yaml
from pathlib import Path

# Pour plus de sécurité, il faut renseigner le chemin du fichier de configuration ET le chemin des poids
# PQ ? Car chaque run d'entrainement a le même nom, mais c'est lors du début de l'entrainement que le nom est incrémenté (sur base de )

config_path = "configs/strady_deepdarts_6.yaml" # À modifier

# Chargement du fichier de configuration pour récupérer les infos de l'entrainement du modèle
with open(config_path, "r") as file:
    cfg = yaml.safe_load(file)

run_name = cfg['train']['name']
project_dir = cfg['train'].get('project', 'runs/detect/strady')

# Construction dynamique du chemin des poids
weights_path = Path("runs") / "detect" / Path(project_dir) / run_name / "weights" / "best.pt"

# Vérification de sécurité avant de lancer le modèle
if not weights_path.exists():
    raise FileNotFoundError(f"Fichier de poids introuvable : {weights_path}")

# Affichage du chemin et attente de confirmation
print(f"\n[INFO] Poids sélectionnés pour le test : {weights_path}")
input("Appuyez sur Entrée pour continuer l'évaluation ou faites Ctrl+C pour annuler...")

# Initialiser le run W&B avec le vrai nom du dossier
wandb.init(project="Strady", name=f"test_{run_name}", job_type="test", tags=["test_set"])


# Initialiser le run W&B en spécifiant qu'il s'agit d'un test
wandb.init(project="Strady", name=f"test_{run_name}", job_type="test", tags=["test_set"])

# Charger les meilleurs poids de l'entraînement choisi
model = YOLO(weights_path)

# Lancer l'évaluation explicitement sur l'ensemble de test
metrics = model.val(data=cfg['data']['path'], split="test", name="test_results")

# wandb.log(metrics.results_dict) # Forcer l'envoi des données vers wandb

# Enregistrement manuel dans wandb

# 1. Créer un dictionnaire global pour tout stocker
donnees_wandb = {}

# 2. Ajouter les vraies valeurs numériques (mAP, Précision, etc.)
donnees_wandb.update(metrics.results_dict)

# 3. Récupérer et ajouter toutes les images générées
import glob
from pathlib import Path

dossier_sortie = metrics.save_dir
images_a_logguer = glob.glob(f"{dossier_sortie}/*.png") + glob.glob(f"{dossier_sortie}/*.jpg")

for img_path in images_a_logguer:
    nom_image = Path(img_path).stem
    donnees_wandb[nom_image] = wandb.Image(img_path)

# 4. Envoyer métriques ET images en un seul appel
wandb.log(donnees_wandb)

# Clôturer la session W&B
wandb.finish()