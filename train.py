# Code pour entrainer notre modèle avec le dataset de DeepDarts et le nôtre. Les deux fichiers de configuration se trouvent dans le dossier "configs" et sont nommés "deepdarts_d1.yaml" et "strady.yaml".

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Fallback si une commande pour mon mac n'est pas supportée par le backend MPS (Metal Performance Shaders). Cela permet d'éviter les erreurs de type "RuntimeError: MPS backend is not supported for this operation" lors de l'entrainement du modèle sur mon Mac.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True" # Empêche les erreurs de type "RuntimeError: Found duplicate OpenMP library" lors de l'entrainement du modèle sur mon Mac.

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

import wandb

# Chargement du fichier de configuration pour l'entrainement du modèle
with open("configs/deepdarts_d1.yaml", "r") as file:
    cfg = yaml.safe_load(file)

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Connexion à wandb (wandb trouve lui même la clé d'API depuis le fichier .env) et initialisation du projet
if wandb.login(): # On s'assure d'être bien connecté
    wandb.init(project="strady", name=cfg["train"]["name"], config=cfg) # On lui précise le projet, le nom du run et on lui passe tout le fichier de configuration pour qu'il puisse le sauvegarder dans wandb

# Chargement du modèle YOLO
model = YOLO(cfg["model"]["weights_path"]) # Pointe vers "yolo26n.pt" - YOLO26 Tiny

# Lancement de l'entrainement du modèle avec les paramètres définis dans le fichier de configuration
results = model.train(
    data=cfg["data"]["path"], # Pointe vers ex. "datasets/deepdarts_d1_yolo/data.yaml"
    imgsz=cfg["model"]["input_size"], # 800 (800x800)
    epochs=cfg["train"]["epochs"],
    batch=cfg["train"]["batch_size"],
    # lr0=cfg["train"]["lr"], # -> Je le dégage car je laisse le modèle gérer lui même le learning rate (il est déjà optimisé pour YOLO26). Si ça va pas je le remettrai avec AdamW comme optimiseur.
    seed=cfg["train"]["seed"],

    # Data aug
    fliplr=cfg["aug"]["flip_lr_prob"],
    flipud=cfg["aug"]["flip_ud_prob"],
    degrees=cfg["aug"]["rot_step"],
    translate=cfg["aug"]["jitter_max"],
    mosaic=cfg["aug"]["mosaic"],
    perspective=cfg["aug"]["perspective"], # Ajout de la perspective (que DeepDarts a créé lui même)

    name=cfg["train"]["name"], # Nom du run en local, par défaut "train" incrémenté à chaque run

    # Optimisation pour entrainement sur mon mac
    device="mps",
    workers=8,
    deterministic=False
)

# Clôture propre du suivi cloud
if wandb.run is not None:
    wandb.finish()