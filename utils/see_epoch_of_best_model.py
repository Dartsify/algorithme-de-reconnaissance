import torch

# Remplacer par le bon chemin vers le fichier du modèle
path = 'runs/detect/deepdarts_d1_run/weights/best.pt'
ckpt = torch.load(path, weights_only=False)

print(f"L'époque du best.pt est : {ckpt['epoch']}")