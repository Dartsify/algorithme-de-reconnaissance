import torch
# Remplace par le bon chemin vers ton fichier
ckpt = torch.load('runs/detect/deepdarts_d1_run/weights/best.pt', weights_only=False) 
print(f"L'époque du best.pt est : {ckpt['epoch']}")
