# Code pour entrainer notre modèle avec le dataset de DeepDarts et le nôtre. Les deux fichiers de configuration se trouvent dans le dossier "configs" et sont nommés "deepdarts_d1.yaml" et "strady.yaml".

from ultralytics import YOLO

model = YOLO("yolo26n.pt") # YOLOv26 Nano