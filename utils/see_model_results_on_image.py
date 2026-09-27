from ultralytics import YOLO

# 1. Charger les poids du modèle entrainé (générés lors de l'entrainement avec train.py)
model = YOLO("runs/detect/deepdarts_d1_run/weights/best.pt")

# 2. Définir le chemin de l'image
image_path = "datasets/deepdarts_d1_yolo/images/val/d1_02_16_2020__IMG_2955.jpg"  # Remplace par le chemin de ton image

# 3. Lancer la prédiction (en reprenant l'imgsz de 800 et l'accélération Mac)
results = model.predict(
    source=image_path,
    imgsz=800,
    conf=0.25,      # Seuil de confiance (à ajuster si besoin)
    device="mps"    # Accélération Metal
)

# 4. Exploiter et afficher les résultats
for result in results:
    # Donne les différents temps d'exécution pour chaque étape de la prédiction (prétraitement, inférence, post-traitement)
    print(f"Temps d'exécution (ms) : {result.speed}")

    # Affiche l'image avec les BBoxes tracées directement
    result.show()
    
    # Sauvegarde l'image générée avec les prédictions
    result.save(filename="prediction_sortie.jpg")
    
    # Récupérer les données brutes des BBoxes si tu dois les traiter ensuite (ex: Dartsify)
    boxes = result.boxes.xyxy.cpu().numpy()  # Coordonnées [x1, y1, x2, y2]
    confs = result.boxes.conf.cpu().numpy()  # Scores de confiance
    classes = result.boxes.cls.cpu().numpy() # Index des classes
    
    print(f"BBoxes détéctées : {len(boxes)}")
    for box, conf, cls in zip(boxes, confs, classes):
        print(f"Classe: {int(cls)} | Confiance: {conf:.2f} | Coordonnées: {box}")
