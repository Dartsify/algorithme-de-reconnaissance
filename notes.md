# Notes pour l'entrainement du modèle

Je vois dans un guide de bonnes pratiques d'ultralytics ([lien](https://docs.ultralytics.com/guides/model-training-tips)) qu'un bon début est de paramétrer 300 epochs et de voir si ça overfit. Je vais peut-être faire pareil.

Par défaut apparement YOLO26 utilise les poids de COCO. Donc pas besoin d'utiliser les poids d'ImageNet comme c'était courant à l'époque.

Je viens de découvrir qu'il faut plutôt, pour le format YOLO, un fichier `.txt` d'annotation par image. Donc je vais faire un script qui traduit le `labels.pkl` de DeepDarts dans le bon format.

Je me demandais aussi pq on doit pas utiliser de DataLoaders avec pytoch etc comme aux TPs, ici tout est fait tout seul. Pas besoin de s'occuper de ça.
