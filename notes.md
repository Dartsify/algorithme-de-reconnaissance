# Notes pour l'entrainement du modèle

`25-sept-2026`

Je vois dans un guide de bonnes pratiques d'ultralytics ([lien](https://docs.ultralytics.com/guides/model-training-tips)) qu'un bon début est de paramétrer 300 epochs et de voir si ça overfit. Je vais peut-être faire pareil.

Par défaut apparement YOLO26 utilise les poids de COCO. Donc pas besoin d'utiliser les poids d'ImageNet comme c'était courant à l'époque.

Je viens de découvrir qu'il faut plutôt, pour le format YOLO, un fichier `.txt` d'annotation par image. Donc je vais faire un script qui traduit le `labels.pkl` de DeepDarts dans le bon format.

Je me demandais aussi pq on doit pas utiliser de DataLoaders avec pytoch etc comme aux TPs, ici tout est fait tout seul. Pas besoin de s'occuper de ça.

`26-sept-2026`

Pendant l'entrainement sur deepdarts_d1, je vois que toutes les métriques sont bonnes :

- Précision (0.977) ;
- Rappel / Recall (0.962) ;
- mAP50 (0.984) ;
- mAP50-95 (0.648).

Cependant on voit que mAP50-95 est pas top : ça signifie que le modèle est pas hyper précis sur le tracé de la bbox autour de l'impact. Ça voudrait donc dire que l'estimation du score sera pas méga précise (il risque de décaller un peu trop la bbox par rapport à ce qu'il devrait faire). Apparement un YOLO Pose résoudrait ce problème car il est plus fait pour les points, MAIS il est plus lourd et donc l'inférence sera possiblement plus lente sur raspberry.

Donc je propose :

- On continue l'entrainement et le transfer learning de ce modèle Tiny sur notre dataset et voir ce que ça donne (niveau temps, précision etc surtout sur Raspberry) ;
- Lancer en simultané un second entrainement d'un Pose et faire aussi ce transfer learning dessus et comparer (le temmps etc) ;
- On peut aussi tenter de donner plus d'importance au bon placement des bbox via la pondération de la fonction de perte (lors du transfer, décider de opti plutôt la box_loss que la cls_loss).
