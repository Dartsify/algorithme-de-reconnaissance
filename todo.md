# To Do

- [x] Restructurer le dossier local datasets pour avoir toutes les photos du dataset d1 dans un seul et même dossiers ;
- [x] Dégager les infos sur les points de calibrage du fichier labels.pkl -> créer un script pour le faire ;
- [x] Vérifier que le traitement des labels n'a pas foutu tout en l'air ;
- [x] Regarder rappel de comment importer les données, se connecter à wandb pour log l'entrainement, à quoi servent les checkpoints etc ;
- [x] Charger données dans code d'entrainement ;
- [x] Connexion à wan db ;
- [x] Adapter le fichier configs/deepdarts_d1.yaml pour convenir à notre projet ;
- [x] Paramétrage des checkpoints ? ;
- [x] Regarder s'il faut comme DeepDarts établir des poids par défaut depuis ImageNet -> Non c'est par défaut via COCO ;
- [x] Attaquer le code de l'entrainement ;
- [x] Script pour convertir `labels.pkl` au format YOLO ;
- [x] Lancer entrainement ;
- [x] Trouver un moyen de vérifier que ça a "fonctionné" ;
- [x] Changer tous les 'deepdarts_d1' en 'deepdarts' -> je me suis trompé deepdarts contient d1 et d2 directement dans ce dataset, donc je vais créer un autre script qui crée un vrai dossier deepdarts_d1 en ne prenant que les images du d1 ;
- [x] Si lancement d'un autre entrainement -> faire gaffe à bien envoyer un max de données vers wandb et de peut-être enregistrer plus de checkpoints pour pouvoir retourner en arrière si le `best.pt` n'est pas en fin de compte le meilleur choix ;
- [x] Se renseigner pour lancer en parallèle de la suite l'entrainement sur deepdarts_d1 d'un modèle YOLO Pose et non plus Tiny ;
- [ ] Regarder à RunPod ;
- [ ] Tester l'annotation dans Roboflow d'un projet pour Keypoints (Pose) et voir si effectivement on peut utiliser les annotations pour entrainer à la fois un YOLO Pose et un YOLO classique (après avoir dégager les valeurs non intéressantes pour le classique).
