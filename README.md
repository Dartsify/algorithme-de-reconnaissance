# Algorithme de Reconnaissance <!-- omit in toc -->

Système de reconnaissance et de localisation de fléchettes du projet Dartsify, exécuté sur Raspberry Pi avec trois caméras, un modèle FastAI de segmentation, une correction d’homographie et un envoi des coordonnées vers le backend.

## Table des matières <!-- omit in toc -->

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Arborescence du projet](#arborescence-du-projet)
- [Prérequis](#prérequis)
- [Lancement](#lancement)
- [Scripts utiles](#scripts-utiles)
- [Données et calibration](#données-et-calibration)
- [Compatibilité et dépannage](#compatibilité-et-dépannage)
- [Notes sur la Raspberry Pi](#notes-sur-la-raspberry-pi)

## Vue d'ensemble

Le projet détecte automatiquement un lancer de fléchette, identifie la caméra la plus pertinente, segmente la pointe avec un modèle de vision, extrait la position de l’impact, corrige cette position avec une matrice d’homographie et transmet le résultat au backend du projet.

Le code est organisé autour de trois usages principaux :

- la capture en temps réel depuis les trois caméras ;
- l’inférence sur les images pour repérer la fléchette ;
- les outils d’annotation, de calibration et de génération des données.

## Architecture

Le flux principal implémenté dans [final/main.py](final/main.py) suit ces étapes :

1. ouverture des trois caméras en parallèle ;
2. détection d’un mouvement significatif via différence entre deux frames successives ;
3. attente d’un court délai pour laisser la fléchette se stabiliser ;
4. capture simultanée des trois images ;
5. segmentation de chaque image avec le modèle FastAI ;
6. comptage des contours valides pour choisir la caméra la plus fiable ;
7. isolation de la nouvelle fléchette par différence de masques entre deux lancers ;
8. extraction du centre de l’impact ;
9. projection de ce point par homographie ;
10. envoi des coordonnées au backend HTTP.

Le script [inference+backend.py](inference+backend.py) correspond à une version plus ancienne et plus expérimentale du même principe. Le code de production à privilégier est celui du dossier [final/](final).

## Arborescence du projet

- [final/main.py](final/main.py) : boucle principale d’auto-scoring.
- [final/main_test_rapide.py](final/main_test_rapide.py) : version de test plus simple pour valider rapidement la capture et la détection de mouvement.
- [take_photo.py](take_photo.py) : capture manuelle de photos sur les trois caméras pour constituer un jeu d’images.
- [annotation.py](annotation.py) : interface graphique d’annotation des points d’impact sur les images de calibration.
- [write_label.py](write_label.py) : création ou mise à jour du fichier de labels associé aux images.
- [find_coord.py](find_coord.py) : scripts de recherche et de validation de l’homographie.
- [detect_dart.py](detect_dart.py) : prototype de détection de mouvement et de capture automatique.
- [test_inference.py](test_inference.py) : test isolé du modèle de segmentation sur une image donnée.
- [test_homographieV1.py](test_homographieV1.py), [test_homographieV2.py](test_homographieV2.py), [test_homographie_autoV3.py](test_homographie_autoV3.py) : étapes successives de calibration homographique.
- [Dartsify/](Dartsify) : données d’entraînement et masques.
- [final/](final) : ressources utilisées par le pipeline final, dont les images sauvegardées et la matrice d’homographie.
- [image_test/](image_test) : jeux d’images de test et de calibration.

## Prérequis

Le projet cible Python 3.13, comme indiqué dans [Pipfile](Pipfile). L’environnement principal est géré avec `pipenv`.

Le projet utilise au minimum les bibliothèques suivantes :

- `opencv-python` pour la capture et le traitement d’images ;
- `numpy` pour les calculs matriciels ;
- `pandas` pour certaines manipulations de données ;
- `keyboard` pour interrompre la boucle principale depuis le clavier ;
- `requests` pour communiquer avec le backend ;
- `torch` et `fastai` pour l’inférence du modèle de segmentation ;
- `matplotlib` pour les outils d’annotation et d’analyse ;
- `imutils` dans certains scripts de test plus anciens.

Installation typique :

```bash
pipenv install
```

Si nécessaire, active ensuite l’environnement virtuel créé par Pipenv avant d’exécuter les scripts.

## Lancement

Le point d’entrée principal est [final/main.py](final/main.py).

```bash
pipenv run python final/main.py
```

Avant le lancement, il faut vérifier les éléments suivants :

- les trois caméras sont branchées et accessibles ;
- les index caméra attendus correspondent bien au matériel local ;
- le fichier [final/matrice_homographie](final/matrice_homographie) est présent ;
- le modèle [final/dartsify_ai_final_radius15.pkl](final/dartsify_ai_final_radius15.pkl) est disponible ;
- le backend HTTP est joignable à l’URL configurée par `DARTS_API_URL`.

Le script de test rapide se lance de la même manière avec [final/main_test_rapide.py](final/main_test_rapide.py) si l’on veut uniquement vérifier la partie capture et déclenchement sur mouvement.

## Scripts utiles

- [take_photo.py](take_photo.py) : ouvre les trois caméras, affiche les flux et enregistre une image sur chaque caméra quand on appuie sur `ESPACE`.
- [annotation.py](annotation.py) : ouvre une interface graphique pour ajouter, supprimer et naviguer dans les points d’annotation.
- [write_label.py](write_label.py) : ajoute automatiquement les images absentes au fichier `label.csv`.
- [test_inference.py](test_inference.py) : teste un modèle de segmentation sur une image fixe et affiche la détection obtenue.
- [test_homographieV1.py](test_homographieV1.py), [test_homographieV2.py](test_homographieV2.py), [test_homographie_autoV3.py](test_homographie_autoV3.py) : permettent de recalculer ou de valider une homographie à partir d’images de référence.
- [detect_dart.py](detect_dart.py) : version intermédiaire du pipeline de détection automatique.

## Données et calibration

Le dossier [Dartsify/](Dartsify) contient les données utilisées pour l’entraînement du modèle.

- [Dartsify/dataset/](Dartsify/dataset) : images brutes d’entraînement ou de validation.
- [Dartsify/masks/](Dartsify/masks) : masques de segmentation associés.
- [Dartsify/label.csv](Dartsify/label.csv) : fichier de labels au format spécifique, où la colonne `Position` contient du JSON brut.

Ce point est important : ce fichier ne doit pas être traité comme un CSV classique avec un parseur naïf, car la colonne `Position` peut contenir des virgules à l’intérieur du JSON. Le code d’annotation lit donc les lignes une par une et découpe seulement sur la première virgule.

La calibration homographique est gérée par les scripts de la racine et par [final/matrice_homographie](final/matrice_homographie) :

- les points de correspondance sont déterminés à partir d’images de la cible ;
- une matrice 3x3 est calculée pour chaque caméra ;
- le point détecté dans l’image est ensuite projeté sur une vue de référence.

## Compatibilité et dépannage

Le projet contient une adaptation spécifique à Windows pour les modèles FastAI exportés sous Linux : plusieurs scripts remplacent `pathlib.PosixPath` par `pathlib.WindowsPath` avant l’appel à `load_learner`. Sans cette adaptation, le chargement du modèle peut échouer sur certaines configurations Windows.

Quelques points à surveiller si le pipeline ne démarre pas correctement :

- vérifier que le modèle `.pkl` correspond bien au script qui le charge ;
- vérifier que les index caméra configurés dans le code correspondent au branchement réel ;
- vérifier que le backend attend bien la route `/throws/` et la clé API configurée ;
- vérifier que les images de calibration existent bien dans [image_test/](image_test).

## Notes sur la Raspberry Pi

### Connexion à la Raspberry

Afin de se connecter à la Raspberry, nous pouvons :

- brancher un câble Ethernet dans le port Ethernet de la Raspberry afin d’assurer une connexion directe entre la box WiFi et la Raspberry ;
- brancher un câble Ethernet afin d’assurer une connexion entre un ordinateur personnel et la Raspberry ;
- connecter la Raspberry au WiFi sans fil.

#### Câble Ethernet Raspberry - Box WiFi

Cette solution est la plus simple mais nous ne pouvons pas l’utiliser pour la présentation du projet, car nous n’aurons pas accès à un port Ethernet fournissant une connexion dans le local où nous présenterons.

#### Câble Ethernet Raspberry - Ordinateur

Cette solution nous permet de nous rendre indépendants d’une connexion filaire avec un port Ethernet délivrant la connexion réseau.
Après la configuration du partage réseau et la connexion de l’ordinateur au réseau WiFi, la Raspberry peut recevoir la connexion via l’ordinateur.

Afin de se connecter à la Raspberry, il faut déterminer son adresse IP. Pour ceci, j’ai exécuté `arp -a -i bridge100`. Cette commande interroge la table ARP en filtrant spécifiquement sur l’interface réseau appelée `bridge100`. Grâce au retour de cette commande, nous pouvons voir quelque chose comme `? (192.168.2.2) at 2c:cf:67:db:6d:c2 on bridge100 ifscope [bridge]`. Ainsi, l’adresse IP de la Raspberry s’avère être `192.168.2.2`. Sur base de cette adresse, nous pouvons nous connecter via SSH dans le terminal : `ssh raspberry@192.168.2.2`.

Cependant, cette solution présente le problème qu’un câble sera apparent entre la Raspberry et l’ordinateur faisant tourner le site web. Cela peut donner l’impression que la solution n’est pas totalement sans fil, alors qu’elle fonctionne bien en pratique dans une logique 100 % sans fil pour la partie reconnaissance.

> N.B. : il serait aussi possible, dans ce cas, de récupérer les données de la base de données via l’API sans dépendre d’une connexion sans fil.

#### WiFi

Pour être totalement sans fil, nous pouvons connecter la Raspberry au WiFi sans fil. Deux choix s’offrent à nous :

- utiliser la carte SD et le logiciel Raspberry Pi Imager : en configurant la carte SD de manière appropriée, avec le nom du réseau et le mot de passe, la Raspberry se connecte au réseau lors du démarrage ;
- utiliser l’interface graphique : en branchant d’abord un câble Ethernet entre la Raspberry et l’ordinateur personnel pour lui fournir une connexion, puis en passant par Raspberry Pi Connect, on peut accéder à l’interface graphique de la Raspberry et renseigner les paramètres WiFi.

[Documentation officielle](https://www.raspberrypi.com/documentation/services/connect.html)

![Documentation officielle](doc_raspberry.png "Commandes de la documentation officielle")

> N.B. : la connexion est coupée momentanément lors du changement de réseau, mais on peut ensuite revenir via le portail ou via SSH.

### Éteindre la Raspberry

Il ne faut pas juste débrancher le câble. Il vaut mieux exécuter `sudo shutdown -h now`. Débrancher directement peut endommager la carte SD ou provoquer une corruption du système de fichiers.
