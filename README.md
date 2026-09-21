# Algorithme de Reconnaissance <!-- omit in toc -->

Système de reconnaissance et de localisation de fléchettes du projet Dartsify, avec trois caméras, segmentation d’image, correction d’homographie et envoi des coordonnées vers le backend.

Cette branche (`math-3`) contient aussi un réentraînement local du modèle de segmentation à partir du dataset présent sur l’ordinateur. Le modèle obtenu est exporté au format FastAI `.pkl` pour les tests locaux. Il n’est pas encore le modèle ONNX utilisé par les scripts de production.

## Table des matières <!-- omit in toc -->

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Arborescence du projet](#arborescence-du-projet)
- [Prérequis](#prérequis)
- [Lancement](#lancement)
- [Réentraînement local](#réentraînement-local)
  - [Objectif](#objectif)
  - [Données utilisées](#données-utilisées)
  - [Séparation des données](#séparation-des-données)
  - [Lancer le réentraînement](#lancer-le-réentraînement)
  - [Fichiers créés](#fichiers-créés)
  - [Tester l’export FastAI](#tester-lexport-fastai)
    - [Fonctionnement du testeur](#fonctionnement-du-testeur)
    - [Signification de la croix verte](#signification-de-la-croix-verte)
    - [Différence avec `main.py`](#différence-avec-mainpy)
    - [Exemples générés](#exemples-générés)
- [Scripts utiles](#scripts-utiles)
- [Nettoyage de la branche](#nettoyage-de-la-branche)
- [Données et calibration](#données-et-calibration)
- [Compatibilité et dépannage](#compatibilité-et-dépannage)
- [Notes sur la Raspberry Pi](#notes-sur-la-raspberry-pi)
  - [Connexion à la Raspberry](#connexion-à-la-raspberry)
    - [Câble Ethernet Raspberry - Box WiFi](#câble-ethernet-raspberry---box-wifi)
    - [Câble Ethernet Raspberry - Ordinateur](#câble-ethernet-raspberry---ordinateur)
    - [WiFi](#wifi)
  - [Éteindre la Raspberry](#éteindre-la-raspberry)

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
5. segmentation de chaque image avec un modèle exécuté via ONNX Runtime ;
6. comptage des contours valides pour choisir la caméra la plus fiable ;
7. isolation de la nouvelle fléchette par différence de masques entre deux lancers ;
8. extraction du centre de l’impact ;
9. projection de ce point par homographie ;
10. envoi des coordonnées au backend HTTP.

Le code de production à privilégier est celui du dossier [final/](final). Les scripts de ce dossier attendent un modèle `.onnx`. Le réentraînement local décrit plus bas produit d’abord un modèle FastAI `.pkl`, qui doit être testé puis éventuellement converti en ONNX.

## Arborescence du projet

- [final/main.py](final/main.py) : boucle principale d’auto-scoring.
- [finalRaspberry/main_onnx_avecPause.py](finalRaspberry/main_onnx_avecPause.py) : variante ONNX avec pause après trois lancers.
- [mon_fichier_train.py](mon_fichier_train.py) : réentraînement local FastAI sur `Dartsify_mur`.
- [test_modele_fastai.py](test_modele_fastai.py) : test d’un export `.pkl` sur une image fixe avec masque, contours et centroïde.
- [export_modele_onnx.py](export_modele_onnx.py) : conversion du modèle FastAI `.pkl` vers ONNX.
- [test_pipeline_onnx.py](test_pipeline_onnx.py) : test batch ONNX sans caméra ni backend.
- [take_photo.py](take_photo.py) : capture manuelle de photos sur les trois caméras pour constituer un jeu d’images.
- [annotation.py](annotation.py) : interface graphique d’annotation des points d’impact.
- [Dartsify_mur/](Dartsify_mur) : dataset local et annotations utilisées par le réentraînement.
- [final/](final) : script de production et calibration homographique.

## Prérequis

Le projet cible Python 3.13, comme indiqué dans [Pipfile](Pipfile). L’environnement principal est géré avec `pipenv`.

Le projet utilise au minimum les bibliothèques suivantes :

- `opencv-python` pour la capture et le traitement d’images ;
- `numpy` pour les calculs matriciels ;
- `pandas` pour certaines manipulations de données ;
- `keyboard` pour interrompre la boucle principale depuis le clavier ;
- `requests` pour communiquer avec le backend ;
- `torch` et `fastai` pour le réentraînement et les tests du modèle `.pkl` ;
- `onnxruntime` pour l’inférence du modèle utilisé par le pipeline final ;
- `matplotlib` pour les outils d’annotation et d’analyse ;
- `imutils` dans certains scripts de test plus anciens.

Installation typique :

```bash
pipenv install
```

Si nécessaire, active ensuite l’environnement virtuel créé par Pipenv avant d’exécuter les scripts.

## Lancement

Le point d’entrée principal de production est [final/main.py](final/main.py).

```bash
pipenv run python final/main.py
```

Avant le lancement, il faut vérifier les éléments suivants :

- les trois caméras sont branchées et accessibles ;
- les index caméra attendus correspondent bien au matériel local ;
- le fichier [final/matrice_homographie](final/matrice_homographie) est présent ;
- le modèle ONNX attendu par le script est disponible dans `final/models_onnx/dartsify_resnet34_local.onnx` ;
- le backend HTTP est joignable à l’URL configurée par `DARTS_API_URL`.

Pour tester le modèle FastAI réentraîné sur une image fixe, utiliser la section suivante.

## Réentraînement local

### Objectif

Cette branche permet de réentraîner un modèle de segmentation sur les données locales du dossier [Dartsify_mur/](Dartsify_mur), puis de vérifier visuellement ce que le modèle retourne avant toute intégration avec les caméras.

Le modèle apprend à produire un masque binaire autour des points d’impact annotés. Chaque point du fichier `label.csv` est converti en disque blanc de rayon `15` pixels dans un masque noir. Le modèle ne reçoit donc pas directement les coordonnées comme cible : il apprend une segmentation de ces petites zones.

### Données utilisées

Le dataset local contient :

- `1500` images en résolution `1280 x 720` ;
- `1300` images référencées dans `label.csv` ;
- `957` images avec au moins un point annoté ;
- `343` images avec une annotation vide, utilisées comme images sans fléchette ;
- `200` images non référencées dans le CSV, exclues de cet entraînement ;
- `2283` points annotés au total.

Le fichier [Dartsify_mur/label.csv](Dartsify_mur/label.csv) contient une colonne `Position` avec du JSON. Comme ce JSON contient des virgules, le script ne le lit pas avec un `pd.read_csv` standard : il sépare chaque ligne uniquement sur la première virgule.

### Séparation des données

La séparation est aléatoire mais reproductible avec `random_state=42` :

- entraînement : `832` images ;
- validation : `208` images ;
- test : `260` images.

Le test est conservé à part jusqu’à l’évaluation finale. Les images sont redimensionnées vers `640 x 360` pendant la préparation des DataLoaders, comme dans le pipeline ONNX historique.

### Lancer le réentraînement

Utiliser l’environnement Python contenant `torch`, `fastai`, `opencv-python`, `pandas`, `scikit-learn` et `tqdm` :

```bash
conda activate DartsifyAlgoReconnaissance
python mon_fichier_train.py
```

W&B est désactivé par défaut. Pour l’activer volontairement :

```bash
wandb login
export DARTSIFY_USE_WANDB=1
python mon_fichier_train.py
```

Sur macOS avec le backend Apple MPS, le script active `PYTORCH_ENABLE_MPS_FALLBACK=1` pour les opérations de déformation perspective non supportées par certaines versions de PyTorch. Ces opérations peuvent alors être exécutées sur CPU.

### Fichiers créés

Le script crée ou met à jour :

- `Dartsify_mur/masks/` : un masque PNG par image annotée ;
- `trained_models/dartsify_resnet34_local.pkl` : export FastAI du modèle entraîné ;
- `final/models_onnx/dartsify_resnet34_local.onnx` : conversion ONNX utilisée par `final/main.py` ;
- `models/` : éventuels checkpoints locaux créés par `SaveModelCallback` ;
- `model_test_output/` : résultats des tests FastAI et ONNX, si les testeurs sont lancés.

Les modèles `.pkl` et les sorties générées sont ignorés par Git. Ils doivent être conservés localement ou exportés séparément.

### Tester l’export FastAI

Après l’entraînement :

```bash
python test_modele_fastai.py \
  --model trained_models/dartsify_resnet34_local.pkl \
  --image Dartsify_mur/dataset/cam1_020526_124044.jpg
```

#### Fonctionnement du testeur

[test_modele_fastai.py](test_modele_fastai.py) est un test isolé du modèle
FastAI. Il ne démarre aucune caméra et ne contacte pas le backend. Il :

1. charge le Learner depuis le fichier `.pkl` avec `load_learner` ;
2. applique les transformations enregistrées dans le Learner ;
3. exécute `learner.predict` sur une image ;
4. conserve la classe `1`, utilisée pour représenter la fléchette ;
5. convertit la prédiction en masque binaire noir/blanc ;
6. recherche les contours externes ;
7. élimine les contours dont l’aire est inférieure à `75` pixels ;
8. sélectionne le plus grand contour et calcule son centroïde ;
9. remet ce point à l’échelle `1280 x 720` pour l’afficher sur l’image originale.

Le script affiche la taille du masque, le nombre de pixels de la classe
fléchette et le centroïde. Il sauvegarde aussi :

- un masque PNG dans `model_test_output/` ;
- une superposition de l’image, du masque en rouge et du centroïde prédit en vert.

#### Signification de la croix verte

La croix verte est ajoutée par le script uniquement pour visualiser le point
calculé par le post-traitement. Elle correspond au centroïde du plus grand
contour valide du masque prédit, après filtrage par l’aire minimale de `75`
pixels et remise à l’échelle vers l’image `1280 x 720`.

Il n’y a volontairement qu’une seule croix par image : le code sélectionne un
seul contour, celui dont l’aire est la plus grande. La croix ne représente donc
pas tous les points annotés dans `label.csv`. Par exemple, une image peut
contenir trois points annotés, mais le testeur n’affiche qu’un seul point
prédit, celui associé au plus grand contour détecté.

La croix verte n’est pas la vérité terrain. Les annotations du CSV ne sont pas
dessinées sur l’overlay actuel ; elles servent seulement à comparer ensuite la
coordonnée prédite à une coordonnée attendue. La croix peut donc être décalée
par rapport à la pointe réelle, notamment si le masque couvre une partie large
de la fléchette ou si plusieurs zones segmentées sont fusionnées.

Cette logique est volontairement proche de la fonction
`extraire_point_cible` du pipeline final, mais `main.py` applique d’abord la
différence avec le masque du lancer précédent et choisit une caméra. La croix
du testeur est donc un point de diagnostic du modèle sur une image isolée, pas
encore le point final après homographie envoyé au backend.

#### Différence avec `main.py`

Ce testeur n’est pas strictement équivalent au pipeline de [final/main.py](final/main.py).
Il valide la segmentation et le post-traitement d’une image isolée, mais pas le
fonctionnement complet du système.

| Testeur FastAI                                           | `main.py`                                                                       |
| -------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Charge un modèle `.pkl` avec FastAI                      | Charge un modèle `.onnx` avec ONNX Runtime                                      |
| Traite une image                                         | Traite les trois caméras en batch                                               |
| Utilise le prétraitement FastAI embarqué dans le Learner | Redimensionne et normalise manuellement avec OpenCV/NumPy                       |
| Utilise un seuil de contour de `75` pixels               | Utilise actuellement un seuil redimensionné de `50` pixels                      |
| Cherche le plus grand contour du masque courant          | Soustrait le masque du lancer précédent avant de chercher la nouvelle fléchette |
| Ne choisit aucune caméra                                 | Compte les contours puis choisit une caméra                                     |
| Ne fait pas d’homographie                                | Projette le point par homographie                                               |
| Ne fait aucun appel réseau                               | Envoie les coordonnées au backend HTTP                                          |

Le résultat du testeur répond donc à la question :

```text
image fixe -> modèle -> masque -> contour -> centroïde
```

Il ne garantit pas que `main.py` produira exactement le même point. Pour une
validation stricte, il faudra exporter ce modèle en ONNX, exécuter les deux
formats sur les mêmes images et comparer les masques et coordonnées obtenus.

#### Exemples générés

Des tests ont été réalisés sur trois caméras et sur une image sans annotation :

- caméra 1 : `cam1_020526_124100.jpg`, centroïde prédit `(366, 292)`, annotation la plus proche à environ `7,5` pixels ; [overlay](model_test_output/examples/cam1_020526_124100_overlay.jpg) ;
- caméra 2 : `cam2_020526_124044.jpg`, centroïde prédit `(676, 208)`, annotation la plus proche à environ `11` pixels ; [overlay](model_test_output/examples/cam2_020526_124044_overlay.jpg) ;
- caméra 3 : `cam3_020526_124044.jpg`, centroïde prédit `(598, 234)`, annotation la plus proche à environ `5,1` pixels ; [overlay](model_test_output/examples/cam3_020526_124044_overlay.jpg) ;
- image sans annotation : `cam1_060526_155703.jpg`, petit masque de `93` pixels détecté malgré l’absence de point attendu ; [overlay](model_test_output/examples/cam1_060526_155703_overlay.jpg).

Ces résultats sont encourageants, mais ils ne constituent pas encore une
évaluation statistique du modèle. Une évaluation complète devra mesurer les
erreurs sur plusieurs dizaines ou centaines d’images, ainsi que le nombre de
faux positifs sur les images sans annotation.

Le `.pkl` est destiné à l’expérimentation FastAI. Pour le pipeline final,
il est converti en `.onnx` avec `export_modele_onnx.py`. Le modèle actuellement
généré est `final/models_onnx/dartsify_resnet34_local.onnx`.
La constante `MODEL_PATH` de [final/main.py](final/main.py) a été mise à jour
pour charger ce fichier au démarrage.

### Conversion vers ONNX

Installer les bibliothèques d’export et d’exécution dans le même environnement :

```bash
python -m pip install onnx onnxruntime
```

Puis convertir le Learner :

```bash
python export_modele_onnx.py \
  --learner trained_models/dartsify_resnet34_local.pkl \
  --output final/models_onnx/dartsify_resnet34_local.onnx
```

L’export utilise le contrat attendu par le pipeline : entrée `(N, 3, 360, 640)`
et sortie `(N, 2, 360, 640)`, avec une dimension batch dynamique. Le
prétraitement RGB et la normalisation ImageNet sont réalisés par `main.py`, pas
à l’intérieur du fichier ONNX.

### Tester le pipeline ONNX sans caméra

Avant de brancher les caméras, tester le modèle au format final sur trois images :

```bash
python test_pipeline_onnx.py
```

Le script vérifie le chargement ONNX, exécute une inférence batch de trois images,
applique le même prétraitement que `main.py`, produit les masques et affiche les
centres détectés. Les résultats sont enregistrés dans `model_test_output/onnx/`.

Le test ne lance pas toute la boucle `main.py` : il ne simule ni caméra, ni
détection de mouvement, ni différence entre deux lancers, ni requête HTTP. Il
valide uniquement la partie modèle et post-traitement avant le test matériel.

### Test matériel de `main.py`

Une fois le modèle ONNX validé, le lancement complet se fait avec :

```bash
python final/main.py
```

Ce test nécessite les trois caméras, leurs index configurés, une calibration
homographique valide et un backend HTTP accessible. Il ne peut pas être
reproduit uniquement avec les images fixes. La version actuelle force le
backend caméra `cv2.CAP_MSMF`, adapté à Windows ; le test complet n’est donc
pas prévu tel quel sur macOS. De plus, `keyboard` peut poser un problème
d’import spécifique à macOS, alors que le test ONNX hors caméra reste
fonctionnel.

## Scripts utiles

- [mon_fichier_train.py](mon_fichier_train.py) : génère les masques, entraîne et exporte le modèle local.
- [test_modele_fastai.py](test_modele_fastai.py) : vérifie le modèle exporté sur une image fixe.
- [take_photo.py](take_photo.py) : ouvre les trois caméras, affiche les flux et enregistre une image sur chaque caméra quand on appuie sur `ESPACE`.
- [annotation.py](annotation.py) : ouvre une interface graphique pour ajouter, supprimer et naviguer dans les points d’annotation.
- [homographie.py](homographie.py) : calcule et vérifie une homographie à partir de points correspondants.

## Nettoyage de la branche

La branche a été nettoyée pour ne conserver que les fichiers nécessaires au
réentraînement local, aux tests du modèle et au pipeline de production.

Les éléments suivants ont été supprimés :

- `find_coord.py`, outil ponctuel qui dépendait d’anciens fichiers du dossier `image_test/` et ne faisait pas partie du pipeline final ;
- `final/main_DEBUG_3cams_newmask.py`, variante de debug du pipeline ONNX ;
- `final/main_DEBUG_memory.py`, autre variante expérimentale du pipeline ONNX ;
- les dossiers `__pycache__/` et autres caches Python générés automatiquement.

Le pipeline final est désormais représenté par [final/main.py](final/main.py)
et [finalRaspberry/main_onnx_avecPause.py](finalRaspberry/main_onnx_avecPause.py).
Les notebooks et outils d’annotation ou de calibration sont conservés car ils
peuvent encore servir à reproduire l’entraînement ou à modifier la calibration.

## Données et calibration

Le dossier [Dartsify_mur/](Dartsify_mur) contient les données utilisées pour le réentraînement local.

- [Dartsify_mur/dataset/](Dartsify_mur/dataset) : images brutes en `1280 x 720`.
- [Dartsify_mur/masks/](Dartsify_mur/masks) : masques générés automatiquement par `mon_fichier_train.py`.
- [Dartsify_mur/label.csv](Dartsify_mur/label.csv) : points annotés dans la colonne `Position`.

Ce point est important : ce fichier ne doit pas être traité comme un CSV classique avec un parseur naïf, car la colonne `Position` peut contenir des virgules à l’intérieur du JSON. Le code d’annotation lit donc les lignes une par une et découpe seulement sur la première virgule.

La calibration homographique est gérée par les scripts de la racine et par [final/matrice_homographie](final/matrice_homographie) :

- les points de correspondance sont déterminés à partir d’images de la cible ;
- une matrice 3x3 est calculée pour chaque caméra ;
- le point détecté dans l’image est ensuite projeté sur une vue de référence.

## Compatibilité et dépannage

Le projet contient une adaptation spécifique à Windows pour les modèles FastAI exportés sous Linux : plusieurs scripts remplacent `pathlib.PosixPath` par `pathlib.WindowsPath` avant l’appel à `load_learner`. Sans cette adaptation, le chargement du modèle peut échouer sur certaines configurations Windows.

Quelques points à surveiller si le pipeline ne démarre pas correctement :

- vérifier qu’un modèle `.pkl` est chargé avec FastAI et qu’un modèle `.onnx` est chargé avec ONNX Runtime ;
- vérifier que les index caméra configurés dans le code correspondent au branchement réel ;
- vérifier que le backend attend bien la route `/throws/` et la clé API configurée ;
- vérifier que les points et images utilisés pour une nouvelle calibration homographique sont disponibles localement ; les matrices de référence utilisées par le pipeline sont déjà stockées dans `final/matrice_homographie` et `finalRaspberry/matrice_homographie`.

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

> N.B. : la connexion est coupée momentanément lors du changement de réseau, mais on peut ensuite revenir via le portail ou via SSH.

### Éteindre la Raspberry

Il ne faut pas juste débrancher le câble. Il vaut mieux exécuter `sudo shutdown -h now`. Débrancher directement peut endommager la carte SD ou provoquer une corruption du système de fichiers.
