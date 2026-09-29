# Algorithme de reconnaissance <!-- omit in toc -->

Ce dépôt contient la partie code de l’algorithme de reconnaissance du projet **Strady**. Il est retravaillé à partir du projet de BAC III **Dartsify**.

L’objectif est d’installer cet algorithme sur une Raspberry Pi afin de détecter les fléchettes et de compter automatiquement les points.

Le projet repart actuellement de zéro sur la partie détection. Une première approche avec **UNET** a été testée, mais elle était trop lente et produisait de mauvais résultats. La nouvelle piste explorée est **YOLO**, qui devrait être mieux adaptée à la détection en temps réel sur Raspberry Pi.

Je vais opter pour YOLOv26 Tiny.

## Table des matières <!-- omit in toc -->

- [Création d'un environnement conda](#création-dun-environnement-conda)
- [Préparation du dataset DeepDarts avant entrainement](#préparation-du-dataset-deepdarts-avant-entrainement)
  - [1. Création du dataset préparé](#1-création-du-dataset-préparé)
  - [2. Nettoyage des annotations](#2-nettoyage-des-annotations)
  - [3. Vérification des labels pour une image donnée](#3-vérification-des-labels-pour-une-image-donnée)
  - [4. Conversion finale des labels au format YOLO](#4-conversion-finale-des-labels-au-format-yolo)
- [Configuration de l'entraînement](#configuration-de-lentraînement)
  - [Les fichiers `data.yaml`](#les-fichiers-datayaml)
  - [Le fichier configs/deepdarts_yolo.yaml](#le-fichier-configsdeepdarts_yoloyaml)
- [Premier entraînement (Baseline) et Suivi](#premier-entraînement-baseline-et-suivi)
- [Création de notre dataset](#création-de-notre-dataset)
- [Pipeline de préparation d'entrainement sur notre dataset](#pipeline-de-préparation-dentrainement-sur-notre-dataset)
  - [Récupération du dataset](#récupération-du-dataset)
  - [Traitement des images](#traitement-des-images)
  - [Conversion du dataset pour YOLO classique (Tiny ou autre)](#conversion-du-dataset-pour-yolo-classique-tiny-ou-autre)
- [Notes sur la Raspberry Pi](#notes-sur-la-raspberry-pi)
  - [Connexion à la Raspberry](#connexion-à-la-raspberry)
    - [Câble Ethernet Raspberry - Box WiFi](#câble-ethernet-raspberry---box-wifi)
    - [Câble Ethernet Raspberry - Ordinateur](#câble-ethernet-raspberry---ordinateur)
    - [WiFi](#wifi)
  - [Éteindre la Raspberry](#éteindre-la-raspberry)

## Création d'un environnement conda

Le fichier [environment.yml](environment.yml) contient l'ensemble des librairies installées via conda. Il faut donc avant tout créer un environnement conda et y installer toutes les dépendances nécessaires.

Pour créer un environnement et y installer tous les paquets listés dans le fichier :

```bash
conda env create -f environment.yaml
```

Pour activer l'environnement :

```bash
conda activate Strady_AlgoReconnaissance
```

Pour installer les dépendances dans un environnement déjà existant :

```bash
conda env update -f environment.yaml
```

> [!WARNING]
> Si votre environnement a un nom différent de celui établi (à savoir `Strady_AlgoReconnaissance`), vous devez remplacer `Strady_AlgoReconnaissance` par votre véritable nom d'environnement dans la commande `conda activate`.

## Préparation du dataset DeepDarts avant entrainement

> Le dataset DeepDarts doit être extrait et traité afin de correspondre aux attentes et formats de YOLO26. Je vais donc détailler dans cette section toutes les étapes nécessaires avant de lancer un entrainement. Ces étapes doivent être exécutés dans l'ordre établi.

> [!NOTE]
> Lors du premier entrainement, je me suis trompé : je n'ai pas vu que le dossier `cropped_images` extrait de IEEE contenait également le dataset `d2`. J'ai donc lancé un entrainement sur `d1` ET sur `d2` en même temps (or ce n'était pas voulu). J'ai donc par après modifié l'ensemble des scripts pour permettre à l'utilisateur de choisir quel dataset (ou plutôt subset) il veut préparer pour l'entrainement. Si certaines partie de ce texte contient des références à un dossier arbitraire `deepdarts_d1_yolo`, c'est une erreur : ce dossier dépend uniquement du choix de l'utilisateur. Idem si le texte parle d'un dossier de base `deepdarts_d1` dans lequel on place les différentes images extraites depuis IEEE : il s'agit en fait du dossier actualisé `deepdarts`.

Le dataset DeepDarts doit d’abord être extrait ([lien vers le dataset](https://ieee-dataport.org/open-access/deepdarts-dataset)) puis placé dans le dossier `datasets/deepdarts/`.
Il faut conserver l’arborescence fournie par IEEE : le dossier `cropped_images/800/` contient les sous-dossiers de sessions et le fichier `labels.pkl` se trouve à la racine de `deepdarts/` :

```text
datasets/
├── deepdarts/
│   ├── cropped_images/
│   │   └── 800/
│   │       ├── d1_02_04_2020/
│   │       ├── d1_02_06_2020/
│   │       └── ...
│   └── labels.pkl
```

Le dossier `deepdarts/` est conservé comme archive brute. Il ne faut pas y renommer ou déplacer les images, car le fichier `labels.pkl` original fait le lien entre chaque image et son dossier d’origine grâce aux colonnes `img_folder` et `img_name`.

Les étapes suivantes détaillent les différents script à exécuter dans cet ordre précis :

1. **Création du dataset préparé :** Exécuter [utils/flatten_deepdarts_images.py](utils/flatten_deepdarts_images.py) pour structurer les dossiers images/train et images/val par session.
2. **Nettoyage des annotations :** Exécuter [utils/prepare_yolo_labels.py](utils/prepare_yolo_labels.py) pour isoler les coordonnées des fléchettes et créer des boîtes normalisées de 2,5 %.
3. **Vérification (Optionnel, la vérification a déjà été faite) :** Utiliser [utils/see_labels_on_image.py](utils/see_labels_on_image.py) pour s'assurer visuellement que les boîtes encadrent bien les fléchettes.
4. **Conversion finale :** Lancer le script [utils/convert_labels.py](utils/convert_labels.py) pour générer les fichiers textes individuels exigés par l'architecture YOLO.

### 1. Création du dataset préparé

Le script [utils/flatten_deepdarts_images.py](utils/flatten_deepdarts_images.py) prépare une copie adaptée à la suite du projet :

- il lit les images depuis `deepdarts/cropped_images/800/` et les annotations depuis `deepdarts/labels.pkl` ;
- il crée `datasets/deepdarts_d1_yolo/` (ou deepdarts_yolo / deepdarts_d2_yolo) ;
- il copie les images dans `images/train/` et `images/val/` ;
- il renomme chaque image avec le nom de sa session pour éviter les doublons, par exemple `d1_02_04_2020__IMG_1081.JPG` ;
- il crée une première copie de `labels.pkl` contenant le nouveau nom de l’image et sa `bbox` ;
- il conserve le fichier original et les données brutes inchangés.

Le script doit être lancé avec l’environnement Conda du projet, `Strady_AlgoReconnaissance` :

```bash
conda run -n Strady_AlgoReconnaissance \
	python utils/flatten_deepdarts_images.py
```

(ou en exécutant le script depuis VS Code en ayant choisi le bon environnement conda en bas à droite)

Avant de faire la copie, il est possible de vérifier les opérations avec `--dry-run` :

```bash
conda run -n Strady_AlgoReconnaissance \
	python utils/flatten_deepdarts_images.py --dry-run
```

Pour spécifier le subset que vous désirez (i.e. `all`, `d1` ou `d2`), vous devez ajouter le flag `--subset` suivi d'une des trois valeurs possibles. Le script crée lui même les dossiers associés (e.g. `deepdarts_d1_yolo`)

Le dossier `images` dans le dossier de sortie doit être vide ou ne pas encore exister. Le script s’arrête sinon afin d’éviter d’écraser une préparation précédente.

Après exécution, l’arborescence obtenue est la suivante :

```text
datasets/
├── deepdarts/                # données brutes conservées
│   ├── cropped_images/800/      # images et sessions originales
│   └── labels.pkl               # annotations originales
├── deepdarts_d1_yolo/
│   ├── images/
│   │   ├── train/               # 80 % des sessions
│   │   └── val/                 # 20 % des sessions
│   └── labels.pkl               # labels YOLO préparés
```

> [!NOTE]
> Le dossier crée s'appelle `deepdarts_d1_yolo` (comme sur l'exemple d'arborescence) si vous avez choisi de préparer le dataset associé à `d1` (`--flag d1`).

La séparation est faite par session complète et non image par image. Les images d’une même session sont très proches ; mettre certaines dans `train` et d’autres dans `val` donnerait une évaluation artificiellement trop optimiste. La graine utilisée par défaut est `0`, ce qui rend la séparation reproductible.

Pour le pré-entraînement sur les 15 000 images DeepDarts, aucun ensemble `test` n’est créé. Le dossier `val` sert à suivre l’entraînement et à sélectionner le meilleur modèle. Un véritable ensemble `test` sera plus pertinent lors du fine-tuning sur les images Strady : une partie de ces images devra alors être conservée à l’écart jusqu’à l’évaluation finale.

### 2. Nettoyage des annotations

De base, les données brutes de `labels.pkl` contiennent les colonnes : `img_folder`, `img_name`, `bbox` et `xy`. On va modifier tout ça pour ne garder que ce dont on a besoin.

> [!WARNING] Attention
> Le fichier `bbox` du dataset original ne correspond pas aux boîtes de détection dans les images `800x800` : il sert au recadrage des images originales avant leur redimensionnement. Les annotations utiles pour les fléchettes se trouvent dans la colonne `xy` du `labels.pkl` original : les quatre premiers points sont des points de calibration et les suivants sont les centres des fléchettes.

Le script [utils/prepare_yolo_labels.py](utils/prepare_yolo_labels.py) relit donc le `labels.pkl` original et remplace la copie située dans [datasets/deepdarts_d1_yolo/labels.pkl](datasets/deepdarts_d1_yolo/labels.pkl) :

- il supprime les quatre points de calibration de chaque image ;
- il conserve uniquement les points correspondant aux fléchettes ;
- il crée une boîte YOLO de classe `0` autour de chaque point (DeepDarts donnait la classe 0 aux fléchettes, et les classe 1 à 4 aux points de calibrage) ;
- il utilise une boîte de `0.025 x 0.025` en coordonnées normalisées, soit `20 x 20` pixels pour une image `800x800` (les chercheurs de DeepDarts ont utilisé des bbox de 2,5% de la résolution des images 800x800) ;
- il ignore les points pour lesquels cette boîte dépasserait les limites de l’image (à la manière encore de DeepDarts, dans `deep-darts/data-loader.py` - visible dans leur repo - on peut le voir).

La commande à lancer est :

```bash
conda run -n Strady_AlgoReconnaissance \
	python utils/prepare_yolo_labels.py
```

Après exécution, le `labels.pkl` préparé contient deux colonnes : `img_name` et `labels`. Chaque élément de `labels` suit la forme YOLO `classe, x_centre, y_centre, largeur, hauteur`, avec des coordonnées normalisées entre `0` et `1`. Le `labels.pkl` original dans `deepdarts/` reste inchangé.

### 3. Vérification des labels pour une image donnée

Afin de vérifier si l'ensemble des conversions n'a pas cassé les labels, on peut vérifier avec le script [utils/see_labels_on_image.py](utils/see_labels_on_image.py) en écrivant en dur dans le code :

- le nom de l'image ;
- le tableau de labels généré par le script [utils/prepare_yolo_labels.py](utils/prepare_yolo_labels.py) (sous le format `[[classe, x_centre, y_centre, largeur, hauteur]]`).

Le script affiche l'image et montre les éventuelles bbox sur les fléchettes.

### 4. Conversion finale des labels au format YOLO

> [!WARNING]
> Cette étape doit impérativement être réalisée après le nettoyage et la vérification des annotations.

YOLO ne sait pas lire les fichiers de données Python sérialisés (`.pkl`). Il exige une arborescence stricte où chaque image possède un fichier texte homonyme. Le script [convert_labels.py](convert_labels.py) se charge de cette ultime adaptation :

- il lit le fichier `labels.pkl` préalablement nettoyé ;
- il vérifie la présence de chaque image dans `images/train` ou `images/val` ;
- il génère dynamiquement les dossiers correspondants `labels/train` et `labels/val` ;
- il crée un fichier `.txt` par image avec les coordonnées au format YOLO : `classe x_centre y_centre largeur hauteur`.

## Configuration de l'entraînement

Tous les paramètres importants de l'algorithme sont centralisés dans deux fichiers distincts.

### Les fichiers `data.yaml`

Il est purement informatif pour le framework YOLO. Il indique uniquement les chemins relatifs vers les dossiers d'images (`images/train` et `images/val`), le nombre de classes (`nc: 1`), et le nom de la classe ciblée (`names: ["dart"]`).

### Le fichier [configs/deepdarts_yolo.yaml](configs/deepdarts_yolo.yaml)

C'est le centre de contrôle du projet. Il regroupe les hyperparamètres d'apprentissage, les paramètres du modèle (YOLOv26 Nano), les règles d'augmentation visuelle, et les chemins de sauvegarde.

> [!NOTE]
> Il s'agit du fichier associé à `deepdarts_yolo` (regroupant les subsets `d1` et `d2`) mais il existe un fichier similaire pour les deux autres configuration de subsets.

> [!NOTE]
> Les seuls paramètres qui ne figurent pas dans ce fichier de configuration sont workers, device et le caractère deterministic. Étant strictement liés à l'optimisation de l'exécution matérielle locale, ils sont écrits en dur directement dans l'appel de la fonction du script [train.py](train.py).

## Premier entraînement (Baseline) et Suivi

Pour cette première approche sur les données DeepDarts, plusieurs partis pris techniques ont été appliqués dans [train.py](train.py) et [configs/deepdarts_d1_yolo.yaml](configs/deepdarts_d1_yolo.yaml) :

- **Optimisation matérielle (Mac M1 Max) :** Le `batch_size` a été fixé à 32 pour exploiter la mémoire unifiée. Le déterminisme strict a été désactivé (`deterministic=False`) afin de contourner une limitation du backend MPS de PyTorch, évitant ainsi les crashs et accélérant les calculs.
- **Apprentissage intelligent :** Le taux d'apprentissage (Learning Rate) manuel a été retiré pour laisser le mode auto-pilote de YOLO déterminer le meilleur optimiseur. L'entraînement est configuré sur 300 epochs, couplé à un mécanisme d'Early Stopping (`patience: 50`) qui stoppe l'apprentissage si les performances de validation stagnent, empêchant le surapprentissage.
- **Augmentations visuelles :** La fonction mosaic a été désactivée (0.0) car elle détruit la cohérence globale de la cible. En revanche, une perspective de 0.001 a été ajoutée pour reproduire nativement les déformations d'angle de caméra initialement codées par DeepDarts.
- **Suivi avec Weights & Biases (WandB) :** L'entraînement est connecté à WandB via `wandb.login()` et `wandb.init()`. Cependant, le script indique à YOLO de sauvegarder les poids et les résultats de détection en local dans le dossier pointé par `name=cfg["train"]["name"]`. Par conséquent, WandB ne réceptionne que les logs et affiche les métriques système de la machine (température CPU, utilisation GPU, etc.). Pour pallier cela lors de cette première session, l'évolution de la précision (mAP) et des pertes (Loss) a été tracée manuellement via des graphiques Excel à partir du fichier results.csv généré en local.

## Création de notre dataset

On a décidé d'utiliser les anciennes images afin d'avoir un dataset plus grand et de commencer à annoter avant de re disposer du dispositif complètement monté. On a annoté les images dans Roboflow pour un modèle YOLO Pose car cela permet, après conversion, d'avoir également un dataset prêt à l'emploi pour un modèle YOLO classique. On exporte les images de Roboflow sans pré-traitement car nous le faisons nous même dans les scripts du dossier `utils/strady/`. Pourquoi ? Car nous avons décidé de rogner les côtés gauche et droit des images avant de les resize pour ne pas donner ça à YOLO (c'est complètement superflus) -> on ne sait pas le faire lors de l'export du dataset dans Roboflow. J'ai donc créé différents scripts afin de convertir le dataset dans les formats dont nous avons besoin :

- **[`utils/strady/crop_and_resize.py`](utils/strady/crop_and_resize.py) :** s'occupe de prendre les images du dossier exporté de Roboflow et de les placer dans un nouveau dossier `datasets/strady_yolo_pose` (il les coupe et les redimensionne) ainsi que d'y placer les labels et une conversion du fichier `data.yaml` ;
- **[`utils/strady/convert_from_pose_to_tiny.py`](utils/strady/convert_from_pose_to_tiny.py) :** s'occupe de convertir les labels dans le format YOLO classique et de créer le dossier `datasets/strady_yolo` avec l'ensemble des images et des labels ;
- **[utils/strady/see_labels_on_image.py](utils/strady/see_labels_on_image.py) :** s'occupe d'afficher un retour visuel, il prend une image choisie dans le code et affiche les labels dessus (les bbox, la pointe etc) en fonction du dossier de l'image choisie.

## Pipeline de préparation d'entrainement sur notre dataset

### Récupération du dataset

Il faut télécharger une version danas Roboflow contenant l'ensemble des images avec le bon split décidé lors de l'ajout des images au dataset. Il ne faut pas choisir de pré-traitement (pas de rognage, redimensionnement, passage au gris etc). Ce sont nos scripts qui s'en occupent. Ensuite, on décide de télécharger en `.zip`.

Après le téléchargement du zip, il faut placer tous les fichiers et dossiers se trouvant dans le dossier dézippé dans le dossier [datasets/strady](datasets/strady/).

### Traitement des images

Il faut maintenant exécuter le script [utils/strady//crop_and_resize.py](utils/strady//crop_and_resize.py). Il va créer le dossier `datasets/strady_yolo_pose` et y placer tous les fichiers nécessaires.

Ce script effectue quatre actions principales pour adapter tes données brutes au format de ton projet (YOLO Pose, 800x800) :

1. Rognage et redimensionnement des images : Il supprime les marges inutiles de l'image d'origine en fonction des pixels définis (ex: 100px à gauche et à droite), puis redimensionne la zone restante au format strict de 800x800 pixels.
2. Recalcul des Bounding Boxes : Il convertit les coordonnées relatives de la boîte globale en pixels réels, applique le décalage mathématique lié au rognage, s'assure que la boîte est toujours visible, puis recalcule les coordonnées relatives (0 à 1) par rapport à la nouvelle dimension de l'image.
3. Recalcul des Keypoints (YOLO Pose) : Il applique exactement la même logique de décalage spatial aux coordonnées des points clés. Si le rognage amène un point clé en dehors du cadre, ses valeurs sont forcées à 0 0 0 pour indiquer à YOLO qu'il n'est plus visible.
4. Restructuration et configuration : Il enregistre les nouvelles images et labels dans une arborescence propre et conforme aux standards d'Ultralytics (images/val au lieu de valid). Enfin, il met à jour les chemins dans le fichier data.yaml et supprime les métadonnées liées à Roboflow.

### Conversion du dataset pour YOLO classique (Tiny ou autre)

Afin de disposer du dataset pour entraîner un YOLO classique, il faut exécuter le script [utils/strady/convert_from_pose_to_tiny.py](utils/strady/convert_from_pose_to_tiny.py).

Ce script effectue trois actions concises pour passer de ton dataset Pose au format YOLO classique (façon DeepDarts) :

1. Copie des images : Il transfère directement les images de strady_yolo_pose vers strady_yolo. Il ne fait aucun redimensionnement car les images sont déjà en 800x800.
2. Re-calcul des annotations (Labels) : Pour chaque fichier .txt, il ignore la bounding box globale existante. Il extrait uniquement les coordonnées x et y du premier point clé (la pointe de la fléchette). Il crée ensuite une nouvelle bounding box de taille fixe (0.025, soit 20x20 pixels) centrée exactement sur cette pointe. Le nouveau label respecte le format standard : classe x_centre y_centre largeur hauteur.
3. Mise à jour du data.yaml : Il copie ton fichier de configuration en supprimant les paramètres propres à YOLO Pose (kpt_shape et flip_idx). Cela permet à YOLO d'interpréter le dataset comme un simple problème de détection d'objets (YOLO Classique) sans générer d'erreur.

> [!WARNING]
> Ce script ne peut être exécuté uniquement après l'exécution du script précédent. Il se base strictement sur le dossier `datasets/strady_yolo_pose` pour créer le nouveau dossier.

---

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

> [!NOTE]
> Il serait aussi possible, dans ce cas, de récupérer les données de la base de données via l’API sans dépendre d’une connexion sans fil.

#### WiFi

Pour être totalement sans fil, nous pouvons connecter la Raspberry au WiFi sans fil. Deux choix s’offrent à nous :

- utiliser la carte SD et le logiciel Raspberry Pi Imager : en configurant la carte SD de manière appropriée, avec le nom du réseau et le mot de passe, la Raspberry se connecte au réseau lors du démarrage ;
- utiliser l’interface graphique : en branchant d’abord un câble Ethernet entre la Raspberry et l’ordinateur personnel pour lui fournir une connexion, puis en passant par Raspberry Pi Connect, on peut accéder à l’interface graphique de la Raspberry et renseigner les paramètres WiFi.

[Documentation officielle](https://www.raspberrypi.com/documentation/services/connect.html)

![Documentation officielle](doc_raspberry.png "Commandes de la documentation officielle")

> [!NOTE]
> La connexion est coupée momentanément lors du changement de réseau, mais on peut ensuite revenir via le portail ou via SSH.

### Éteindre la Raspberry

Il ne faut pas juste débrancher le câble. Il vaut mieux exécuter `sudo shutdown -h now`. Débrancher directement peut endommager la carte SD ou provoquer une corruption du système de fichiers.
