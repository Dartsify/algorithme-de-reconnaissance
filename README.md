# Algorithme de reconnaissance

Ce dépôt contient la partie code de l’algorithme de reconnaissance du projet **Strady**. Il est retravaillé à partir du projet de BAC III **Dartsify**.

L’objectif est d’installer cet algorithme sur une Raspberry Pi afin de détecter les fléchettes et de compter automatiquement les points.

Le projet repart actuellement de zéro sur la partie détection. Une première approche avec **UNET** a été testée, mais elle était trop lente et produisait de mauvais résultats. La nouvelle piste explorée est **YOLO**, qui devrait être mieux adaptée à la détection en temps réel sur Raspberry Pi.

Je vais opter pour YOLOv26.

## Préparation du dataset DeepDarts

Le dataset DeepDarts doit d’abord être extrait ([lien vers le dataset](https://ieee-dataport.org/open-access/deepdarts-dataset)) puis placé dans le dossier `datasets/deepdarts_d1/`.
Il faut conserver l’arborescence fournie par IEEE : le dossier `cropped_images/800/` contient les sous-dossiers de sessions et le fichier `labels.pkl` se trouve à la racine de `deepdarts_d1/` :

```text
datasets/
├── deepdarts_d1/
│   ├── cropped_images/
│   │   └── 800/
│   │       ├── d1_02_04_2020/
│   │       ├── d1_02_06_2020/
│   │       └── ...
│   └── labels.pkl
```

Le dossier `deepdarts_d1/` est conservé comme archive brute. Il ne faut pas y renommer ou déplacer les images, car le fichier `labels.pkl` original fait le lien entre chaque image et son dossier d’origine grâce aux colonnes `img_folder` et `img_name`.

### Création du dataset préparé

Le script [utils/flatten_deepdarts_images.py](utils/flatten_deepdarts_images.py) prépare une copie adaptée à la suite du projet :

- il lit les images depuis `deepdarts_d1/cropped_images/800/` et les annotations depuis `deepdarts_d1/labels.pkl` ;
- il crée `datasets/deepdarts_d1_yolo/` ;
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

Le dossier de sortie doit être vide ou ne pas encore exister. Le script s’arrête sinon afin d’éviter d’écraser une préparation précédente.

Après exécution, l’arborescence obtenue est la suivante :

```text
datasets/
├── deepdarts_d1/                # données brutes conservées
│   ├── cropped_images/800/      # images et sessions originales
│   └── labels.pkl               # annotations originales
├── deepdarts_d1_yolo/
│   ├── images/
│   │   ├── train/               # 80 % des sessions
│   │   └── val/                 # 20 % des sessions
│   └── labels.pkl               # labels YOLO préparés
```

La séparation est faite par session complète et non image par image. Les images d’une même session sont très proches ; mettre certaines dans `train` et d’autres dans `val` donnerait une évaluation artificiellement trop optimiste. La graine utilisée par défaut est `0`, ce qui rend la séparation reproductible.

Pour le pré-entraînement sur les 15 000 images DeepDarts, aucun ensemble `test` n’est créé. Le dossier `val` sert à suivre l’entraînement et à sélectionner le meilleur modèle. Un véritable ensemble `test` sera plus pertinent lors du fine-tuning sur les images Strady : une partie de ces images devra alors être conservée à l’écart jusqu’à l’évaluation finale.

### Nettoyage des annotations

De base, les données brutes de `labels.pkl` contiennent les colonnes : `img_folder`, `img_name`, `bbox` et `xy`. On va modifier tout ça pour ne garder que ce dont on a besoin.

Attention, le fichier `bbox` du dataset original ne correspond pas aux boîtes de détection dans les images `800x800` : il sert au recadrage des images originales avant leur redimensionnement. Les annotations utiles pour les fléchettes se trouvent dans la colonne `xy` du `labels.pkl` original : les quatre premiers points sont des points de calibration et les suivants sont les centres des fléchettes.

Le script [utils/prepare_yolo_labels.py](utils/prepare_yolo_labels.py) relit donc le `labels.pkl` original et remplace la copie située dans `deepdarts_d1_yolo/labels.pkl` :

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

Après exécution, le `labels.pkl` préparé contient deux colonnes : `img_name` et `labels`. Chaque élément de `labels` suit la forme YOLO `classe, x_centre, y_centre, largeur, hauteur`, avec des coordonnées normalisées entre `0` et `1`. Le `labels.pkl` original dans `deepdarts_d1/` reste inchangé.

### Vérification des labels pour une image donnée

Afin de vérifier si l'ensemble des conversions n'a pas cassé les labels, on peut vérifier avec le script `utils/see_labels_on_image.py` en écrivant en dur dans le code :

- le nom de l'image ;
- le tableau de labels généré par le script `utils/prepare_yolo_labels.py` (sous le format `[[classe, x_centre, y_centre, largeur, hauteur]]`).

Le script affiche l'image et montre les éventuelles bbox sur les fléchettes.

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
