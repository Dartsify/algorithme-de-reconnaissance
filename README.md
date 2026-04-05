# Algorithme-de-Reconnaissance

Algorithme de reconnaissance stocké sur une Raspberry Pi pour le projet Dartsify

> Notes de Mathias - Dimanche 5 avril 2026

## Table des Matières

## Notes sur la Raspberry Pi

### Connexion à la Raspberry

Afin de se connecter à la Raspberry, nous pouvons :

- Brancher un cable Ethernet dans le port Ethernet de la Raspberry afin d'assurer une connexion directe entre la box WiFi et la Raspberry ;
- Brancher un cable Ethernet afin d'assurer une connexion entre un ordinateur personnel et la Raspberry ;
- Connecter la Raspberry au WiFi (connexion sans fil).

#### Cable Ethernet Raspberry - Box WiFi

Cette solution est la plus simple mais nous ne pouvons l'utiliser pour la présentation du projet. En effet, nous n'aurons pas accès à un port Ethernet délivrant une connexion dans le local où nous présenterons.

#### Cable Ethernet Raspberry - Ordinateur

Cette solution nous permet de nous rendre indépendant d'une connexion filiaire avec un port Ethernet délivrant de la connexion réseau.
Après la configuration du _Partage Réseau_ et de la connexion de l'ordinateur au réseau WiFi, la Raspberry peut se voir délivrer la connexion via l'ordinateur.

Afin de se connecter à la Raspberry, il faut déterminer son adresse IP. Pour ceci, j'ai exécuté `arp -a -i bridge100`. Cette commande interroge la table _ARP_ (_Address Resolution Protocol_) en filtrant spécifiquement sur l'interface réseau appelée _bridge100_. Grâce au retour de cette commande, nous pouvons voir quelque chose comme `? (192.168.2.2) at 2c:cf:67:db:6d:c2 on bridge100 ifscope [bridge]`. Ainsi, l'adresse IP de la Raspberry s'avère être 192.168.2.2. Sur base de cette adresse, nous pouvons nous connecter via _ssh_ dans le terminal : `ssh raspberry@192.168.2.2`.

Cependant, cette solution présente le problème que, lors de la présentation de notre dispositif, un cable sera apparant entre la Raspberry et l'ordinateur faisant tourner le site web (mon ordinateur). Ceci laisse penser que notre solution n'est pas à 100% sans fil. Je trouve que cela peut paraitre comme un point négatif, alors que notre solution fonctionne en effet en 100% sans fil.

> N.B. : À noter que nous pourrions tout de même dans ce cas, j'imagine, récupérer les données de la base de données via l'_API_ sans dépendre d'une connexion sans fil, cela pourrait justement être une sécurité le jour de la présentation.

#### WiFi

Pour être 100% sans fil, nous pouvons simplement connecter la Raspberry au WiFi sans fil. Pour ceci, deux choix s'offrent à nous :

- Utiliser la carte SD et le logiciel _Raspberry Pi Imager_ :
  En configurant la carte SD de manière appropriée (i.e. fournir nom du réseau et mdp), la Raspberry est capable de se connecter au réseau lors de son _bootage_.
  > N.B. : Alexandre a essayé mais n'a pas réussi à faire fonctionner cette technique, c'est pourquoi j'ai trouvé un autre moyen de le faire.
- Utiliser l'interface graphique :
  Cette interface graphique était déjà installée sur la Raspberry. En utilisant la seconde méthode consistant à brancher un cable Ethernet entre la Raspberry et l'ordinateur personnel (afin de délivrer du réseau à la Raspberry, la première méthode étant évidemment aussi possible) et en se dirigeant vers le portail **Raspberry Pi Connect**, nous pouvons, après avoir connecté la Raspberry au compte déjà créé dans le portail, accéder à l'interface graphique de la Raspberry. Dès lors, nous sommes capable de fournir les informations de connexion au réseau WiFi sans fil.
  > N.B. : La connexion sera évidemment momentanément coupée, mais après quelques secondes nous pouvons faire la même opération afin de se connecter soit via le portail (en se dirigeant cette fois vers le terminal et non l'interface graphique) soit dans un terminal via ssh.

### Éteindre la Raspberry

Ne pas juste débrancher le cable mais plutôt exécuter `sudo shutdown -h now`. Juste débrancher le cable peut causer des problèmes avec la carte SD etc.
