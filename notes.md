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

`27-sept-2026`

Fin d'entrainement. Je pensais qu'il allait prendre le best vers 120, mais par peur de tout faire buguer je ne l'ai pas coupé à la main, je voulais qu'il se coupe tout seul.
Cependant, je vois ce matin qu'il ne s'est jamais coupé et qu'il était à l'epoch 221... Après vérification, il a effectivement pris l'epoch 175 comme meilleure même si selon le graphique elle est clairement moins bonne (le framework utilise une fonction qui prend en compte différentes métriques, donc il juge que le modèle était meilleur à l'epoch 175 qu'à l'epoch 120 malgré la remontée de la val_box - askip il se base aussi sur les mAP).

![alt text](runs/detect/deepdarts_d1_run/graphiques_perso/box_loss.png)

![alt text](runs/detect/deepdarts_d1_run/graphiques_perso/mAP.png)

![alt text](runs/detect/deepdarts_d1_run/graphiques_perso/precision_recall.png)

| epoch | time    | train/box_loss | train/cls_loss | train/l1_loss | metrics/precision (B) | metrics/recall(B) | metrics/mAP50(B) | metrics/mAP50-95(B) | val/box_loss | val/cls_loss | val/l1_loss | lr/pg0   | lr/pg1   | lr/pg2   | lr/pg3   | lr/pg4   | lr/pg5   | lr/pg6   | lr/pg7   |
| :---- | :------ | :------------- | -------------- | ------------- | --------------------- | ----------------- | ---------------- | ------------------- | ------------ | ------------ | ----------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 120   | 75353,9 | 1,16603        | 0,49774        | 0,00195       | 0,97804               | 0,96269           | 0,98447          | 0,64908             | 1,17285      | 0,43119      | 0,00159     | 0,018219 | 0,006073 | 0,018219 | 0,006073 | 0,018219 | 0,006073 | 0,018219 | 0,006073 |
| 125   | 78493,2 | 1,1595         | 0,49045        | 0,00193       | 0,97738               | 0,96235           | 0,98454          | 0,64903             | 1,17325      | 0,42826      | 0,00159     | 0,017724 | 0,005908 | 0,017724 | 0,005908 | 0,017724 | 0,005908 | 0,017724 | 0,005908 |
| 129   | 81003,2 | 1,15323        | 0,50392        | 0,00193       | 0,97731               | 0,96269           | 0,98458          | 0,6492              | 1,17204      | 0,42951      | 0,00159     | 0,017328 | 0,005776 | 0,017328 | 0,005776 | 0,017328 | 0,005776 | 0,017328 | 0,005776 |
| 175   | 109981  | 1,09955        | 0,46215        | 0,00182       | 0,97768               | 0,96616           | 0,98491          | 0,6501              | 1,18098      | 0,41821      | 0,0016      | 0,012774 | 0,004258 | 0,012774 | 0,004258 | 0,012774 | 0,004258 | 0,012774 | 0,004258 |

<details>
<summary><b>Notes de Gemini sur le choix de l'epoch 175 plutôt qu'une aux alentours de la 120</b></summary>

> La mAP50-95 (Mean Average Precision) dépend effectivement du bon placement de tes boîtes. Elle se base sur l'IoU (Intersection over Union), qui calcule le pourcentage de chevauchement parfait entre la boîte prédite par ton modèle et la boîte réelle. Le "50-95" signifie que l'algorithme est de plus en plus exigeant sur cette précision géographique, jusqu'à demander 95% de chevauchement.
>
> Alors comment la mAP50-95 peut-elle monter (amélioration) pendant que la val/box_loss monte aussi (dégradation) ?
>
> Voici ce qu'il se passe sous le capot entre l'époque 125 et 175 :
>
> - **Le poids de la confiance (cls_loss) :** La mAP ne regarde pas uniquement la géométrie pure, elle classe les prédictions selon la certitude du modèle. À l'époque 175, la perte de classification (val/cls_loss) a fortement baissé (de 0.428 à 0.418). Le modèle identifie les cibles avec beaucoup plus d'assurance. Ses meilleures prédictions remontent en haut du classement, ce qui fait mathématiquement grimper le score mAP global, même si les boîtes ont très légèrement bougé.
> - **Les seuils vs l'erreur brute :** La box_loss calcule la distance absolue d'erreur au pixel près sur l'ensemble de l'image. Si le modèle décale ses boîtes d'une fraction de millimètre supplémentaire par rapport à l'époque 120, la box_loss se dégrade. Cependant, si ce micro-décalage maintient tout de même le chevauchement (IoU) au-dessus du seuil requis (par exemple 80% ou 90%), la mAP ne pénalisera pas la détection.
>
> En résumé, à l'époque 175, le modèle est infiniment moins "parfait" au pixel près sur le tracé de la boîte (box_loss plus haute), mais il se trompe beaucoup moins sur l'identification de ce qu'il encadre (cls_loss plus basse). Le compromis calculé par la fonction de fitness estime que cette solidité de détection est préférable à la perfection du contour, d'où ce choix final.

</details>

Ensuite, j'ai testé sur des images issues d'internet. Évidemment, y'a tellement de conditions différentes que les résultats sont pas dingues. Néanmoins, niveau temps ça semble relativement rapide : sur mon mac je suis à 11.5ms pour l'inférence, 108.6ms pour le preprocess et 8.9ms pour le postprocess. On verra ce que ça donne sur la Raspberry mais c'est prometteur.
