"""Visualiser les données contenues dans les fichiers labels.pkl (variante avec pandas pour créer un .csv dans l'arborescence)."""

import pandas as pd

# path = "datasets/deepdarts_d1/labels.pkl"
path = "datasets/deepdarts_d1_yolo/labels.pkl"

df = pd.read_pickle(path)

# Exporter en CSV
df.to_csv('labels_traduites.csv', index=False)