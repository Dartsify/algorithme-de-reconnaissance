from pathlib import Path
import csv
"""
Script python pour écrire les labels dans un fichier CSV.
Il parcourt les images dans le dossier "dataset" et ajoute leur nom dans le fichier "label.csv" 
s'ils n'y sont pas déjà présents.
Le fichier CSV aura deux colonnes : "ID" pour le nom de l'image et "Label" pour le label associé (initialement vide).
"""

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Dartsify" / "dataset"
CSV_PATH = ROOT / "Dartsify" / "label.csv"


def main() -> None:
	images = sorted(
		path for path in DATASET_DIR.iterdir()
		if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
	)

	if not images:
		return

	existing_ids = set()
	if CSV_PATH.exists():
		with CSV_PATH.open("r", newline="", encoding="utf-8") as csv_file:
			reader = csv.DictReader(csv_file)
			for row in reader:
				existing_ids.add(row["ID"])

	with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

		for image in images[1:]:
			if image.name not in existing_ids:
				writer.writerow([image.name, ""])


if __name__ == "__main__":
	main()
