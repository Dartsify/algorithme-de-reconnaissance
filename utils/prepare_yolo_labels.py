"""Préparer les labels de fléchettes au format YOLO à partir des annotations DeepDarts."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


DEFAULT_IMAGE_SIZE = 800 # Taille des images carrees (800x800) pour YOLO.
DEFAULT_BBOX_SIZE = 0.025 # Comme stipulé dans le rapport de DeepDarts, 2.5% de la largeur/hauteur de l'image (800x800) pour chaque boîte englobante.


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    dataset_dir = repository_root / "datasets" / "deepdarts_d1"
    output_dir = repository_root / "datasets" / "deepdarts_d1_yolo"

    parser = argparse.ArgumentParser(
        description="Prepare les labels de flechettes au format YOLO."
    )
    parser.add_argument(
        "--source-labels",
        type=Path,
        default=dataset_dir / "labels.pkl",
        help="labels.pkl original contenant img_folder, img_name et xy.",
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        default=output_dir / "labels.pkl",
        help="labels.pkl simplifie a creer dans le dataset YOLO.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=output_dir / "images",
        help="Dossier contenant les images train et val.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Taille des images carrees (defaut : 800).",
    )
    parser.add_argument(
        "--bbox-size",
        type=float,
        default=DEFAULT_BBOX_SIZE,
        help="Largeur et hauteur normalisees des boites (defaut : 0.025).",
    )
    return parser.parse_args()


def load_source_labels(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Labels source introuvables : {path}")

    with path.open("rb") as handle:
        labels = pickle.load(handle)

    required_columns = {"img_folder", "img_name", "xy"}
    missing_columns = required_columns - set(labels.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colonnes absentes de labels.pkl : {missing}")
    return labels


def output_image_path(images_dir: Path, image_name: str) -> Path:
    matches = list(images_dir.glob(f"*/{image_name}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Impossible de retrouver une image unique pour {image_name} "
            f"dans {images_dir} (trouvees : {len(matches)})."
        )
    return matches[0]


def prepare_labels(
    source_labels_path: Path,
    output_labels_path: Path,
    images_dir: Path,
    image_size: int,
    bbox_size: float,
) -> int:
    if image_size <= 0:
        raise ValueError("--image-size doit etre strictement positif.")
    if not 0 < bbox_size <= 1:
        raise ValueError("--bbox-size doit etre compris entre 0 exclu et 1 inclus.")

    source_labels = load_source_labels(source_labels_path)
    prepared_rows = []
    expected_pixel_size = image_size * bbox_size
    discarded_points = 0

    for row in source_labels.itertuples(index=False):
        image_name = f"{row.img_folder}__{row.img_name}"
        image_path = output_image_path(images_dir, image_name)
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            raise ValueError(f"Format d'image non gere : {image_path}")

        points = row.xy[4:]
        yolo_labels = []
        for point in points:
            center_x, center_y = float(point[0]), float(point[1])
            if (
                center_x - bbox_size / 2 <= 0
                or center_x + bbox_size / 2 >= 1
                or center_y - bbox_size / 2 <= 0
                or center_y + bbox_size / 2 >= 1
            ):
                discarded_points += 1
                continue
            yolo_labels.append([0, center_x, center_y, bbox_size, bbox_size])

        prepared_rows.append({"img_name": image_name, "labels": yolo_labels})

    output_labels_path.parent.mkdir(parents=True, exist_ok=True)
    with output_labels_path.open("wb") as handle:
        pickle.dump(pd.DataFrame(prepared_rows), handle)

    print(f"{len(prepared_rows)} image(s) traitees.")
    print(f"{sum(len(row['labels']) for row in prepared_rows)} flechette(s) conservee(s).")
    print(f"{discarded_points} point(s) ignore(s) car la boite depassait l'image.")
    print(f"Calibration supprimee : 4 point(s) par image.")
    print(
        f"Boites YOLO : {bbox_size} x {bbox_size} normalise, "
        f"soit {expected_pixel_size:g} x {expected_pixel_size:g} pixels en "
        f"{image_size}x{image_size}."
    )
    print(f"Labels ecrits dans : {output_labels_path}")
    return len(prepared_rows)


def main() -> None:
    args = parse_args()
    prepare_labels(
        args.source_labels,
        args.output_labels,
        args.images_dir,
        args.image_size,
        args.bbox_size,
    )


if __name__ == "__main__":
    main()
