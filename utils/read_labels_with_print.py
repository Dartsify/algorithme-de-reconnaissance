"""Visualiser les données contenues dans les fichiers labels.pkl."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Affiche le contenu d'un ou des fichiers labels.pkl."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Chemin vers labels.pkl. Sans argument, les deux chemins connus sont testes.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Affiche toutes les lignes d'un DataFrame au lieu des 10 premieres.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Nombre de lignes a afficher (defaut : 10). Ignore avec --all.",
    )
    parser.set_defaults(
        default_paths=(
            repository_root / "datasets" / "deepdarts_d1" / "labels.pkl",
            repository_root / "datasets" / "deepdarts_d1_yolo" / "labels.pkl",
        )
    )
    return parser.parse_args()


def load_labels(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def display_labels(path: Path, labels: Any, show_all: bool, head: int) -> None:
    print(f"\n=== {path} ===")
    print(f"Type : {type(labels).__name__}")

    if hasattr(labels, "shape"):
        print(f"Taille : {labels.shape}")
    if hasattr(labels, "columns"):
        print(f"Colonnes : {list(labels.columns)}")
        rows = labels if show_all else labels.head(head)
        print(rows.to_string(index=False))
    else:
        print(labels)


def main() -> None:
    args = parse_args()
    if args.head < 0:
        raise ValueError("--head doit etre positif ou nul.")

    paths = (args.path,) if args.path else args.default_paths
    existing_paths = [path for path in paths if path.is_file()]
    missing_paths = [path for path in paths if not path.is_file()]

    for path in missing_paths:
        print(f"Fichier introuvable : {path}")

    if not existing_paths:
        raise FileNotFoundError("Aucun fichier labels.pkl trouve.")

    for path in existing_paths:
        display_labels(path, load_labels(path), args.all, args.head)


if __name__ == "__main__":
    main()
