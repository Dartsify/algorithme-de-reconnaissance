"""Préparer une copie aplatie et separée du dataset DeepDarts."""

from __future__ import annotations

import argparse
import pickle
import random
import shutil
from pathlib import Path

import pandas as pd

# Note : le code prend en argument le subset qu'on veut extraire (all, d1, d2) et le dossier de sortie est généré automatiquement si non spécifié. Le fichier labels.pkl est également pris en compte pour filtrer les images selon le subset demandé.


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare une copie aplatie et separee du dataset DeepDarts."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=repository_root / "datasets" / "deepdarts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Dossier de sortie. Généré automatiquement si non spécifié.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Chemin vers labels.pkl. Par défaut : <dataset>/labels.pkl",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["all", "d1", "d2"],
        default="all",
        help="Sous-ensemble à extraire (all, d1, d2). Définit le dossier de sortie par défaut.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proportion de sessions reservees a la validation (defaut : 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Graine de separation reproductible (defaut : 0).",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Place toutes les images dans images/train.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les operations sans creer ni copier de fichiers.",
    )
    return parser.parse_args()


def load_labels(labels_path: Path, subset: str) -> pd.DataFrame:
    with labels_path.open("rb") as handle:
        labels = pickle.load(handle)

    required_columns = {"img_folder", "img_name", "bbox"}
    missing_columns = required_columns - set(labels.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colonnes absentes de labels.pkl : {missing}")

    # Filtrage du dataset selon le subset demandé
    if subset != "all":
        if "dataset" in labels.columns:
            labels = labels[labels["dataset"] == subset]
        else:
            # Rabattement sur le nom du dossier si la colonne 'dataset' n'existe pas
            labels = labels[labels["img_folder"].str.contains(subset, case=False)]
            
    if labels.empty:
        raise ValueError(f"Aucune image trouvée pour le sous-ensemble '{subset}'.")

    return labels


def build_session_split(
    sessions: list[str], val_ratio: float, seed: int, no_split: bool
) -> dict[str, str]:
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio doit etre compris entre 0 inclus et 1 exclu.")

    if no_split or val_ratio == 0 or len(sessions) < 2:
        return {session: "train" for session in sessions}

    shuffled_sessions = sessions.copy()
    random.Random(seed).shuffle(shuffled_sessions)
    validation_count = max(1, round(len(sessions) * val_ratio))
    validation_sessions = set(shuffled_sessions[:validation_count])
    return {
        session: "val" if session in validation_sessions else "train"
        for session in sessions
    }


def prepare_dataset(
    dataset_dir: Path,
    output_dir: Path,
    labels_path: Path,
    subset: str,
    val_ratio: float,
    seed: int,
    no_split: bool,
    dry_run: bool,
) -> tuple[int, int]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_dir}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels introuvables : {labels_path}")
    images_root = dataset_dir / "cropped_images" / "800"
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images introuvables : {images_root}")
    if (output_dir / "images").exists() or (output_dir / "labels.pkl").exists():
        raise FileExistsError(
            f"Le dossier de sortie contient déjà un dossier 'images' ou un fichier 'labels.pkl' : {output_dir}. "
            "Choisis un autre chemin avec --output ou nettoie le dossier."
        )

    labels = load_labels(labels_path, subset)
    sessions = sorted(labels["img_folder"].unique())
    session_split = build_session_split(sessions, val_ratio, seed, no_split)
    source_files = {}

    for row in labels.itertuples(index=False):
        source = images_root / row.img_folder / row.img_name
        if not source.is_file():
            raise FileNotFoundError(f"Image referencee absente : {source}")
        key = (row.img_folder, row.img_name)
        if key in source_files:
            raise ValueError(f"Reference dupliquee dans labels.pkl : {key}")
        source_files[key] = source

    if not dry_run:
        for split in set(session_split.values()):
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)

    output_labels = []
    for row in labels.itertuples(index=False):
        split = session_split[row.img_folder]
        output_name = f"{row.img_folder}__{row.img_name}"
        destination = output_dir / "images" / split / output_name
        source = source_files[(row.img_folder, row.img_name)]
        if destination.exists():
            raise FileExistsError(f"Destination dupliquee : {destination}")
        print(f"{source} -> {destination}")
        if not dry_run:
            shutil.copy2(source, destination)
        output_labels.append({"img_name": output_name, "bbox": row.bbox})

    if not dry_run:
        with (output_dir / "labels.pkl").open("wb") as handle:
            pickle.dump(pd.DataFrame(output_labels), handle)

    return len(output_labels), sum(split == "val" for split in session_split.values())


def main() -> None:
    args = parse_args()

    # Remplacement des chemins par défaut manquants
    if args.labels is None:
        args.labels = args.dataset / "labels.pkl"
        
    if args.output is None:
        if args.subset == "all":
            args.output = args.dataset.parent / "deepdarts_yolo"
        else:
            args.output = args.dataset.parent / f"deepdarts_{args.subset}_yolo"


    image_count, validation_sessions = prepare_dataset(
        args.dataset,
        args.output,
        args.labels,
        args.subset,
        args.val_ratio,
        args.seed,
        args.no_split,
        args.dry_run,
    )
    action = "seraient préparées" if args.dry_run else "préparées"
    print(f"{image_count} image(s) {action} vers {args.output.name}.")
    print(f"{validation_sessions} session(s) reservee(s) à la validation.")


if __name__ == "__main__":
    main()