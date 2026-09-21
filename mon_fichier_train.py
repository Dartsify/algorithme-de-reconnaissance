# Notes :
# J'ai trouvé dans mon ordi un ancien dossier d'annotations : Dartsify_mur. Il contient un CSV label.csv, un dossier dataset avec des images et un .zip des images.
# J'ai décidé de l'utiliser pour entraîner un modèle localement, en utilisant le code du notebook original. J'ai transféré tout le code du notebook dans ce fichier mon_fichier_train.py, en supprimant les parties inutiles et en adaptant le code pour qu'il fonctionne en dehors du notebook. J'ai commencé à transférer le code moi-même, mais je suis vite devenu dingue et j'ai demandé à l'IA dans vs code pour me le faire. Il a changé pour que ça fonctionne sur mon mac avec MPS l'os (dans les premières lignes), il a modifié l'import de tqdm pour que ça fonctionne en dehors du notebook et il a dégagé wan db (pas besoin pour juste tester). Il a modifié aussi la syntaxe de certaines variables je pense. 


import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Certaines augmentations FastAI utilisent torch.linalg.solve, non disponible
# sur MPS dans certaines versions de PyTorch; elles basculeront alors sur CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cv2
import fastai
from fastai.vision.all import *
from fastai.basics import *
from fastai.callback.all import *
import imutils
import torch
import ast
from pathlib import Path
from sklearn.model_selection import train_test_split
from fastai.callback.wandb import WandbCallback
from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback
import json


ROOT = Path(__file__).resolve().parent
LABEL_PATH = ROOT / "Dartsify_mur" / "label.csv"
DATASET_PATH = ROOT / "Dartsify_mur" / "dataset"
MASK_PATH = ROOT / "Dartsify_mur" / "masks"
OUTPUT_PATH = ROOT / "trained_models"
MASK_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MASK_RADIUS = 15
BATCH_SIZE = 2
EPOCHS = 10
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
USE_WANDB = os.getenv("DARTSIFY_USE_WANDB", "0") == "1"


def read_label_rows(label_path: Path) -> list[dict[str, str]]:
    """Lit le CSV dont la colonne Position contient des virgules JSON."""
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        image_id, _, position_text = line.partition(",")
        rows.append({"ID": image_id.strip().strip('"'), "Position": position_text.strip()})
    return rows


def parse_points(position_text: str) -> list[tuple[float, float]]:
    """Convertit la colonne Position en liste de points, vide si aucun point."""
    data = json.loads(position_text) if position_text else {}
    points = []
    for item in data.get("objects", []):
        point = item.get("point", {})
        if isinstance(point, dict) and {"x", "y"}.issubset(point):
            points.append((float(point["x"]), float(point["y"])))
    return points


def create_masks(dataframe: pd.DataFrame) -> None:
    """Crée un masque binaire avec un disque autour de chaque point annoté."""
    for row in tqdm(dataframe.itertuples(index=False), total=len(dataframe), desc="Création des masques"):
        mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        for x, y in parse_points(row.Position):
            if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
                cv2.circle(mask, (round(x), round(y)), MASK_RADIUS, 1, thickness=-1)
        if not cv2.imwrite(str(row.mask_path), mask):
            raise RuntimeError(f"Impossible d'écrire le masque: {row.mask_path}")


def split_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sépare les lignes en entraînement, validation et test."""
    train_validation, test = train_test_split(
        dataframe,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def build_dataloaders(train: pd.DataFrame, validation: pd.DataFrame):
    """Construit les DataLoaders FastAI avec les mêmes principes que le notebook."""
    combined = pd.concat([train, validation], ignore_index=True)
    valid_indices = list(range(len(train), len(combined)))

    data_block = DataBlock(
        blocks=(ImageBlock, MaskBlock),
        get_x=ColReader("image_path"),
        get_y=ColReader("mask_path"),
        splitter=IndexSplitter(valid_indices),
        item_tfms=[Resize((IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2))],
        batch_tfms=[
            *aug_transforms(
                mult=1.5,
                do_flip=True,
                flip_vert=True,
                max_rotate=20.0,
                max_zoom=1.15,
                max_lighting=0.3,
                max_warp=0.1,
            ),
            Normalize.from_stats(*imagenet_stats),
        ],
    )
    return data_block.dataloaders(combined, bs=BATCH_SIZE, num_workers=0)


def create_learner(dls):
    """Construit le U-Net utilisé dans le notebook d'origine."""
    learner = unet_learner(
        dls=dls,
        arch=resnet34,
        metrics=[DiceMulti(), JaccardCoeffMulti()],
        loss_func=FocalLossFlat(axis=1, gamma=4.437892927205953),
        self_attention=True,
        act_cls=Mish,
        pretrained=True,
        blur_final=True,
        opt_func=ranger,
        n_out=2,
    )
    return learner


# Lire toutes les annotations du dataset actuel.
df = pd.DataFrame(read_label_rows(LABEL_PATH), columns=["ID", "Position"])

# Supprimer les lignes qui n'ont pas d'ID
df = df.dropna(subset=["ID"])

df["image_path"] = df["ID"].apply(
    lambda x: DATASET_PATH / x
)

df["mask_path"] = df["ID"].apply(
    lambda x: MASK_PATH / f"{Path(x).stem}.png"
)


# Les images présentes dans le dossier mais absentes du CSV ne sont pas annotées.
df = df[df["image_path"].map(Path.exists)].copy()
create_masks(df)

df_train, df_validation, df_test = split_dataframe(df)
print(
    f"Dataset: {len(df)} images | train={len(df_train)} | "
    f"validation={len(df_validation)} | test={len(df_test)}"
)

dls = build_dataloaders(df_train, df_validation)
learn = create_learner(dls)

callbacks = [
    EarlyStoppingCallback(monitor="jaccard_coeff_multi", patience=3),
    SaveModelCallback(monitor="jaccard_coeff_multi", comp=np.greater),
]

wandb_module = None
if USE_WANDB:
    import wandb as wandb_module
    from fastai.callback.wandb import WandbCallback

    wandb_module.init(project="Dartsify AI", name="local_dartsify_training")
    callbacks.append(WandbCallback(log_model=True))

learn.fit_one_cycle(
    EPOCHS,
    lr_max=0.0007539865498213478,
    pct_start=0.9,
    wd=3.5639359903428303e-05,
    cbs=callbacks,
)

test_loader = learn.dls.test_dl(df_test, with_labels=True)
test_results = list(learn.validate(dl=test_loader))
print(f"Test loss: {test_results[0]:.4f}")
print(f"Test Dice: {test_results[1]:.4f}")
print(f"Test Jaccard: {test_results[2]:.4f}")

model_path = OUTPUT_PATH / "dartsify_resnet34_local.pkl"
learn.export(model_path)
print(f"Modèle exporté: {model_path}")

if wandb_module is not None:
    wandb_module.finish()


