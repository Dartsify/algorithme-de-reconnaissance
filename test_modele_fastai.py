from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from fastai.vision.all import load_learner


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "trained_models" / "dartsify_resnet34_local.pkl"
DEFAULT_IMAGE = ROOT / "Dartsify_mur" / "dataset" / "cam1_020526_124044.jpg"
DEFAULT_OUTPUT = ROOT / "model_test_output"


def as_mask(prediction) -> np.ndarray:
    """Convertit la classe prédite FastAI en masque binaire 0/255."""
    mask = np.asarray(prediction)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0) if mask.shape[0] > 1 else mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Sortie FastAI inattendue: {mask.shape}; attendu une carte 2D.")
    return (mask == 1).astype(np.uint8) * 255


def find_center(mask: np.ndarray, minimum_area: float) -> tuple[int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        x, y, width, height = cv2.boundingRect(contour)
        return x + width // 2, y + height // 2
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Tester un export FastAI .pkl sur une image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-area", type=float, default=75.0)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Modèle introuvable: {args.model}")
    if not args.image.exists():
        raise FileNotFoundError(f"Image introuvable: {args.image}")

    learner = load_learner(args.model, cpu=True)
    prediction, _, probabilities = learner.predict(args.image)
    mask = as_mask(prediction)
    center = find_center(mask, args.min_area)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Impossible de lire l'image: {args.image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / f"{args.image.stem}_mask_fastai.png"
    overlay_path = args.output_dir / f"{args.image.stem}_overlay_fastai.jpg"
    cv2.imwrite(str(mask_path), mask)

    resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = resized_mask
    overlay = cv2.addWeighted(overlay, 0.7, red, 0.3, 0)

    if center is not None:
        scale_x = image.shape[1] / mask.shape[1]
        scale_y = image.shape[0] / mask.shape[0]
        point = (round(center[0] * scale_x), round(center[1] * scale_y))
        cv2.drawMarker(overlay, point, (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(overlay, f"centre {point}", (point[0] + 12, point[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(str(overlay_path), overlay)
    print(f"Modele: {args.model}")
    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")
    print(f"Masque: {mask.shape[1]}x{mask.shape[0]}")
    print(f"Pixels classe 1: {int(np.count_nonzero(mask))}")
    print(f"Centroide: {center}")
    print(f"Probabilites: {getattr(probabilities, 'shape', None)}")
    print(f"Masque sauvegarde: {mask_path}")
    print(f"Overlay sauvegarde: {overlay_path}")


if __name__ == "__main__":
    main()
