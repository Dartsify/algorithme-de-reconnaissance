from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "final" / "models_onnx" / "dartsify_resnet34_local.onnx"
DEFAULT_IMAGES = [
    # ROOT / "Dartsify_mur" / "dataset" / "cam1_020526_124044.jpg",
    # ROOT / "Dartsify_mur" / "dataset" / "cam2_020526_124044.jpg",
    # ROOT / "Dartsify_mur" / "dataset" / "cam3_020526_124044.jpg",
    ROOT / "Dartsify_mur" / "dataset" / "cam1_020526_124100.jpg",
    ROOT / "Dartsify_mur" / "dataset" / "cam2_020526_124100.jpg",
    ROOT / "Dartsify_mur" / "dataset" / "cam3_020526_124100.jpg",
]


def prepare_batch(images: list[np.ndarray]) -> np.ndarray:
    batch = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for image in images:
        resized = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - mean) / std
        batch.append(np.transpose(normalized, (2, 0, 1)))
    return np.asarray(batch, dtype=np.float32)


def mask_and_center(prediction: np.ndarray, minimum_area: float) -> tuple[np.ndarray, tuple[int, int] | None]:
    classes = np.argmax(prediction, axis=1)
    mask = (classes[0] == 1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
    if not valid:
        return mask, None
    contour = max(valid, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        x, y, width, height = cv2.boundingRect(contour)
        return mask, (x + width // 2, y + height // 2)
    return mask, (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tester l'inference ONNX du pipeline sans cameras.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--images", nargs="*", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model_test_output" / "onnx")
    parser.add_argument("--min-area", type=float, default=50.0)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Modele ONNX introuvable: {args.model}")
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in args.images]
    if any(image is None for image in images):
        missing = [str(path) for path, image in zip(args.images, images) if image is None]
        raise FileNotFoundError(f"Images introuvables ou illisibles: {missing}")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    tensor = prepare_batch(images)
    prediction = session.run([output_info.name], {input_info.name: tensor})[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modele: {args.model}")
    print(f"Entree: {input_info.name} {input_info.shape}")
    print(f"Sortie: {output_info.name} {prediction.shape}")

    for index, (path, image) in enumerate(zip(args.images, images)):
        mask, center = mask_and_center(prediction[index:index + 1], args.min_area)
        mask_path = args.output_dir / f"{path.stem}_mask_onnx.png"
        overlay_path = args.output_dir / f"{path.stem}_overlay_onnx.jpg"
        cv2.imwrite(str(mask_path), mask)
        resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = image.copy()
        red = np.zeros_like(image)
        red[:, :, 2] = resized_mask
        overlay = cv2.addWeighted(overlay, 0.7, red, 0.3, 0)
        point = None
        if center is not None:
            point = (round(center[0] * image.shape[1] / mask.shape[1]), round(center[1] * image.shape[0] / mask.shape[0]))
            cv2.drawMarker(overlay, point, (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.imwrite(str(overlay_path), overlay)
        print(f"{path.name}: pixels={int(np.count_nonzero(mask))}, centre_640x360={center}, centre_1280x720={point}")


if __name__ == "__main__":
    main()