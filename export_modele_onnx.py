from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.utils.parametrize as parametrize
import fastai.layers
from fastai.vision.all import load_learner


ROOT = Path(__file__).resolve().parent
DEFAULT_LEARNER = ROOT / "trained_models" / "dartsify_resnet34_local.pkl"
DEFAULT_OUTPUT = ROOT / "final" / "models_onnx" / "dartsify_resnet34_local.onnx"


def clean_model(model: nn.Module) -> None:
    """Retire les wrappers FastAI et parametrisations incompatibles avec ONNX."""
    for module in model.modules():
        if hasattr(module, "parametrizations"):
            for name in list(module.parametrizations.keys()):
                try:
                    parametrize.remove_parametrizations(module, name, leave_parametrized=True)
                except Exception:
                    pass
        if hasattr(module, "weight_u"):
            try:
                nn_utils.remove_spectral_norm(module)
            except Exception:
                pass

    def strip_wrappers(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, fastai.layers.ToTensorBase):
                setattr(module, name, nn.Identity())
            else:
                strip_wrappers(child)

    strip_wrappers(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertir un Learner FastAI en modele ONNX.")
    parser.add_argument("--learner", type=Path, default=DEFAULT_LEARNER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    if not args.learner.exists():
        raise FileNotFoundError(f"Learner introuvable: {args.learner}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Chargement: {args.learner}")
    learner = load_learner(args.learner, cpu=True)
    model = learner.model.eval().to("cpu")
    clean_model(model)
    dummy_input = torch.randn(1, 3, args.height, args.width)

    print(f"Export ONNX: {args.output}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(args.output),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    print(f"Modele ONNX cree: {args.output}")
    print(f"Contrat: input=(N, 3, {args.height}, {args.width}), output=(N, 2, {args.height}, {args.width})")


if __name__ == "__main__":
    main()
