"""Export all published LGT-Net checkpoints to simplified ONNX models."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify
import torch
from torch import nn

import models
from config.defaults import merge_from_file


REPO_ROOT = Path(__file__).resolve().parent
INPUT_SHAPE = (1, 3, 512, 1024)


@dataclass(frozen=True)
class ExportSpec:
    config: Path
    checkpoint: Path


EXPORT_SPECS = {
    "mp3d": ExportSpec(
        config=Path("src/config/mp3d.yaml"),
        checkpoint=Path("checkpoints/SWG_Transformer_LGT_Net/mp3d/best.pkl"),
    ),
    "zind": ExportSpec(
        config=Path("src/config/zind.yaml"),
        checkpoint=Path("checkpoints/SWG_Transformer_LGT_Net/zind/best.pkl"),
    ),
    "pano": ExportSpec(
        config=Path("src/config/pano.yaml"),
        checkpoint=Path("checkpoints/SWG_Transformer_LGT_Net/pano/best.pkl"),
    ),
    "s2d3d": ExportSpec(
        config=Path("src/config/s2d3d.yaml"),
        checkpoint=Path("checkpoints/SWG_Transformer_LGT_Net/s2d3d/best.pkl"),
    ),
    "ablation_study_full": ExportSpec(
        config=Path("src/config/ablation_study/full.yaml"),
        checkpoint=Path(
            "checkpoints/SWG_Transformer_LGT_Net/ablation_study_full/best.pkl"
        ),
    ),
}


class OnnxExportWrapper(nn.Module):
    """Give the dictionary-based model output stable ONNX output names."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(image)
        return output["depth"], output["ratio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export LGT-Net checkpoints to ONNX and simplify each model with onnxsim."
        )
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/onnx"),
        help="output directory, relative to the repository root by default",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch export device, for example cpu or cuda:0 (default: cpu)",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="export a dynamic batch dimension instead of the default fixed batch size 1",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(EXPORT_SPECS),
        default=list(EXPORT_SPECS),
        help="models to export (default: all published checkpoints)",
    )
    args = parser.parse_args()
    if args.opset <= 0:
        parser.error("--opset must be a positive integer")
    return args


def resolve_from_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def validate_inputs(model_names: Sequence[str]) -> None:
    missing = []
    for model_name in model_names:
        spec = EXPORT_SPECS[model_name]
        for path in (spec.config, spec.checkpoint):
            resolved = resolve_from_repo(path)
            if not resolved.is_file():
                missing.append(resolved)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required export files are missing:\n{formatted}")


def load_model(spec: ExportSpec, device: torch.device) -> OnnxExportWrapper:
    config = merge_from_file(str(resolve_from_repo(spec.config)))
    model_args = dict(config.MODEL.ARGS[0])
    model_args["pretrained_backbone"] = False
    model = getattr(models, config.MODEL.NAME)(**model_args)

    checkpoint = torch.load(
        resolve_from_repo(spec.checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint["net"] if "net" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return OnnxExportWrapper(model.eval()).eval().to(device)


def export_raw_onnx(
    model: OnnxExportWrapper,
    sample: torch.Tensor,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "depth": {0: "batch"},
            "ratio": {0: "batch"},
        }

    with torch.inference_mode():
        torch_outputs = tuple(tensor.cpu().numpy() for tensor in model(sample))
        torch.onnx.export(
            model,
            (sample,),
            output_path,
            opset_version=opset,
            input_names=["image"],
            output_names=["depth", "ratio"],
            dynamic_axes=dynamic_axes,
            # The legacy exporter is explicit because it supports the requested opset 17
            # and this model without adding an onnxscript dependency.
            dynamo=False,
            external_data=False,
            do_constant_folding=True,
        )
    return torch_outputs


def simplify_onnx(raw_path: Path, output_path: Path) -> None:
    raw_model = onnx.load(raw_path)
    onnx.checker.check_model(raw_model)
    simplified_model, check_ok = simplify(
        raw_model,
        check_n=1,
        test_input_shapes={"image": list(INPUT_SHAPE)},
    )
    if not check_ok:
        raise RuntimeError(f"onnxsim validation failed for {raw_path}")
    onnx.checker.check_model(simplified_model)
    onnx.save_model(simplified_model, output_path)


def validate_onnx(
    output_path: Path,
    sample: torch.Tensor,
    torch_outputs: tuple[np.ndarray, np.ndarray],
    dynamic_batch: bool,
) -> None:
    onnx.checker.check_model(str(output_path))
    model = onnx.load(output_path, load_external_data=False)
    graph_values = [*model.graph.input, *model.graph.output]
    for value in graph_values:
        batch_dimension = value.type.tensor_type.shape.dim[0]
        if dynamic_batch and batch_dimension.dim_param != "batch":
            raise RuntimeError(f"{value.name} does not have a dynamic batch dimension")
        if not dynamic_batch and batch_dimension.dim_value != 1:
            raise RuntimeError(f"{value.name} does not have a fixed batch size of 1")
    del model

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_outputs = session.run(None, {"image": sample.cpu().numpy()})
    for name, torch_output, onnx_output in zip(
        ("depth", "ratio"),
        torch_outputs,
        onnx_outputs,
        strict=True,
    ):
        np.testing.assert_allclose(
            onnx_output,
            torch_output,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"PyTorch and ONNX output differ for {name}",
        )


def export_checkpoint(
    model_name: str,
    spec: ExportSpec,
    output_dir: Path,
    opset: int,
    device: torch.device,
    dynamic_batch: bool,
) -> Path:
    output_path = output_dir / f"lgt_net_{model_name}_opset{opset}.onnx"
    raw_path = output_dir / f".{output_path.stem}.unsimplified.onnx"
    simplified_path = output_dir / f".{output_path.stem}.simplified.onnx"
    print(f"[{model_name}] Loading {resolve_from_repo(spec.checkpoint)}", flush=True)
    model = load_model(spec, device)
    generator = torch.Generator(device="cpu").manual_seed(0)
    sample = torch.rand(INPUT_SHAPE, generator=generator).to(device)

    try:
        batch_mode = "dynamic batch" if dynamic_batch else "fixed batch 1"
        print(
            f"[{model_name}] Exporting opset {opset} ({batch_mode}) to {raw_path}",
            flush=True,
        )
        torch_outputs = export_raw_onnx(
            model,
            sample,
            raw_path,
            opset,
            dynamic_batch,
        )
        print(f"[{model_name}] Simplifying with onnxsim", flush=True)
        simplify_onnx(raw_path, simplified_path)
        print(f"[{model_name}] Validating with ONNX Runtime", flush=True)
        validate_onnx(simplified_path, sample, torch_outputs, dynamic_batch)
        simplified_path.replace(output_path)
    finally:
        raw_path.unlink(missing_ok=True)
        simplified_path.unlink(missing_ok=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"[{model_name}] Wrote {output_path} ({size_mib:.1f} MiB)", flush=True)
    return output_path


def main() -> None:
    args = parse_args()
    validate_inputs(args.models)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    output_dir = resolve_from_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = [
        export_checkpoint(
            model_name=model_name,
            spec=EXPORT_SPECS[model_name],
            output_dir=output_dir,
            opset=args.opset,
            device=device,
            dynamic_batch=args.dynamic_batch,
        )
        for model_name in args.models
    ]

    print("Export complete:")
    for output_path in exported:
        print(f"  - {output_path}")


if __name__ == "__main__":
    main()
