#!/usr/bin/env python
"""Run LGT-Net ONNX models with selectable ONNX Runtime backends."""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

from postprocessing.post_process import post_process
from preprocessing.pano_lsd_align import panoEdgeDetection, rotatePanorama
from utils.boundary import corners2boundaries
from utils.conversion import depth2xyz
from utils.writer import xyz2json
from visualization.boundary import draw_boundaries
from visualization.floorplan import draw_floorplan, draw_iou_floorplan


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("checkpoints/onnx/lgt_net_mp3d_opset17.onnx")
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512

PRIMARY_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LGT-Net ONNX inference with the CPU, CUDA, or TensorRT "
            "ONNX Runtime backend."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"ONNX model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--img-glob",
        "--img_glob",
        dest="img_glob",
        required=True,
        help="panorama image path or glob pattern",
    )
    parser.add_argument(
        "--backend",
        choices=tuple(PRIMARY_PROVIDERS),
        default="cpu",
        help="ONNX Runtime backend (default: cpu)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device ID used by the CUDA and TensorRT backends (default: 0)",
    )
    parser.add_argument(
        "--post-processing",
        "--post_processing",
        dest="post_processing",
        choices=("manhattan", "atalanta", "original"),
        default="manhattan",
        help="layout post-processing method (default: manhattan)",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=Path("src/output_onnx"),
        help="output directory (default: src/output_onnx)",
    )
    parser.add_argument(
        "--trt-engine-cache-dir",
        type=Path,
        default=None,
        help=(
            "TensorRT engine cache directory "
            "(default: <model directory>/.trt_cache/<model name>)"
        ),
    )
    parser.add_argument(
        "--trt-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable TensorRT FP16 inference (default: enabled)",
    )
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be zero or greater")
    return args


def resolve_path(path: Path) -> Path:
    """Resolve repository-relative defaults without restricting absolute paths."""
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()
    return (REPO_ROOT / path).resolve()


def build_providers(
    backend: str,
    device_id: int,
    trt_engine_cache_dir: Path,
    trt_fp16: bool,
) -> list[str | tuple[str, dict[str, Any]]]:
    """Build the same primary-to-fallback provider order used by demo_bpc.py."""
    if backend == "cpu":
        return ["CPUExecutionProvider"]

    cuda_options: dict[str, Any] = {"device_id": device_id}
    cuda_provider: tuple[str, dict[str, Any]] = (
        "CUDAExecutionProvider",
        cuda_options,
    )
    if backend == "cuda":
        return [cuda_provider, "CPUExecutionProvider"]

    trt_engine_cache_dir.mkdir(parents=True, exist_ok=True)
    tensorrt_options: dict[str, Any] = {
        "device_id": device_id,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(trt_engine_cache_dir),
        "trt_fp16_enable": trt_fp16,
        # Keep compatibility with the current ONNX Runtime TensorRT EP behavior.
        "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign",
    }
    return [
        ("TensorrtExecutionProvider", tensorrt_options),
        cuda_provider,
        "CPUExecutionProvider",
    ]


class OnnxLgtNet:
    """Small ONNX Runtime wrapper for exported LGT-Net models."""

    def __init__(
        self,
        model_path: Path,
        backend: str,
        device_id: int,
        trt_engine_cache_dir: Path,
        trt_fp16: bool,
    ) -> None:
        self.model_path = model_path
        self.backend = backend

        if backend != "cpu" and hasattr(ort, "preload_dlls"):
            # Load the CUDA and cuDNN libraries bundled with PyTorch before the
            # ONNX Runtime GPU providers are initialized.
            ort.preload_dlls()

        primary_provider = PRIMARY_PROVIDERS[backend]
        available_providers = ort.get_available_providers()
        if primary_provider not in available_providers:
            raise RuntimeError(
                f"{primary_provider} is not available. "
                f"Available providers: {available_providers}. "
                "Use --backend cpu or install the runtime libraries required "
                f"by the {backend} backend."
            )

        providers = build_providers(
            backend=backend,
            device_id=device_id,
            trt_engine_cache_dir=trt_engine_cache_dir,
            trt_fp16=trt_fp16,
        )
        ort.set_default_logger_severity(3)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize the {backend} backend for {model_path}. "
                "Confirm that its ONNX Runtime provider and runtime libraries "
                "are installed correctly."
            ) from exc

        active_providers = self.session.get_providers()
        if primary_provider not in active_providers:
            raise RuntimeError(
                f"The requested {primary_provider} could not be enabled. "
                f"Active providers: {active_providers}. Refusing to silently "
                "run with a fallback backend."
            )
        print(f"Enabled ONNX ExecutionProviders: {active_providers}")

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(
                f"Expected one ONNX input, but {model_path} has {len(inputs)}"
            )
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in outputs]
        if inputs[0].type != "tensor(float)":
            raise ValueError(
                f"Expected a float32 image input, but got {inputs[0].type}"
            )
        if not {"depth", "ratio"}.issubset(self.output_names):
            raise ValueError(
                "Expected ONNX outputs named 'depth' and 'ratio', but got "
                f"{self.output_names}"
            )

        input_shape = inputs[0].shape
        if len(input_shape) != 4 or input_shape[1:] != [3, IMAGE_HEIGHT, IMAGE_WIDTH]:
            raise ValueError(
                "Expected image shape [batch, 3, 512, 1024], but got "
                f"{input_shape}"
            )

    def __call__(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        input_tensor = np.ascontiguousarray(
            image.transpose(2, 0, 1)[None],
            dtype=np.float32,
        )
        start = time.perf_counter()
        outputs = self.session.run(
            output_names=self.output_names,
            input_feed={self.input_name: input_tensor},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return dict(zip(self.output_names, outputs, strict=True)), elapsed_ms


def align_to_vanishing_points(
    image: np.ndarray,
    vp_cache_path: Path,
    q_error: float = 0.7,
    refine_iter: int = 3,
) -> np.ndarray:
    """Apply the same Manhattan vanishing-point alignment as inference.py."""
    if vp_cache_path.is_file():
        with vp_cache_path.open(encoding="utf-8") as file:
            vp = np.asarray(
                [
                    [float(value) for value in line.rstrip().split(" ")]
                    for line in file
                ]
            )
    else:
        _, vp, _, _, _, _, _ = panoEdgeDetection(
            image,
            qError=q_error,
            refineIter=refine_iter,
        )
        with vp_cache_path.open("w", encoding="utf-8") as file:
            for row in vp:
                file.write("%.6f %.6f %.6f\n" % tuple(row))
    return rotatePanorama(image, vp[2::-1])


def show_alpha_floorplan(
    xyz: np.ndarray,
    side_length: int = IMAGE_HEIGHT,
) -> np.ndarray:
    floorplan = draw_floorplan(
        xz=xyz[..., ::2],
        fill_color=[0.2, 0.2, 0.2, 0.2],
        border_color=[0, 1, 0, 1],
        side_l=side_length,
        show=False,
        center_color=[1, 0, 0, 1],
    )
    floorplan_image = Image.fromarray(
        (floorplan * 255).astype(np.uint8),
        mode="RGBA",
    )
    background = np.empty((side_length, side_length, 4), dtype=np.float64)
    background[...] = [0.8, 0.8, 0.8, 1]
    background_image = Image.fromarray(
        (background * 255).astype(np.uint8),
        mode="RGBA",
    )
    return np.asarray(
        Image.alpha_composite(background_image, floorplan_image).convert("RGB")
    ) / 255.0


def save_visualization(
    image: np.ndarray,
    depth: np.ndarray,
    ratio: np.ndarray,
    processed_xyz: np.ndarray | None,
    save_path: Path,
) -> None:
    raw_xyz = depth2xyz(np.abs(depth[0]))
    ceiling_ratio = float(ratio[0, 0])
    boundaries = corners2boundaries(
        ceiling_ratio,
        corners_xyz=raw_xyz,
        step=None,
        visible=False,
        length=image.shape[1],
    )
    visualization = draw_boundaries(
        image,
        boundary_list=boundaries,
        boundary_color=[0, 1, 0],
    )

    if processed_xyz is not None:
        processed_boundaries = corners2boundaries(
            ceiling_ratio,
            corners_xyz=processed_xyz[0],
            step=None,
            visible=False,
            length=image.shape[1],
        )
        visualization = draw_boundaries(
            visualization,
            boundary_list=processed_boundaries,
            boundary_color=[1, 0, 0],
        )
        floorplan = draw_iou_floorplan(
            processed_xyz[0][..., ::2],
            raw_xyz[..., ::2],
            dt_board_color=[1, 0, 0, 1],
            gt_board_color=[0, 1, 0, 1],
        )
    else:
        floorplan = show_alpha_floorplan(raw_xyz)

    visualization = np.concatenate(
        [visualization, floorplan[:, 60:-60, :]],
        axis=1,
    )
    Image.fromarray((visualization * 255).astype(np.uint8)).save(save_path)


def save_prediction_json(xyz: np.ndarray, ratio: float, save_path: Path) -> None:
    with save_path.open("w", encoding="utf-8") as file:
        json.dump(xyz2json(xyz, ratio), file, indent=4)
        file.write("\n")


def load_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        resized = image.convert("RGB").resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            Image.Resampling.BICUBIC,
        )
    return np.asarray(resized)


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trt_engine_cache_dir = args.trt_engine_cache_dir
    if trt_engine_cache_dir is None:
        trt_engine_cache_dir = (
            model_path.parent / ".trt_cache" / model_path.stem
        )
    else:
        trt_engine_cache_dir = trt_engine_cache_dir.resolve()

    model = OnnxLgtNet(
        model_path=model_path,
        backend=args.backend,
        device_id=args.device_id,
        trt_engine_cache_dir=trt_engine_cache_dir,
        trt_fp16=args.trt_fp16,
    )

    image_paths = [Path(path) for path in sorted(glob.glob(args.img_glob))]
    image_paths = [path for path in image_paths if path.is_file()]
    if not image_paths:
        raise FileNotFoundError(f"No images matched: {args.img_glob}")

    progress = tqdm(image_paths, ncols=100)
    for image_path in progress:
        name = image_path.stem
        progress.set_description(name)
        image = load_image(image_path)
        if args.post_processing == "manhattan":
            image = align_to_vanishing_points(
                image,
                output_dir / f"{name}_vp.txt",
            )
        normalized_image = (image / 255.0).astype(np.float32)

        outputs, elapsed_ms = model(normalized_image)
        depth = outputs["depth"]
        ratio = outputs["ratio"]
        processed_xyz = None
        if args.post_processing != "original":
            processed_xyz = post_process(
                depth,
                type_name=args.post_processing,
            )

        save_visualization(
            normalized_image,
            depth,
            ratio,
            processed_xyz,
            output_dir / f"{name}_pred.png",
        )
        output_xyz = (
            processed_xyz[0]
            if processed_xyz is not None
            else depth2xyz(depth[0])
        )
        save_prediction_json(
            output_xyz,
            float(ratio[0, 0]),
            output_dir / f"{name}_pred.json",
        )
        progress.set_postfix(inference=f"{elapsed_ms:.1f} ms")

    print(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    main()
