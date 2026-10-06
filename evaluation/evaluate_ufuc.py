"""Evaluate one UFUC inference run with paired and distribution metrics.

Expected inference layout (created by infer.py --mode batch):

    pred_dir/<font>/test/<char>/<name>_styled.png
    pred_dir/_gt_from_input/<font>/test/<char>/<name>_gt.png

The script reports SSIM, RMSE, LPIPS, FID, Stroke-IoU, normalized symmetric
Chamfer distance, and direct binary-topology metrics. All paired metrics are
averaged per image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.ndimage import distance_transform_edt
from skimage import io
from skimage.color import gray2rgb, rgb2gray
from skimage.metrics import structural_similarity
from skimage.measure import euler_number, label
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import find_boundaries
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PRED_SUFFIXES = ("_styled", "_prediction", "_pred")
GT_SUFFIXES = ("_gt", "_target", "_truth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one UFUC ablation run")
    parser.add_argument("--pred_dir", required=True, help="Root directory of generated images")
    parser.add_argument(
        "--gt_dir",
        default=None,
        help="Ground-truth root; defaults to <pred_dir>/_gt_from_input",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for CSV/JSON results")
    parser.add_argument("--variant", required=True, help="Ablation name, e.g. full or no_mid")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help=(
            "If set, resize both prediction and ground truth to this square "
            "resolution before computing every metric (e.g. 128)."
        ),
    )
    parser.add_argument(
        "--stroke_threshold",
        type=float,
        default=0.5,
        help="Foreground threshold in [0,1] after grayscale conversion",
    )
    parser.add_argument(
        "--foreground",
        choices=("auto", "dark", "light"),
        default="auto",
        help="Whether glyph strokes are darker or lighter than the background",
    )
    parser.add_argument(
        "--topology_min_area",
        type=int,
        default=4,
        help=(
            "Ignore foreground components and enclosed holes smaller than this "
            "many pixels before topology measurement; use 0 to disable cleanup"
        ),
    )
    parser.add_argument(
        "--fid_dims",
        type=int,
        choices=(64, 192, 768, 2048),
        default=2048,
        help="Inception feature dimension used for FID",
    )
    return parser.parse_args()


def _strip_known_suffix(stem: str, suffixes: Sequence[str]) -> str:
    lower = stem.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _pair_key(path: Path, root: Path, suffixes: Sequence[str]) -> str:
    relative = path.relative_to(root)
    stem = _strip_known_suffix(relative.stem, suffixes)
    return (relative.parent / stem).as_posix()


def _list_images(root: Path, exclude_gt_folder: bool = False) -> List[Path]:
    result = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if exclude_gt_folder and "_gt_from_input" in path.parts:
            continue
        result.append(path)
    return sorted(result)


def build_pairs(pred_dir: Path, gt_dir: Path) -> List[Tuple[str, Path, Path]]:
    pred_map = {
        _pair_key(path, pred_dir, PRED_SUFFIXES): path
        for path in _list_images(pred_dir, exclude_gt_folder=True)
    }
    gt_map = {
        _pair_key(path, gt_dir, GT_SUFFIXES): path
        for path in _list_images(gt_dir)
    }
    common = sorted(set(pred_map) & set(gt_map))
    missing_gt = sorted(set(pred_map) - set(gt_map))
    missing_pred = sorted(set(gt_map) - set(pred_map))
    if missing_gt or missing_pred:
        preview_gt = ", ".join(missing_gt[:3]) or "none"
        preview_pred = ", ".join(missing_pred[:3]) or "none"
        raise RuntimeError(
            "Prediction/GT filenames do not match. "
            f"Missing GT: {len(missing_gt)} ({preview_gt}); "
            f"missing predictions: {len(missing_pred)} ({preview_pred})."
        )
    if not common:
        raise RuntimeError(f"No matched images found under {pred_dir} and {gt_dir}")
    return [(key, pred_map[key], gt_map[key]) for key in common]


def load_rgb(path: Path) -> np.ndarray:
    image = io.imread(path)
    if image.ndim == 2:
        image = gray2rgb(image)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = gray2rgb(image[..., 0])
    elif image.ndim == 3 and image.shape[2] >= 4:
        image = image[..., :3]
    image = image.astype(np.float32)
    if image.size > 0 and float(image.max()) > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=reference.shape[:2],
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return tensor.squeeze(0).permute(1, 2, 0).numpy()


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[:2] == (size, size):
        return image
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return tensor.squeeze(0).permute(1, 2, 0).numpy()


def stroke_mask(image: np.ndarray, threshold: float, foreground: str) -> np.ndarray:
    gray = rgb2gray(image)
    direction = foreground
    if direction == "auto":
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        direction = "dark" if float(np.median(border)) >= 0.5 else "light"
    return gray < threshold if direction == "dark" else gray > threshold


def stroke_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return 1.0 if union == 0 else float(intersection / union)


def normalized_chamfer(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_edge = find_boundaries(pred_mask, mode="inner")
    gt_edge = find_boundaries(gt_mask, mode="inner")
    pred_count = int(pred_edge.sum())
    gt_count = int(gt_edge.sum())
    if pred_count == 0 and gt_count == 0:
        return 0.0
    if pred_count == 0 or gt_count == 0:
        return 1.0
    distance_to_gt = distance_transform_edt(~gt_edge)
    distance_to_pred = distance_transform_edt(~pred_edge)
    forward = float(distance_to_gt[pred_edge].mean())
    backward = float(distance_to_pred[gt_edge].mean())
    diagonal = math.hypot(*pred_mask.shape)
    return ((forward + backward) * 0.5) / max(diagonal, 1.0)


def topology_signature(mask: np.ndarray, min_area: int) -> Tuple[int, int, int]:
    """Return (beta_0, beta_1, Euler characteristic) for a binary glyph mask.

    Foreground uses 8-connectivity. With this convention, ``euler_number``
    uses the complementary connectivity for the background. The same optional
    raster-noise cleanup is applied to predictions and ground truth.
    """
    clean = np.asarray(mask, dtype=bool)
    if min_area > 0:
        clean = remove_small_objects(clean, min_size=min_area, connectivity=2)
        clean = remove_small_holes(clean, area_threshold=min_area, connectivity=2)
    beta_0 = int(label(clean, connectivity=2).max())
    euler = int(euler_number(clean, connectivity=2))
    beta_1 = int(beta_0 - euler)
    return beta_0, beta_1, euler


def frechet_distance(real: np.ndarray, fake: np.ndarray, eps: float = 1e-6) -> float:
    if real.shape[0] < 2 or fake.shape[0] < 2:
        raise ValueError("FID requires at least two real and two generated images")
    mu_real, mu_fake = real.mean(axis=0), fake.mean(axis=0)
    sigma_real = np.cov(real, rowvar=False)
    sigma_fake = np.cov(fake, rowvar=False)
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("FID covariance product has a large imaginary component")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean))


def chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve() if args.gt_dir else pred_dir / "_gt_from_input"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    pairs = build_pairs(pred_dir, gt_dir)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(f"Matched samples: {len(pairs)}")
    print(f"Device: {device}")
    if args.image_size is not None:
        if args.image_size <= 0:
            raise ValueError(f"--image_size must be positive, got {args.image_size}")
        print(f"Metric resolution: {args.image_size}x{args.image_size}")
    else:
        print("Metric resolution: native ground-truth resolution")

    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS is required. Install with: pip install lpips") from exc
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as exc:
        raise RuntimeError("pytorch-fid is required. Install with: pip install pytorch-fid") from exc

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    block_index = InceptionV3.BLOCK_INDEX_BY_DIM[args.fid_dims]
    fid_model = InceptionV3([block_index]).to(device).eval()

    rows: List[Dict[str, object]] = []
    pred_features: List[np.ndarray] = []
    gt_features: List[np.ndarray] = []

    for batch_pairs in tqdm(list(chunks(pairs, args.batch_size)), desc="Evaluating"):
        pred_arrays = []
        gt_arrays = []
        batch_rows = []
        for key, pred_path, gt_path in batch_pairs:
            gt = load_rgb(gt_path)
            pred = load_rgb(pred_path)
            if args.image_size is not None:
                gt = resize_square(gt, args.image_size)
                pred = resize_square(pred, args.image_size)
            else:
                pred = resize_like(pred, gt)
            pred_mask = stroke_mask(pred, args.stroke_threshold, args.foreground)
            gt_mask = stroke_mask(gt, args.stroke_threshold, args.foreground)
            pred_beta0, pred_beta1, pred_euler = topology_signature(
                pred_mask, args.topology_min_area
            )
            gt_beta0, gt_beta1, gt_euler = topology_signature(
                gt_mask, args.topology_min_area
            )
            batch_rows.append(
                {
                    "key": key,
                    "prediction": str(pred_path),
                    "ground_truth": str(gt_path),
                    "ssim": float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0)),
                    "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
                    "stroke_iou": stroke_iou(pred_mask, gt_mask),
                    "chamfer": normalized_chamfer(pred_mask, gt_mask),
                    "pred_beta0": pred_beta0,
                    "gt_beta0": gt_beta0,
                    "beta0_error": abs(pred_beta0 - gt_beta0),
                    "pred_beta1": pred_beta1,
                    "gt_beta1": gt_beta1,
                    "beta1_error": abs(pred_beta1 - gt_beta1),
                    "pred_euler": pred_euler,
                    "gt_euler": gt_euler,
                    "euler_error": abs(pred_euler - gt_euler),
                    "topology_match": int(
                        pred_beta0 == gt_beta0 and pred_beta1 == gt_beta1
                    ),
                }
            )
            pred_arrays.append(pred)
            gt_arrays.append(gt)

        pred_tensor = torch.from_numpy(np.stack(pred_arrays)).permute(0, 3, 1, 2).to(device)
        gt_tensor = torch.from_numpy(np.stack(gt_arrays)).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            lpips_values = lpips_model(pred_tensor * 2.0 - 1.0, gt_tensor * 2.0 - 1.0).flatten()
            pred_activation = fid_model(pred_tensor)[0]
            gt_activation = fid_model(gt_tensor)[0]
            if pred_activation.shape[2:] != (1, 1):
                pred_activation = F.adaptive_avg_pool2d(pred_activation, output_size=(1, 1))
                gt_activation = F.adaptive_avg_pool2d(gt_activation, output_size=(1, 1))

        for row, value in zip(batch_rows, lpips_values.detach().cpu().numpy()):
            row["lpips"] = float(value)
        rows.extend(batch_rows)
        pred_features.append(pred_activation.squeeze(3).squeeze(2).cpu().numpy())
        gt_features.append(gt_activation.squeeze(3).squeeze(2).cpu().numpy())

    pred_feature_array = np.concatenate(pred_features, axis=0)
    gt_feature_array = np.concatenate(gt_features, axis=0)
    fid = frechet_distance(gt_feature_array, pred_feature_array)

    per_image_path = output_dir / "per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    paired_names = (
        "ssim",
        "rmse",
        "lpips",
        "stroke_iou",
        "chamfer",
        "beta0_error",
        "beta1_error",
        "euler_error",
        "topology_match",
    )
    summary: Dict[str, object] = {
        "variant": args.variant,
        "num_samples": len(rows),
        "image_size": args.image_size,
        "fid_dims": args.fid_dims,
        "stroke_threshold": args.stroke_threshold,
        "foreground": args.foreground,
        "topology_min_area": args.topology_min_area,
        "fid": fid,
    }
    for name in paired_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        summary[name] = float(values.mean())
        summary[f"{name}_sample_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    # Publication-facing alias: topology consistency rate is the mean of the
    # per-image exact (beta_0, beta_1) match indicator.
    summary["topology_consistency_rate"] = summary["topology_match"]
    summary["topology_consistency_rate_sample_std"] = summary["topology_match_sample_std"]

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Per-image metrics: {per_image_path}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()
