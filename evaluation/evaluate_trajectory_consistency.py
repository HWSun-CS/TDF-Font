"""Evaluate geometric agreement between TDF-Font and Log-Domain trajectories.

The frozen keypoint detector is used only as a common measurement space.  The
Log-Domain intermediate images remain the reference trajectory; they are not
treated as keypoint labels used during training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import t as student_t
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.frames_dataset import FramesDataset
from infer import build_models
from log.logger import Logger
from modules.keypoint_detector import KPDetector


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
METRIC_COLUMNS = [
    "pred_coord_error_norm",
    "linear_coord_error_norm",
    "pred_coord_error_pixel",
    "linear_coord_error_pixel",
]


def natural_key(value: str) -> List[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", value)]


def resolve_path(value: Optional[str], config_path: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, config_path.parent / path, PROJECT_ROOT / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def build_eval_dataset(config: dict, input_dir: str, is_train: bool = False) -> FramesDataset:
    src = config.get("dataset_params", {})
    allowed = {
        "frame_shape",
        "id_sampling",
        "pairs_list",
        "augmentation_params",
        "use_last_frame_as_style",
        "font_prefix",
        "use_mid_frame",
        "style_from_same_font",
    }
    params = {k: v for k, v in src.items() if k in allowed}
    params["use_mid_frame"] = False
    return FramesDataset(root_dir=input_dir, is_train=is_train, **params)


def image_names(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return [
        name
        for name in os.listdir(path)
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS and not name.startswith(".")
    ]


def validate_sequences(dataset: FramesDataset, expected_frames: int) -> Tuple[List[int], List[dict]]:
    valid: List[int] = []
    rejected: List[dict] = []
    for index, sample in enumerate(dataset.samples):
        names = image_names(sample["path"])
        if names:
            lexical = sorted(names)
            natural = sorted(names, key=natural_key)
            if lexical != natural:
                raise RuntimeError(
                    "Unsafe frame ordering detected in "
                    f"{sample['path']}. The existing dataset loader uses lexical sorting, "
                    "but lexical and natural order differ. Rename frames with zero-padded "
                    "indices (for example 00.png ... 11.png) before evaluation."
                )
        try:
            item = dataset[index]
            frame_count = int(item["video"].shape[1])
        except Exception as exc:
            rejected.append({"sequence": sample.get("video_id", str(index)), "reason": str(exc)})
            continue
        if frame_count != expected_frames:
            rejected.append(
                {
                    "sequence": sample.get("video_id", str(index)),
                    "reason": f"expected {expected_frames} frames, found {frame_count}",
                }
            )
            continue
        valid.append(index)
    return valid, rejected


def build_style_references(
    dataset: FramesDataset, valid_indices: Sequence[int], num_style_chars: int
) -> Tuple[Dict[str, List[int]], set]:
    by_font: Dict[str, List[int]] = defaultdict(list)
    for index in valid_indices:
        sample = dataset.samples[index]
        by_font[str(sample.get("font_id"))].append(index)

    references: Dict[str, List[int]] = {}
    selected = set()
    for font_id, indices in by_font.items():
        # Match infer.py: character folders are sorted lexicographically.
        ordered = sorted(indices, key=lambda i: str(dataset.samples[i].get("char_name", "")))
        refs = ordered[:num_style_chars]
        if len(refs) < num_style_chars:
            raise RuntimeError(
                f"Font {font_id} has only {len(refs)} valid reference characters; "
                f"{num_style_chars} are required."
            )
        references[font_id] = refs
        selected.update(refs)
    return references, selected


def to_tensor(video: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(video)).to(device=device, dtype=torch.float32)


@torch.no_grad()
def compute_style_vectors(
    dataset: FramesDataset,
    references: Dict[str, List[int]],
    style_encoder: torch.nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[dict]]:
    vectors: Dict[str, torch.Tensor] = {}
    rows: List[dict] = []
    for font_id, indices in references.items():
        images = []
        for index in indices:
            item = dataset[index]
            if "video" in item:
                video = to_tensor(item["video"], device)
                images.append(video[:, -1])
            elif "driving" in item:
                # Training-split items expose source/driving rather than the
                # complete video. Driving is the last frame, matching infer.py.
                images.append(to_tensor(item["driving"], device))
            else:
                raise RuntimeError(
                    f"Style-reference sample {item.get('name', index)} has neither video nor driving."
                )
            sample = dataset.samples[index]
            rows.append(
                {
                    "font_id": font_id,
                    "character_id": sample.get("char_name", ""),
                    "sequence_id": sample.get("video_id", str(index)),
                }
            )
        style_batch = torch.stack(images, dim=0)
        encoded = style_encoder(style_batch, sty=True)
        vectors[font_id] = encoded.mean(dim=0, keepdim=True)
    return vectors, rows


def build_teacher_detector(config: dict, checkpoint: Path, device: torch.device) -> KPDetector:
    common = dict(config["model_params"]["common_params"])
    common.pop("style_dim", None)
    detector = KPDetector(**config["model_params"]["kp_detector_params"], **common).to(device)
    Logger.load_checkpoint(str(checkpoint), kp_detector=detector)
    detector.eval().requires_grad_(False)
    return detector


def pixel_distance(diff: torch.Tensor, height: int, width: int) -> torch.Tensor:
    scaled = diff.clone()
    scaled[..., 0] *= (width - 1) / 2.0
    scaled[..., 1] *= (height - 1) / 2.0
    return torch.linalg.vector_norm(scaled, dim=-1).mean(dim=-1)


def norm_distance(diff: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(diff, dim=-1).mean(dim=-1)


def jacobian_distance(diff: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(diff, ord="fro", dim=(-2, -1)).mean(dim=-1)


def require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        bad = int((~torch.isfinite(value)).sum().item())
        raise RuntimeError(f"{name} contains {bad} NaN/Inf values.")


def summary_stats(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"N": 0, "mean": np.nan, "sd": np.nan, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    mean = float(arr.mean())
    if n == 1:
        return {"N": 1, "mean": mean, "sd": np.nan, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    half = float(student_t.ppf(0.975, n - 1) * se)
    return {"N": n, "mean": mean, "sd": sd, "se": se, "ci_lower": mean - half, "ci_upper": mean + half}


def add_tertile(values: pd.Series) -> pd.Series:
    if len(values) < 3:
        return pd.Series(["All"] * len(values), index=values.index)
    percentile = values.rank(method="first", pct=True)
    return pd.cut(
        percentile,
        bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)


def make_by_time(raw: pd.DataFrame, compute_jacobian: bool) -> pd.DataFrame:
    metrics = list(METRIC_COLUMNS)
    if compute_jacobian:
        metrics.extend(["pred_jac_error", "linear_jac_error"])
    rows = []
    for (time_index, time_value), group in raw.groupby(["time_index", "t"], sort=True):
        row = {"time_index": int(time_index), "t": float(time_value), "N": int(group["sequence_id"].nunique())}
        for metric in metrics:
            stats = summary_stats(group[metric])
            for key in ("mean", "sd", "se", "ci_lower", "ci_upper"):
                row[f"{metric}_{key}"] = stats[key]
        rows.append(row)
    return pd.DataFrame(rows)


def make_global(sample_df: pd.DataFrame, compute_jacobian: bool) -> pd.DataFrame:
    metric_map = {
        "Predicted trajectory vs Log-Domain (normalized)": "pred_coord_error_norm",
        "Teacher-linear trajectory vs Log-Domain (normalized)": "linear_coord_error_norm",
        "Predicted trajectory vs Log-Domain (pixel)": "pred_coord_error_pixel",
        "Teacher-linear trajectory vs Log-Domain (pixel)": "linear_coord_error_pixel",
        "Student endpoint vs Teacher target (normalized)": "endpoint_error_norm",
        "Student endpoint vs Teacher target (pixel)": "endpoint_error_pixel",
        "Predicted minus teacher-linear (normalized)": "pred_minus_linear_norm",
        "Predicted minus teacher-linear (pixel)": "pred_minus_linear_pixel",
    }
    if compute_jacobian:
        metric_map.update(
            {
                "Predicted trajectory Jacobian vs Log-Domain": "pred_jac_error",
                "Teacher-linear Jacobian vs Log-Domain": "linear_jac_error",
                "Student endpoint Jacobian vs Teacher target": "endpoint_jac_error",
            }
        )

    rows = []
    for label, column in metric_map.items():
        stats = summary_stats(sample_df[column])
        rows.append({"metric": label, "analysis_unit": "sequence", **stats})

        font_values = sample_df.groupby("font_id", sort=True)[column].mean()
        font_stats = summary_stats(font_values)
        rows.append({"metric": label, "analysis_unit": "font", **font_stats})
    return pd.DataFrame(rows)


def make_difficulty_table(sample_df: pd.DataFrame) -> pd.DataFrame:
    work = sample_df.copy()
    work["deformation_group"] = add_tertile(work["endpoint_displacement_pixel"])
    work["nonlinearity_group"] = add_tertile(work["path_length_ratio"])
    rows = []
    for dimension, column in (
        ("deformation_magnitude", "deformation_group"),
        ("trajectory_nonlinearity", "nonlinearity_group"),
    ):
        for subset in ("Low", "Medium", "High"):
            group = work[work[column] == subset]
            if group.empty:
                continue
            row = {"dimension": dimension, "subset": subset, "N": int(len(group))}
            for metric in (
                "pred_coord_error_pixel",
                "linear_coord_error_pixel",
                "pred_relative_error",
                "linear_relative_error",
                "pred_minus_linear_pixel",
            ):
                stats = summary_stats(group[metric])
                for key in ("mean", "sd", "ci_lower", "ci_upper"):
                    row[f"{metric}_{key}"] = stats[key]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_trajectory(by_time: pd.DataFrame, output_base: Path, pixel: bool) -> None:
    suffix = "pixel" if pixel else "norm"
    pred = f"pred_coord_error_{suffix}"
    linear = f"linear_coord_error_{suffix}"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = by_time["t"].to_numpy()
    for metric, label, color, marker in (
        (pred, "TDF-Font predicted trajectory", "#1764ab", "o"),
        (linear, "Teacher endpoint-linear trajectory", "#c44e52", "s"),
    ):
        mean = by_time[f"{metric}_mean"].to_numpy()
        low = by_time[f"{metric}_ci_lower"].to_numpy()
        high = by_time[f"{metric}_ci_upper"].to_numpy()
        ax.plot(x, mean, color=color, marker=marker, linewidth=1.8, markersize=4.5, label=label)
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Normalized trajectory time $t$")
    ax.set_ylabel("Mean keypoint deviation (pixels)" if pixel else "Mean keypoint deviation (normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in x], rotation=35, ha="right")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def kp_to_pixel(kp: np.ndarray, height: int, width: int) -> np.ndarray:
    out = kp.copy()
    out[..., 0] = (out[..., 0] + 1.0) * (width - 1) / 2.0
    out[..., 1] = (out[..., 1] + 1.0) * (height - 1) / 2.0
    return out


def save_debug_overlay(
    image: np.ndarray,
    teacher_kp: np.ndarray,
    predicted_kp: np.ndarray,
    output: Path,
) -> None:
    # image is C,H,W in [0,1]
    image_hwc = np.moveaxis(image, 0, -1)
    h, w = image_hwc.shape[:2]
    teacher_px = kp_to_pixel(teacher_kp, h, w)
    pred_px = kp_to_pixel(predicted_kp, h, w)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.clip(image_hwc, 0, 1))
    ax.scatter(teacher_px[:, 0], teacher_px[:, 1], s=34, marker="o", facecolors="none", edgecolors="#00d26a", label="Teacher on Log-Domain frame")
    ax.scatter(pred_px[:, 0], pred_px[:, 1], s=27, marker="x", color="#ff2d55", label="TDF-Font predicted")
    ax.set_xlim(0, w - 1)
    ax.set_ylim(h - 1, 0)
    ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.11), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trajectory geometric consistency evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Trained TDF-Font checkpoint")
    parser.add_argument("--teacher-checkpoint", default=None, help="Optional explicit frozen KPDetector checkpoint; defaults to the detector stored in the TDF-Font checkpoint")
    parser.add_argument("--style-encoder-ckpt", default=None)
    parser.add_argument("--input-dir", required=True, help="Dataset root containing target test-character sequences")
    parser.add_argument("--style-input-dir", default=None, help="Optional dataset root for style references; defaults to --input-dir")
    parser.add_argument("--style-split", choices=["train", "test"], default="test", help="Use train references for SFUC, or test references for UFUC")
    parser.add_argument("--protocol", choices=["SFUC", "UFUC"], default="SFUC", help="Evaluation protocol label; this depends on whether target fonts were seen in training, not on the style-reference split")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-style-chars", type=int, default=8)
    parser.add_argument("--include-style-targets", action="store_true", help="Evaluate reference characters too (not recommended because it leaks style references into targets)")
    parser.add_argument("--compute-jacobian", action="store_true")
    parser.add_argument("--debug-samples", type=int, default=5)
    parser.add_argument("--expected-frames", type=int, default=12)
    parser.add_argument("--expected-sequences", type=int, default=None, help="Optional expected valid count after excluding style references")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    generator, model_kp_detector, style_encoder, kp_transformer, use_transformer = build_models(
        config, args.checkpoint, device, args.style_encoder_ckpt
    )
    del generator
    if not use_transformer:
        raise RuntimeError("The checkpoint does not contain a usable kp_transformer; TDF-Font endpoint prediction cannot be evaluated.")
    model_kp_detector.eval().requires_grad_(False)
    style_encoder.eval().requires_grad_(False)
    kp_transformer.eval().requires_grad_(False)

    teacher_arg = args.teacher_checkpoint
    configured_teacher = config.get("teacher_checkpoint")
    teacher_path = resolve_path(teacher_arg, config_path) if teacher_arg else resolve_path(configured_teacher, config_path)
    if teacher_path is not None and not teacher_path.exists() and not teacher_arg:
        print(
            f"WARNING: configured teacher checkpoint was not found ({teacher_path}); "
            "using the frozen kp_detector stored in the TDF-Font checkpoint."
        )
        teacher_path = None
    if teacher_path is not None:
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
        teacher_detector = build_teacher_detector(config, teacher_path, device)
        teacher_source = str(teacher_path)
    else:
        teacher_detector = model_kp_detector
        teacher_source = f"kp_detector stored in {Path(args.checkpoint).resolve()}"

    # Targets always come from the unseen-character test split. Style
    # references may come from train (SFUC) or test (UFUC) independently.
    dataset = build_eval_dataset(config, args.input_dir, is_train=False)
    valid_indices, rejected = validate_sequences(dataset, args.expected_frames)
    style_root = args.style_input_dir or args.input_dir
    if args.style_split == "test" and Path(style_root).resolve() == Path(args.input_dir).resolve():
        style_dataset = dataset
        style_indices = valid_indices
    else:
        style_dataset = build_eval_dataset(
            config, style_root, is_train=(args.style_split == "train")
        )
        style_indices = list(range(len(style_dataset)))
    references, selected_refs = build_style_references(
        style_dataset, style_indices, args.num_style_chars
    )
    style_vectors, reference_rows = compute_style_vectors(
        style_dataset, references, style_encoder, device
    )
    pd.DataFrame(reference_rows).to_csv(output_dir / "style_references.csv", index=False, encoding="utf-8-sig")
    if rejected:
        pd.DataFrame(rejected).to_csv(output_dir / "rejected_sequences.csv", index=False, encoding="utf-8-sig")

    target_fonts = {str(dataset.samples[i].get("font_id")) for i in valid_indices}
    missing_style_fonts = sorted(target_fonts.difference(style_vectors))
    if missing_style_fonts:
        preview = ", ".join(missing_style_fonts[:10])
        raise RuntimeError(
            f"No {args.style_split} style references were found for "
            f"{len(missing_style_fonts)} target fonts: {preview}"
        )

    same_test_pool = style_dataset is dataset and args.style_split == "test"
    eval_indices = [
        i
        for i in valid_indices
        if args.include_style_targets or not same_test_pool or i not in selected_refs
    ]
    if not eval_indices:
        raise RuntimeError("No valid target sequences remain after reference exclusion.")

    debug_count = min(max(args.debug_samples, 0), len(eval_indices))
    debug_indices = set(random.sample(eval_indices, debug_count))
    debug_time = {index: random.randint(1, args.expected_frames - 2) for index in debug_indices}

    raw_rows: List[dict] = []
    sample_rows: List[dict] = []
    k_expected = int(config["model_params"]["common_params"].get("num_kp", 20))
    batch_size = max(1, int(args.batch_size))

    for start in tqdm(range(0, len(eval_indices), batch_size), desc="Trajectory consistency"):
        batch_indices = eval_indices[start : start + batch_size]
        items = [dataset[index] for index in batch_indices]
        videos = torch.stack([to_tensor(item["video"], device) for item in items], dim=0)  # B,C,T,H,W
        b, c, frame_count, height, width = videos.shape
        frames = videos.permute(0, 2, 1, 3, 4).reshape(b * frame_count, c, height, width)
        teacher_all = teacher_detector(frames)
        teacher_value = teacher_all["value"].reshape(b, frame_count, k_expected, 2)
        if teacher_value.shape[2] != k_expected:
            raise RuntimeError(f"Expected K={k_expected}, found K={teacher_value.shape[2]}")
        require_finite("teacher keypoint coordinates", teacher_value)

        source_images = videos[:, :, 0]
        model_source = model_kp_detector(source_images)
        style_batch = torch.cat(
            [style_vectors[str(dataset.samples[index].get("font_id"))] for index in batch_indices], dim=0
        )
        predicted_endpoint = kp_transformer(model_source, style_batch)
        require_finite("student source keypoint coordinates", model_source["value"])
        require_finite("student predicted endpoint coordinates", predicted_endpoint["value"])

        teacher_source_value = teacher_value[:, 0]
        teacher_target_value = teacher_value[:, -1]
        pred_source_value = model_source["value"]
        pred_endpoint_value = predicted_endpoint["value"]
        times = torch.linspace(0.0, 1.0, frame_count, device=device, dtype=videos.dtype)
        t_view = times.view(1, frame_count, 1, 1)
        teacher_linear_value = teacher_source_value[:, None] + t_view * (
            teacher_target_value[:, None] - teacher_source_value[:, None]
        )
        predicted_value = pred_source_value[:, None] + t_view * (
            pred_endpoint_value[:, None] - pred_source_value[:, None]
        )
        if not torch.allclose(predicted_value[:, 0], pred_source_value, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Sanity check failed: predicted trajectory at t=0 is not the student source.")
        if not torch.allclose(predicted_value[:, -1], pred_endpoint_value, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Sanity check failed: predicted trajectory at t=1 is not the predicted endpoint.")
        if not torch.allclose(teacher_linear_value[:, 0], teacher_source_value, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Sanity check failed: teacher-linear trajectory at t=0 is not the teacher source.")
        if not torch.allclose(teacher_linear_value[:, -1], teacher_target_value, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Sanity check failed: teacher-linear trajectory at t=1 is not the teacher target.")

        pred_norm = norm_distance(predicted_value - teacher_value)
        linear_norm = norm_distance(teacher_linear_value - teacher_value)
        pred_pixel = pixel_distance(predicted_value - teacher_value, height, width)
        linear_pixel = pixel_distance(teacher_linear_value - teacher_value, height, width)
        endpoint_norm = norm_distance(pred_endpoint_value - teacher_target_value)
        endpoint_pixel = pixel_distance(pred_endpoint_value - teacher_target_value, height, width)
        source_gap_norm = norm_distance(pred_source_value - teacher_source_value)

        endpoint_displacement_norm = norm_distance(teacher_target_value - teacher_source_value)
        endpoint_displacement_pixel = pixel_distance(teacher_target_value - teacher_source_value, height, width)
        segment_diff = teacher_value[:, 1:] - teacher_value[:, :-1]
        segment_lengths_pixel = pixel_distance(segment_diff, height, width) * k_expected
        # pixel_distance averages K, so divide back after summing time.
        path_length_pixel = segment_lengths_pixel.sum(dim=1) / k_expected
        path_length_ratio = path_length_pixel / endpoint_displacement_pixel.clamp_min(1e-8)

        pred_jac = linear_jac = endpoint_jac = None
        if args.compute_jacobian:
            if "jacobian" not in teacher_all or "jacobian" not in model_source or "jacobian" not in predicted_endpoint:
                raise RuntimeError("--compute-jacobian was requested, but at least one model output has no Jacobian.")
            teacher_jac = teacher_all["jacobian"].reshape(b, frame_count, k_expected, 2, 2)
            source_jac = model_source["jacobian"]
            endpoint_pred_jac = predicted_endpoint["jacobian"]
            require_finite("teacher Jacobian", teacher_jac)
            require_finite("student source Jacobian", source_jac)
            require_finite("student predicted endpoint Jacobian", endpoint_pred_jac)
            tj = times.view(1, frame_count, 1, 1, 1)
            pred_jac_path = source_jac[:, None] + tj * (endpoint_pred_jac[:, None] - source_jac[:, None])
            teacher_linear_jac = teacher_jac[:, :1] + tj * (teacher_jac[:, -1:] - teacher_jac[:, :1])
            pred_jac = jacobian_distance(pred_jac_path - teacher_jac)
            linear_jac = jacobian_distance(teacher_linear_jac - teacher_jac)
            endpoint_jac = jacobian_distance(endpoint_pred_jac - teacher_jac[:, -1])

        for local, index in enumerate(batch_indices):
            sample = dataset.samples[index]
            sequence_id = sample.get("video_id", str(index))
            font_id = str(sample.get("font_id"))
            character_id = str(sample.get("char_name", ""))
            intermediate_slice = slice(1, frame_count - 1)
            pred_global_norm = float(pred_norm[local, intermediate_slice].mean().cpu())
            linear_global_norm = float(linear_norm[local, intermediate_slice].mean().cpu())
            pred_global_pixel = float(pred_pixel[local, intermediate_slice].mean().cpu())
            linear_global_pixel = float(linear_pixel[local, intermediate_slice].mean().cpu())
            displacement_pixel = float(endpoint_displacement_pixel[local].cpu())
            sample_row = {
                "sequence_id": sequence_id,
                "font_id": font_id,
                "character_id": character_id,
                "pred_coord_error_norm": pred_global_norm,
                "linear_coord_error_norm": linear_global_norm,
                "pred_coord_error_pixel": pred_global_pixel,
                "linear_coord_error_pixel": linear_global_pixel,
                "endpoint_error_norm": float(endpoint_norm[local].cpu()),
                "endpoint_error_pixel": float(endpoint_pixel[local].cpu()),
                "source_detector_gap_norm": float(source_gap_norm[local].cpu()),
                "endpoint_displacement_norm": float(endpoint_displacement_norm[local].cpu()),
                "endpoint_displacement_pixel": displacement_pixel,
                "path_length_pixel": float(path_length_pixel[local].cpu()),
                "path_length_ratio": float(path_length_ratio[local].cpu()),
                "pred_relative_error": pred_global_pixel / max(displacement_pixel, 1e-8),
                "linear_relative_error": linear_global_pixel / max(displacement_pixel, 1e-8),
                "pred_minus_linear_norm": pred_global_norm - linear_global_norm,
                "pred_minus_linear_pixel": pred_global_pixel - linear_global_pixel,
            }
            if args.compute_jacobian:
                sample_row.update(
                    {
                        "pred_jac_error": float(pred_jac[local, intermediate_slice].mean().cpu()),
                        "linear_jac_error": float(linear_jac[local, intermediate_slice].mean().cpu()),
                        "endpoint_jac_error": float(endpoint_jac[local].cpu()),
                    }
                )
            sample_rows.append(sample_row)

            for time_index in range(1, frame_count - 1):
                row = {
                    "sequence_id": sequence_id,
                    "font_id": font_id,
                    "character_id": character_id,
                    "time_index": time_index,
                    "t": float(times[time_index].cpu()),
                    "pred_coord_error_norm": float(pred_norm[local, time_index].cpu()),
                    "linear_coord_error_norm": float(linear_norm[local, time_index].cpu()),
                    "pred_coord_error_pixel": float(pred_pixel[local, time_index].cpu()),
                    "linear_coord_error_pixel": float(linear_pixel[local, time_index].cpu()),
                }
                if args.compute_jacobian:
                    row["pred_jac_error"] = float(pred_jac[local, time_index].cpu())
                    row["linear_jac_error"] = float(linear_jac[local, time_index].cpu())
                raw_rows.append(row)

            if index in debug_indices:
                debug_t = debug_time[index]
                print(f"\nSanity check: {sequence_id}, t={float(times[debug_t]):.6f}")
                print("teacher source kp:\n", teacher_source_value[local].cpu().numpy())
                print("teacher target kp:\n", teacher_target_value[local].cpu().numpy())
                print("student predicted final kp:\n", pred_endpoint_value[local].cpu().numpy())
                safe_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(sequence_id)).strip("_")
                save_debug_overlay(
                    videos[local, :, debug_t].cpu().numpy(),
                    teacher_value[local, debug_t].cpu().numpy(),
                    predicted_value[local, debug_t].cpu().numpy(),
                    debug_dir / f"debug_{safe_id}_t_{debug_t:02d}.png",
                )

    raw = pd.DataFrame(raw_rows)
    sample_df = pd.DataFrame(sample_rows)
    by_time = make_by_time(raw, args.compute_jacobian)
    global_df = make_global(sample_df, args.compute_jacobian)
    difficulty_df = make_difficulty_table(sample_df)
    font_df = sample_df.groupby("font_id", as_index=False).agg(
        N=("sequence_id", "count"),
        pred_coord_error_pixel=("pred_coord_error_pixel", "mean"),
        linear_coord_error_pixel=("linear_coord_error_pixel", "mean"),
        endpoint_error_pixel=("endpoint_error_pixel", "mean"),
        endpoint_displacement_pixel=("endpoint_displacement_pixel", "mean"),
        path_length_ratio=("path_length_ratio", "mean"),
    )

    raw.to_csv(output_dir / "trajectory_errors_raw.csv", index=False, encoding="utf-8-sig")
    sample_df.to_csv(output_dir / "trajectory_errors_by_sequence.csv", index=False, encoding="utf-8-sig")
    by_time.to_csv(output_dir / "trajectory_errors_by_time.csv", index=False, encoding="utf-8-sig")
    global_df.to_csv(output_dir / "trajectory_errors_global.csv", index=False, encoding="utf-8-sig")
    difficulty_df.to_csv(output_dir / "trajectory_errors_by_difficulty.csv", index=False, encoding="utf-8-sig")
    font_df.to_csv(output_dir / "trajectory_errors_by_font.csv", index=False, encoding="utf-8-sig")
    plot_trajectory(by_time, output_dir / "trajectory_consistency", pixel=True)
    plot_trajectory(by_time, output_dir / "trajectory_consistency_normalized", pixel=False)

    endpoint_row = global_df[
        (global_df["metric"] == "Student endpoint vs Teacher target (pixel)")
        & (global_df["analysis_unit"] == "sequence")
    ].iloc[0]
    pred_row = global_df[
        (global_df["metric"] == "Predicted trajectory vs Log-Domain (pixel)")
        & (global_df["analysis_unit"] == "sequence")
    ].iloc[0]
    linear_row = global_df[
        (global_df["metric"] == "Teacher-linear trajectory vs Log-Domain (pixel)")
        & (global_df["analysis_unit"] == "sequence")
    ].iloc[0]

    protocol = args.protocol
    metadata = {
        "protocol": protocol,
        "input_dir": str(Path(args.input_dir).resolve()),
        "config": str(config_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "teacher_detector_source": teacher_source,
        "coordinate_convention": "[-1, 1]",
        "frame_count": args.expected_frames,
        "intermediate_states": args.expected_frames - 2,
        "discovered_sequences": len(dataset),
        "valid_sequences_before_reference_exclusion": len(valid_indices),
        "style_reference_sequences": len(selected_refs),
        "style_split": args.style_split,
        "style_input_dir": str(Path(style_root).resolve()),
        "style_targets_excluded": bool(same_test_pool and not args.include_style_targets),
        "evaluated_sequences": len(eval_indices),
        "rejected_sequences": len(rejected),
        "num_fonts": int(sample_df["font_id"].nunique()),
        "K": k_expected,
        "compute_jacobian": bool(args.compute_jacobian),
    }
    with (output_dir / "trajectory_evaluation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("\nTrajectory Consistency Evaluation")
    print("---------------------------------")
    print(f"Protocol: {protocol}")
    print(f"Valid sequences: {len(eval_indices)}")
    if args.expected_sequences is not None and len(eval_indices) != args.expected_sequences:
        print(f"WARNING: expected {args.expected_sequences} sequences, but evaluated {len(eval_indices)}.")
    print(f"Intermediate states per sequence: {args.expected_frames - 2}")
    print(f"Total evaluated intermediate states: {len(raw)}")
    print(f"K: {k_expected}")
    print("\nPredicted trajectory (pixels):")
    print(f"mean = {pred_row['mean']:.6f}; SD = {pred_row['sd']:.6f}; 95% CI = [{pred_row['ci_lower']:.6f}, {pred_row['ci_upper']:.6f}]")
    print("\nTeacher-linear trajectory (pixels):")
    print(f"mean = {linear_row['mean']:.6f}; SD = {linear_row['sd']:.6f}; 95% CI = [{linear_row['ci_lower']:.6f}, {linear_row['ci_upper']:.6f}]")
    print("\nEndpoint error (pixels):")
    print(f"mean = {endpoint_row['mean']:.6f}; SD = {endpoint_row['sd']:.6f}; 95% CI = [{endpoint_row['ci_lower']:.6f}, {endpoint_row['ci_upper']:.6f}]")
    print(f"\nOutput: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
