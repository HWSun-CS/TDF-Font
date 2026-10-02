#!/usr/bin/env python3
"""Post-hoc L_mid hard-case analysis using endpoint generation quality.

Difficulty is independent of either evaluated model. It is the frozen-teacher
source-to-target endpoint displacement already used by
evaluate_trajectory_consistency.py. Full and w/o Mid share one immutable sample
and style-reference manifest.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_trajectory_consistency import (
    add_tertile,
    build_eval_dataset,
    build_style_references,
    build_teacher_detector,
    compute_style_vectors,
    image_names,
    natural_key,
    pixel_distance,
    resolve_path,
    to_tensor,
)
from evaluation.evaluate_ufuc import (
    normalized_chamfer,
    stroke_iou,
    stroke_mask,
    topology_signature,
)
from infer import build_models


VARIANTS = ("full", "no_mid")
PRIMARY_METRICS = ("ssim", "lpips", "stroke_iou", "chamfer", "tcr")
ALL_METRICS = ("ssim", "rmse", "lpips", "stroke_iou", "chamfer", "tcr")
POSITIVE_IMPROVEMENT = {
    "ssim": ("full", "no_mid"),
    "rmse": ("no_mid", "full"),
    "lpips": ("no_mid", "full"),
    "stroke_iou": ("full", "no_mid"),
    "chamfer": ("no_mid", "full"),
    "tcr": ("full", "no_mid"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-config", default="config/ablation/full.yaml")
    parser.add_argument("--no-mid-config", default="config/ablation/no_mid.yaml")
    parser.add_argument("--full-checkpoint", required=True)
    parser.add_argument("--no-mid-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--style-encoder-ckpt", default=None)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=("UFUC", "UFSC"), default="UFUC")
    parser.add_argument(
        "--target-split", choices=("test", "train"), default="test",
        help="UFUC normally uses test; UFSC normally uses train",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Existing shared evaluation_manifest.json; generated in output-dir if omitted",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-style-chars", type=int, default=8)
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument("--expected-frames", type=int, default=12)
    parser.add_argument("--stroke-threshold", type=float, default=0.5)
    parser.add_argument("--foreground", choices=("auto", "dark", "light"), default="auto")
    parser.add_argument("--topology-min-area", type=int, default=4)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sample_record(dataset, index: int) -> dict:
    sample = dataset.samples[index]
    names = sorted(image_names(sample["path"]))
    if len(names) < 2:
        raise RuntimeError(f"Sequence has fewer than two images: {sample['path']}")
    sequence_id = str(sample.get("video_id", index))
    return {
        "index": int(index),
        "sample_id": sequence_id,
        "source_id": f"{sequence_id}/{names[0]}",
        "target_id": f"{sequence_id}/{names[-1]}",
        "font_id": str(sample.get("font_id")),
        "character_id": str(sample.get("char_name", "")),
        "path": str(Path(sample["path"]).resolve()),
    }


def make_manifest(
    dataset,
    valid_indices: Sequence[int],
    num_style_chars: int,
    protocol: str,
    target_split: str,
    expected_frames: int,
) -> dict:
    references, selected = build_style_references(dataset, valid_indices, num_style_chars)
    eval_indices = [index for index in valid_indices if index not in selected]
    return {
        "version": 2,
        "protocol": protocol,
        "target_split": target_split,
        "input_dir": str(Path(dataset.root_dir).resolve()),
        "expected_frames": int(expected_frames),
        "num_style_chars": int(num_style_chars),
        "style_references": {
            str(font): [sample_record(dataset, index) for index in indices]
            for font, indices in sorted(references.items())
        },
        "evaluation_samples": [sample_record(dataset, index) for index in eval_indices],
    }


def manifest_signature(manifest: dict) -> dict:
    return {
        "input_dir": str(Path(manifest["input_dir"]).resolve()),
        "protocol": manifest.get("protocol", "UFUC"),
        "target_split": manifest.get("target_split", "test"),
        "style_references": {
            str(font): [str(row.get("sample_id", row.get("sequence_id"))) for row in rows]
            for font, rows in manifest["style_references"].items()
        },
        "evaluation_samples": [
            str(row.get("sample_id", row.get("sequence_id")))
            for row in manifest["evaluation_samples"]
        ],
    }


def load_or_create_manifest(path: Path, generated: dict) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if manifest_signature(existing) != manifest_signature(generated):
            raise RuntimeError(
                f"Existing manifest does not match the current protocol/dataset/reference selection: {path}"
            )
        print(f"Reused and verified evaluation manifest: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(generated, handle, ensure_ascii=False, indent=2)
    print(f"Created evaluation manifest: {path}")
    return generated


def manifest_indices(dataset, manifest: dict) -> Tuple[List[int], Dict[str, List[int]]]:
    by_id = {
        str(sample.get("video_id", index)): index
        for index, sample in enumerate(dataset.samples)
    }
    eval_indices: List[int] = []
    for row in manifest["evaluation_samples"]:
        sample_id = str(row.get("sample_id", row.get("sequence_id")))
        if sample_id not in by_id:
            raise RuntimeError(f"Manifest sample not found in dataset: {sample_id}")
        eval_indices.append(by_id[sample_id])
    references: Dict[str, List[int]] = {}
    for font, rows in manifest["style_references"].items():
        indices = []
        for row in rows:
            sample_id = str(row.get("sample_id", row.get("sequence_id")))
            if sample_id not in by_id:
                raise RuntimeError(f"Manifest style reference not found: {sample_id}")
            indices.append(by_id[sample_id])
        references[str(font)] = indices
    return eval_indices, references


def validate_sequences_with_progress(dataset, expected_frames: int) -> Tuple[List[int], List[dict]]:
    valid: List[int] = []
    rejected: List[dict] = []
    for index in tqdm(range(len(dataset)), desc="Validating 12-frame sequences", unit="seq"):
        sample = dataset.samples[index]
        names = image_names(sample["path"])
        if sorted(names) != sorted(names, key=natural_key):
            raise RuntimeError(
                f"Unsafe lexical frame order in {sample['path']}; use 00.png ... 11.png"
            )
        try:
            frame_count = int(dataset[index]["video"].shape[1])
        except Exception as exc:
            rejected.append(
                {"sequence": sample.get("video_id", str(index)), "reason": str(exc)}
            )
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


@torch.no_grad()
def compute_deformation_manifest(
    dataset,
    eval_indices: Sequence[int],
    teacher: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    expected_k: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for start in tqdm(
        range(0, len(eval_indices), batch_size), desc="Frozen-teacher deformation", unit="batch"
    ):
        batch_indices = list(eval_indices[start : start + batch_size])
        items = [dataset[index] for index in batch_indices]
        videos = torch.stack([to_tensor(item["video"], device) for item in items], dim=0)
        _, _, _, height, width = videos.shape
        endpoints = torch.stack((videos[:, :, 0], videos[:, :, -1]), dim=1)
        b, two, c, h, w = endpoints.shape
        teacher_out = teacher(endpoints.reshape(b * two, c, h, w))
        value = teacher_out["value"].reshape(b, two, expected_k, 2)
        if value.shape[2] != expected_k:
            raise RuntimeError(f"Expected frozen-teacher K={expected_k}, found {value.shape[2]}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Frozen-teacher keypoints contain NaN/Inf")
        displacement = pixel_distance(value[:, 1] - value[:, 0], height, width)
        for local, index in enumerate(batch_indices):
            record = sample_record(dataset, index)
            record["D_geo"] = float(displacement[local].cpu())
            rows.append(record)

    frame = pd.DataFrame(rows)
    # Exact existing trajectory-analysis grouping: stable row-order tie break
    # via rank(method='first'), followed by equal-probability tertiles.
    frame["group"] = add_tertile(frame["D_geo"]).str.lower()
    return frame


def quantize_like_infer(tensor: torch.Tensor) -> np.ndarray:
    """Reproduce infer.py PNG saving followed by evaluate_ufuc.py loading."""
    array = tensor.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return (array * 255.0).astype(np.uint8).astype(np.float32) / 255.0


@torch.no_grad()
def evaluate_variant(
    variant: str,
    config: dict,
    checkpoint: Path,
    dataset,
    eval_indices: Sequence[int],
    references: Dict[str, List[int]],
    device: torch.device,
    batch_size: int,
    style_encoder_ckpt: str | None,
    lpips_model: torch.nn.Module,
    stroke_threshold: float,
    foreground: str,
    topology_min_area: int,
) -> pd.DataFrame:
    generator, kp_detector, style_encoder, kp_transformer, use_transformer = build_models(
        config, str(checkpoint), device, style_encoder_ckpt
    )
    if not use_transformer:
        raise RuntimeError(f"Checkpoint lacks a usable kp_transformer: {checkpoint}")
    for model in (generator, kp_detector, style_encoder, kp_transformer):
        model.eval().requires_grad_(False)
    style_vectors, _ = compute_style_vectors(dataset, references, style_encoder, device)

    rows: List[dict] = []
    for start in tqdm(
        range(0, len(eval_indices), batch_size), desc=f"Endpoint metrics: {variant}", unit="batch"
    ):
        batch_indices = list(eval_indices[start : start + batch_size])
        items = [dataset[index] for index in batch_indices]
        videos = torch.stack([to_tensor(item["video"], device) for item in items], dim=0)
        source = videos[:, :, 0]
        target = videos[:, :, -1]
        kp_source = kp_detector(source)
        style_batch = torch.cat(
            [style_vectors[str(dataset.samples[index].get("font_id"))] for index in batch_indices],
            dim=0,
        )
        kp_final = kp_transformer(kp_source, style_batch)
        prediction = generator(source, kp_source=kp_source, kp_driving=kp_final)["prediction"]
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Endpoint shape mismatch for {variant}: {prediction.shape} vs {target.shape}"
            )

        pred_arrays = quantize_like_infer(prediction)
        gt_arrays = quantize_like_infer(target)
        pred_tensor = torch.from_numpy(pred_arrays).permute(0, 3, 1, 2).to(device)
        gt_tensor = torch.from_numpy(gt_arrays).permute(0, 3, 1, 2).to(device)
        lpips_values = (
            lpips_model(pred_tensor * 2.0 - 1.0, gt_tensor * 2.0 - 1.0)
            .flatten().detach().cpu().numpy()
        )

        for local, index in enumerate(batch_indices):
            pred = pred_arrays[local]
            gt = gt_arrays[local]
            pred_mask = stroke_mask(pred, stroke_threshold, foreground)
            gt_mask = stroke_mask(gt, stroke_threshold, foreground)
            pred_beta0, pred_beta1, _ = topology_signature(pred_mask, topology_min_area)
            gt_beta0, gt_beta1, _ = topology_signature(gt_mask, topology_min_area)
            sample = dataset.samples[index]
            rows.append(
                {
                    "sample_id": str(sample.get("video_id", index)),
                    "font_id": str(sample.get("font_id")),
                    "character_id": str(sample.get("char_name", "")),
                    "variant": variant,
                    "ssim": float(
                        structural_similarity(gt, pred, channel_axis=2, data_range=1.0)
                    ),
                    "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
                    "lpips": float(lpips_values[local]),
                    "stroke_iou": stroke_iou(pred_mask, gt_mask),
                    "chamfer": normalized_chamfer(pred_mask, gt_mask),
                    "tcr": int(pred_beta0 == gt_beta0 and pred_beta1 == gt_beta1),
                }
            )

    del generator, kp_detector, style_encoder, kp_transformer, style_vectors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def group_metrics(per_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ("low", "medium", "high"):
        for variant in VARIANTS:
            subset = per_image[
                (per_image["group"] == group) & (per_image["variant"] == variant)
            ]
            row = {"group": group, "variant": variant, "n": int(len(subset))}
            for metric in ALL_METRICS:
                values = subset[metric].to_numpy(dtype=np.float64)
                row[metric] = float(values.mean())
                row[f"{metric}_sample_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else math.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def improvement(full: pd.Series, no_mid: pd.Series, metric: str) -> pd.Series:
    first, second = POSITIVE_IMPROVEMENT[metric]
    values = {"full": full, "no_mid": no_mid}
    return values[first] - values[second]


def make_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ("low", "medium", "high"):
        full = metrics[(metrics["group"] == group) & (metrics["variant"] == "full")].iloc[0]
        no_mid = metrics[(metrics["group"] == group) & (metrics["variant"] == "no_mid")].iloc[0]
        row = {"group": group}
        for metric in ALL_METRICS:
            row[f"delta_{metric}"] = float(
                improvement(full[metric], no_mid[metric], metric)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_wide(per_image: pd.DataFrame) -> pd.DataFrame:
    identity = ["sample_id", "font_id", "character_id", "D_geo", "group"]
    full = per_image[per_image["variant"] == "full"][identity + list(ALL_METRICS)]
    no_mid = per_image[per_image["variant"] == "no_mid"][identity + list(ALL_METRICS)]
    paired = full.merge(
        no_mid,
        on=identity,
        how="inner",
        validate="one_to_one",
        suffixes=("_full", "_no_mid"),
    )
    expected = per_image[per_image["variant"] == "full"]["sample_id"].nunique()
    if len(paired) != expected:
        raise RuntimeError(f"Full/no_mid pairing mismatch: {len(paired)} vs {expected}")
    for metric in ALL_METRICS:
        paired[f"improvement_{metric}"] = improvement(
            paired[f"{metric}_full"], paired[f"{metric}_no_mid"], metric
        )
    return paired


def correlation_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ALL_METRICS:
        result = spearmanr(
            paired["D_geo"].to_numpy(dtype=np.float64),
            paired[f"improvement_{metric}"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "metric": metric,
                "spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
                "n": int(len(paired)),
                "positive_means": "Full better",
            }
        )
    return pd.DataFrame(rows)


def trend_assessment(delta: pd.DataFrame, correlations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ALL_METRICS:
        values = delta.set_index("group")[f"delta_{metric}"]
        rho_row = correlations[correlations["metric"] == metric].iloc[0]
        monotonic = bool(values["low"] <= values["medium"] <= values["high"])
        high_better = bool(values["high"] > 0)
        rho = float(rho_row["spearman_rho"])
        p_value = float(rho_row["p_value"])
        if monotonic and rho > 0 and p_value < 0.05 and abs(rho) >= 0.3:
            strength = "strong positive trend"
        elif values["high"] > values["low"] and rho > 0:
            strength = "weak/mixed positive trend"
        else:
            strength = "no positive increasing trend"
        rows.append(
            {
                "metric": metric,
                "delta_low": float(values["low"]),
                "delta_medium": float(values["medium"]),
                "delta_high": float(values["high"]),
                "monotonic_delta_increase": monotonic,
                "high_group_full_better": high_better,
                "spearman_rho": rho,
                "p_value": p_value,
                "assessment": strength,
            }
        )
    return pd.DataFrame(rows)


def print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.8f}"))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    config_paths = {
        "full": project_path(args.full_config),
        "no_mid": project_path(args.no_mid_config),
    }
    configs = {}
    for variant, path in config_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            configs[variant] = yaml.safe_load(handle)
    for variant in VARIANTS:
        frame_shape = configs[variant].get("dataset_params", {}).get("frame_shape", [])
        num_kp = int(
            configs[variant].get("model_params", {}).get("common_params", {}).get("num_kp", 0)
        )
        if list(frame_shape)[:2] != [256, 256]:
            raise RuntimeError(f"{variant} is not configured for 256x256: {frame_shape}")
        if num_kp != 20:
            raise RuntimeError(f"{variant} does not use K=20: K={num_kp}")
    checkpoints = {
        "full": Path(args.full_checkpoint).expanduser().resolve(),
        "no_mid": Path(args.no_mid_checkpoint).expanduser().resolve(),
    }
    for variant, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"{variant} checkpoint not found: {path}")
    if checkpoints["full"] == checkpoints["no_mid"]:
        raise RuntimeError("Full and no_mid resolved to the same checkpoint file")
    print("Resolved checkpoints:")
    for variant in VARIANTS:
        print(f"  {variant}: {checkpoints[variant]}")

    device = torch.device(
        args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    )
    if args.protocol == "UFUC" and args.target_split != "test":
        raise ValueError("UFUC must use --target-split test")
    if args.protocol == "UFSC" and args.target_split != "train":
        raise ValueError("UFSC must use --target-split train")

    select_train_split = args.target_split == "train"
    dataset = build_eval_dataset(
        configs["full"], str(input_dir), is_train=select_train_split
    )
    if select_train_split:
        # Constructor uses is_train to select train/, but endpoint evaluation
        # requires deterministic full-sequence loading rather than random
        # training pairs/mid frames.
        dataset.is_train = False
        dataset.use_mid_frame = False
    print(
        f"Validating {args.protocol} sequences from split={args.target_split}: "
        f"discovered={len(dataset)}"
    )
    valid_indices, rejected = validate_sequences_with_progress(dataset, args.expected_frames)
    if rejected:
        pd.DataFrame(rejected).to_csv(
            output_dir / "rejected_sequences.csv", index=False, encoding="utf-8-sig"
        )
        raise RuntimeError(
            f"Rejected {len(rejected)} sequences; see {output_dir / 'rejected_sequences.csv'}"
        )
    generated_manifest = make_manifest(
        dataset,
        valid_indices,
        args.num_style_chars,
        args.protocol,
        args.target_split,
        args.expected_frames,
    )
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest else output_dir / "evaluation_manifest.json"
    )
    manifest = load_or_create_manifest(manifest_path, generated_manifest)
    eval_indices, references = manifest_indices(dataset, manifest)
    if args.expected_samples > 0 and len(eval_indices) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} samples after reference exclusion, "
            f"found {len(eval_indices)}"
        )

    print("Loading Full checkpoint for frozen-teacher measurement...")
    full_models = build_models(
        configs["full"], str(checkpoints["full"]), device, args.style_encoder_ckpt
    )
    full_generator, full_kp, full_style, full_transformer, full_use_transformer = full_models
    if not full_use_transformer:
        raise RuntimeError("Full checkpoint lacks kp_transformer")

    configured_teacher = configs["full"].get("teacher_checkpoint")
    config_path = config_paths["full"]
    teacher_path = (
        resolve_path(args.teacher_checkpoint, config_path)
        if args.teacher_checkpoint else resolve_path(configured_teacher, config_path)
    )
    if teacher_path is not None and teacher_path.exists():
        teacher = build_teacher_detector(configs["full"], teacher_path, device)
        teacher_source = str(teacher_path)
    elif args.teacher_checkpoint:
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
    else:
        teacher = full_kp
        teacher.eval().requires_grad_(False)
        teacher_source = f"frozen kp_detector stored in {checkpoints['full']}"

    expected_k = int(configs["full"]["model_params"]["common_params"].get("num_kp", 20))
    if expected_k != 20:
        raise RuntimeError(f"This analysis requires frozen FOMM K=20, config has K={expected_k}")
    deformation = compute_deformation_manifest(
        dataset, eval_indices, teacher, device, max(1, args.batch_size), expected_k
    )
    q33 = float(deformation["D_geo"].quantile(1.0 / 3.0))
    q67 = float(deformation["D_geo"].quantile(2.0 / 3.0))
    deformation.to_csv(
        output_dir / "deformation_group_manifest.csv", index=False, encoding="utf-8-sig"
    )

    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS is required. Install with: pip install lpips") from exc
    lpips_model = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)

    # The generic evaluator loads models itself. Release the initial Full
    # bundle after deformation grouping, then evaluate each variant in turn to
    # keep peak GPU memory bounded and identical.
    del teacher, full_generator, full_kp, full_style, full_transformer, full_models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    frames = []
    for variant in VARIANTS:
        frame = evaluate_variant(
            variant=variant,
            config=configs[variant],
            checkpoint=checkpoints[variant],
            dataset=dataset,
            eval_indices=eval_indices,
            references=references,
            device=device,
            batch_size=max(1, args.batch_size),
            style_encoder_ckpt=args.style_encoder_ckpt,
            lpips_model=lpips_model,
            stroke_threshold=args.stroke_threshold,
            foreground=args.foreground,
            topology_min_area=args.topology_min_area,
        )
        frames.append(frame)
    per_image = pd.concat(frames, ignore_index=True)
    per_image = per_image.merge(
        deformation[["sample_id", "D_geo", "group"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if per_image[["D_geo", "group"]].isna().any().any():
        raise RuntimeError("At least one endpoint result lacks a deformation group")

    metrics = group_metrics(per_image)
    delta = make_delta_table(metrics)
    paired = paired_wide(per_image)
    correlations = correlation_table(paired)
    assessment = trend_assessment(delta, correlations)

    per_image.to_csv(
        output_dir / "deformation_endpoint_per_image.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(
        output_dir / "deformation_group_metrics.csv", index=False, encoding="utf-8-sig"
    )
    delta.to_csv(
        output_dir / "deformation_group_delta.csv", index=False, encoding="utf-8-sig"
    )
    correlations.to_csv(
        output_dir / "deformation_correlation.csv", index=False, encoding="utf-8-sig"
    )
    assessment.to_csv(
        output_dir / "deformation_trend_assessment.csv", index=False, encoding="utf-8-sig"
    )
    paired.to_csv(
        output_dir / "deformation_paired_improvements.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "protocol": args.protocol,
        "target_split": args.target_split,
        "input_dir": str(input_dir),
        "full_checkpoint": str(checkpoints["full"]),
        "no_mid_checkpoint": str(checkpoints["no_mid"]),
        "teacher_source": teacher_source,
        "deformation_definition": (
            "mean_k L2(pixel_scale * (p_target_teacher[k] - p_source_teacher[k]))"
        ),
        "coordinate_convention": "teacher coordinates in [-1,1] scaled by (W-1)/2 and (H-1)/2",
        "K": expected_k,
        "q33": q33,
        "q67": q67,
        "group_counts": deformation["group"].value_counts().to_dict(),
        "manifest": str(manifest_path),
        "same_manifest_for_both_variants": True,
        "grouping_uses_predictions": False,
        "stroke_threshold": args.stroke_threshold,
        "foreground": args.foreground,
        "topology_min_area": args.topology_min_area,
        "trend_assessment_rule": (
            "strong positive: monotonic group deltas, rho>=0.3 and p<0.05; "
            "weak/mixed positive: delta_high>delta_low and rho>0; otherwise no positive trend"
        ),
    }
    with (output_dir / "deformation_analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"\nL_mid hard-case / deformation-magnitude analysis ({args.protocol})")
    print("=================================================")
    print("Deformation definition reused from trajectory analysis: YES")
    print("Code: evaluation/evaluate_trajectory_consistency.py::pixel_distance")
    print(
        "D_geo = mean over K=20 of Euclidean teacher source-target KP displacement; "
        "x scaled by (W-1)/2 and y by (H-1)/2; unit=pixels; Jacobian excluded."
    )
    print(f"Frozen teacher: {teacher_source}")
    print(f"q33={q33:.8f}; q67={q67:.8f}")
    print("Group counts: " + json.dumps(metadata["group_counts"], ensure_ascii=False))
    print("Full/no_mid share one evaluation manifest: YES")
    print("Grouping depends only on source, GT target, and frozen teacher: YES")
    print_table("Endpoint metrics by deformation group", metrics)
    print_table("Unified deltas (positive = Full better)", delta)
    print_table("Spearman correlation: D_geo vs Full improvement", correlations)
    print_table("Objective trend assessment", assessment)
    print("\nHigh-deformation Full-better checks")
    high = assessment.set_index("metric")
    for metric in ("stroke_iou", "chamfer", "lpips", "tcr"):
        print(f"  {metric}: {bool(high.loc[metric, 'high_group_full_better'])}")
    print(f"\nCSV and metadata outputs: {output_dir}")


if __name__ == "__main__":
    main()
