#!/usr/bin/env python3
"""One-factor-at-a-time multi-task loss-weight sensitivity experiment."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


METRICS = {
    "FID": "fid",
    "SSIM": "ssim",
    "LPIPS": "lpips",
    "Stroke_IoU": "stroke_iou",
    "Chamfer": "chamfer",
    "TCR": "topology_consistency_rate",
}
HIGHER = {"SSIM", "Stroke_IoU", "TCR"}
FIELDS = [
    "trial_id", "mid_mult", "perc_mult", "geo_mult", "adv_mult", "fm_mult",
    "lambda_mid", "lambda_perc", "lambda_geo", "lambda_adv", "lambda_fm",
    "FID", "SSIM", "LPIPS", "Stroke_IoU", "Chamfer", "TCR", "Q",
    "checkpoint", "pred_dir", "metric_dir",
]
MULT_NAMES = ("mid_mult", "perc_mult", "geo_mult", "adv_mult", "fm_mult")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/ablation/full.yaml")
    p.add_argument("--train-data", help="Fixed 50-font training root")
    p.add_argument("--ufuc-data", help="Fixed 10-font UFUC root")
    p.add_argument("--output-dir", default="loss_weight_sensitivity")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-ids", default="0")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument(
        "--trials", type=int, default=10,
        help="Number of ordered OFAT perturbations to run (normally all 10)",
    )
    p.add_argument("--budget-fraction", type=float, default=0.25)
    p.add_argument("--num-style-chars", type=int, default=8)
    p.add_argument("--visual-samples", type=int, default=8)
    p.add_argument("--default-checkpoint")
    p.add_argument("--default-summary")
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run Default + one trial for 50 steps on one UFUC font")
    p.add_argument("--smoke-steps", type=int, default=50)
    p.add_argument("--min-free-gb", type=float, default=25.0)
    return p.parse_args()


def uniform(values, name):
    vals = [float(v) for v in values]
    if not vals or any(not math.isclose(v, vals[0]) for v in vals[1:]):
        raise ValueError(f"{name} must contain equal layer weights, got {vals}")
    return vals[0], vals


def defaults_from(config):
    loss = config["train_params"]["loss_weights"]
    fs = config["font_student"]
    perc, perc_layers = uniform(loss["perceptual"], "perceptual")
    fm, fm_layers = uniform(loss["feature_matching"], "feature_matching")
    return {
        "lambda_mid": float(fs["lambda_mid"]),
        "lambda_perc": perc,
        "lambda_perc_layers": perc_layers,
        "lambda_geo": 1.0,
        "lambda_geo_native": False,
        "lambda_adv": float(loss["generator_gan"]),
        "lambda_cls": float(fs["font_cls_weight"]),
        "lambda_fm": fm,
        "lambda_fm_layers": fm_layers,
        "lambda_val": float(loss["kp_value_reg"]),
        "lambda_jac": float(loss["kp_jac_reg"]),
        "discriminator_gan_fixed": float(loss["discriminator_gan"]),
    }


def print_defaults(d):
    print("\nReal defaults read from the current Full config/code:")
    print(f"  lambda_mid  = {d['lambda_mid']}")
    print(f"  lambda_perc = {d['lambda_perc_layers']}")
    print("  lambda_geo  = no native scalar; group baseline = 1.0")
    print(f"  lambda_adv  = {d['lambda_adv']}")
    print(f"  lambda_cls  = {d['lambda_cls']} (fixed)")
    print(f"  lambda_fm   = {d['lambda_fm_layers']}")
    print(f"  lambda_val  = {d['lambda_val']} (fixed inside geometric group)")
    print(f"  lambda_jac  = {d['lambda_jac']} (fixed inside geometric group)\n")


def ofat_plan():
    """Return the ten controlled 0.5x/2.0x single-weight perturbations."""
    plan = []
    for multiplier_name in MULT_NAMES:
        weight_name = multiplier_name.removesuffix("_mult")
        for factor, suffix in ((0.5, "0p5x"), (2.0, "2x")):
            multipliers = {name: 1.0 for name in MULT_NAMES}
            multipliers[multiplier_name] = factor
            plan.append((f"{weight_name}_{suffix}", multipliers))
    return plan


def training_samples(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Training root not found: {root}")
    total = 0
    for font in root.iterdir():
        split = font / "train"
        if font.is_dir() and not font.name.startswith(".") and split.is_dir():
            total += sum(p.is_dir() and not p.name.startswith(".") for p in split.iterdir())
    if total <= 0:
        raise RuntimeError(f"No <font>/train/<character> samples found in {root}")
    return total


def free_space_check(path, minimum_gb):
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / (1024 ** 3)
    print(f"Free disk space: {free:.2f} GB")
    if free < minimum_gb:
        raise RuntimeError(
            f"Only {free:.2f} GB free; at least {minimum_gb:.2f} GB is required."
        )


def make_config(base, d, mult, train_root, max_steps):
    cfg = json.loads(json.dumps(base))
    cfg["dataset_params"]["root_dir"] = str(Path(train_root).resolve())
    train = cfg["train_params"]
    train["max_steps"] = int(max_steps)
    # Sensitivity runs keep only the final checkpoint and no epoch preview PNG.
    train["checkpoint_freq"] = int(train["num_epochs"]) + 1
    train["skip_train_visualizations"] = True
    loss = train["loss_weights"]
    cfg["font_student"]["lambda_mid"] = d["lambda_mid"] * mult["mid_mult"]
    cfg["font_student"]["font_cls_weight"] = d["lambda_cls"]
    loss["perceptual"] = [v * mult["perc_mult"] for v in d["lambda_perc_layers"]]
    loss["geo_group"] = mult["geo_mult"]
    loss["kp_value_reg"] = d["lambda_val"]
    loss["kp_jac_reg"] = d["lambda_jac"]
    loss["generator_gan"] = d["lambda_adv"] * mult["adv_mult"]
    loss["discriminator_gan"] = d["discriminator_gan_fixed"]
    loss["feature_matching"] = [v * mult["fm_mult"] for v in d["lambda_fm_layers"]]
    return cfg


def run_command(command, cwd, log_file):
    print("\n$ " + " ".join(map(str, command)), flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            list(map(str, command)), cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def latest_checkpoint(log_root, label):
    runs = sorted(log_root.glob(f"{label} *"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No run directory for {label} in {log_root}")
    files = sorted(runs[-1].glob("*-checkpoint.pth.tar"))
    if not files:
        raise FileNotFoundError(f"No checkpoint in {runs[-1]}")
    return files[-1].resolve()


def read_metrics(path):
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {name: float(raw[key]) for name, key in METRICS.items()}


def run_one(project, output, label, cfg, checkpoint, args, font_prefix=None):
    cfg_dir = output / "configs"
    log_root = output / "logs"
    pred = output / "predictions" / label
    metrics = output / "metrics" / label
    cmd_log = output / "command_logs" / f"{label}.log"
    for folder in (cfg_dir, log_root, pred.parent, metrics, cmd_log.parent):
        folder.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{label}.yaml"
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)

    if checkpoint is None:
        complete = (
            cmd_log.is_file()
            and "Reached exact training budget:" in cmd_log.read_text(
                encoding="utf-8", errors="replace"
            )
        )
        if complete:
            checkpoint = latest_checkpoint(log_root, label)
            print(f"Reusing completed {label} checkpoint: {checkpoint}")
        else:
            run_command([
                sys.executable, "-u", "run.py", "--config", cfg_path,
                "--log_dir", log_root, "--device_ids", args.device_ids,
            ], project, cmd_log)
            checkpoint = latest_checkpoint(log_root, label)
    else:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    summary = metrics / "summary.json"
    have_predictions = pred.is_dir() and any(pred.rglob("*_styled.png"))
    if not have_predictions:
        infer_cmd = [
            sys.executable, "-u", "infer.py", "--mode", "batch",
            "--config", cfg_path, "--checkpoint", checkpoint,
            "--input_dir", Path(args.ufuc_data).resolve(), "--output_dir", pred,
            "--split", "test", "--style-split", "test",
            "--num-style-chars", str(args.num_style_chars),
            "--exclude-style-targets", "--device", args.device,
        ]
        if font_prefix:
            infer_cmd += ["--font-prefix", font_prefix]
        run_command(infer_cmd, project, cmd_log)
    if not summary.is_file():
        eval_cmd = [
            sys.executable, "-u", "evaluation/evaluate_ufuc.py",
            "--pred_dir", pred, "--output_dir", metrics,
            "--variant", label,
            "--device", args.device, "--batch_size", str(args.eval_batch_size),
        ]
        if args.smoke_test:
            eval_cmd += ["--fid_dims", "64"]
        run_command(eval_cmd, project, cmd_log)
    return read_metrics(summary), checkpoint, pred.resolve(), metrics.resolve()


def q_score(row, default):
    ratios = [
        row["SSIM"] / default["SSIM"],
        row["Stroke_IoU"] / default["Stroke_IoU"],
        row["TCR"] / default["TCR"],
        default["FID"] / row["FID"],
        default["LPIPS"] / row["LPIPS"],
        default["Chamfer"] / row["Chamfer"],
    ]
    return sum(ratios) / 6.0


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), sd, float(values.min()), float(values.max())


def write_summary(path, rows):
    default, trials = rows[0], rows[1:]
    mean, sd, minimum, maximum = stats([float(r["Q"]) for r in trials])
    best = max(trials, key=lambda r: float(r["Q"]))
    worst = min(trials, key=lambda r: float(r["Q"]))
    lines = [
        "Multi-task loss-weight sensitivity summary",
        "==========================================",
        "Q is auxiliary; the six original metrics remain primary.", "",
        f"Default Q: {float(default['Q']):.6f}",
        f"Perturbation Q mean: {mean:.6f}",
        f"Perturbation Q sample std: {sd:.6f}",
        f"Perturbation Q min: {minimum:.6f}",
        f"Perturbation Q max: {maximum:.6f}",
        f"Default vs best gap: {(float(best['Q']) - 1) * 100:+.3f}%",
        f"Default vs mean gap: {(mean - 1) * 100:+.3f}%",
        f"Worst decline from Default: {(1 - float(worst['Q'])) * 100:+.3f}%", "",
    ]
    for metric in METRICS:
        vals = [float(r[metric]) for r in trials]
        m, s, lo, hi = stats(vals)
        rel = [(v / float(default[metric]) - 1) * 100 for v in vals]
        direction = "higher is better" if metric in HIGHER else "lower is better"
        lines.append(
            f"{metric}: mean={m:.8f}, std={s:.8f}, min={lo:.8f}, max={hi:.8f}; "
            f"mean relative change={sum(rel)/len(rel):+.3f}%, "
            f"max absolute relative change={max(map(abs, rel)):.3f}% ({direction})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_scores(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Default"] + [str(r["trial_id"]) for r in rows[1:]]
    values = [float(r["Q"]) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(range(len(values)), values, marker="o")
    ax.axhline(1, color="black", linestyle="--", label="Default Q = 1.0")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Normalized overall score Q")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def copy_visuals(output, rows, count):
    default, trials = rows[0], rows[1:]
    ordered = sorted(trials, key=lambda r: float(r["Q"]))
    chosen = {
        "visual_default": default,
        "visual_best": ordered[-1],
        "visual_median": ordered[len(ordered) // 2],
        "visual_worst": ordered[0],
    }
    root = Path(default["pred_dir"])
    samples = sorted(
        p for p in root.rglob("*_styled.png")
        if "_gt_from_input" not in p.parts and not any(x.startswith(".") for x in p.parts)
    )[:count]
    if not samples:
        raise RuntimeError(f"No qualitative samples found in {root}")
    relative = [p.relative_to(root) for p in samples]
    (output / "visual_samples.txt").write_text(
        "\n".join(str(p).replace("\\", "/") for p in relative) + "\n",
        encoding="utf-8",
    )
    for dirname, row in chosen.items():
        for rel in relative:
            src = Path(row["pred_dir"]) / rel
            dst = output / dirname / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def first_ufuc_font(root):
    fonts = sorted(
        p.name for p in Path(root).iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p / "test").is_dir()
    )
    if not fonts:
        raise RuntimeError(f"No <font>/test directory under {root}")
    return fonts[0]


def main():
    args = arguments()
    project = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project / cfg_path
    with cfg_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    d = defaults_from(base)
    print_defaults(d)
    if args.inspect_only:
        return
    if not args.train_data or not args.ufuc_data:
        raise ValueError("--train-data and --ufuc-data are required")

    output = Path(args.output_dir).resolve()
    font_prefix = None
    if args.smoke_test:
        output = Path(str(output) + "_smoke")
        args.trials = 1
        font_prefix = first_ufuc_font(args.ufuc_data)
        print(f"SMOKE TEST uses UFUC font: {font_prefix}")
    free_space_check(output, 5.0 if args.smoke_test else args.min_free_gb)

    samples = training_samples(args.train_data)
    batch = int(base["train_params"]["batch_size"])
    full_steps = (samples // batch) * int(base["train_params"]["num_epochs"])
    max_steps = args.smoke_steps if args.smoke_test else int(round(full_steps * args.budget_fraction))
    print(f"Training samples={samples}, batch_size={batch}, full_steps={full_steps}")
    print(f"Training steps per configuration={max_steps}")
    metadata = dict(
        d,
        design="one-factor-at-a-time",
        perturbation_factors=[0.5, 2.0],
        varied_weights=list(MULT_NAMES),
        samples=samples,
        batch_size=batch,
        full_steps=full_steps,
        max_steps=max_steps,
        smoke_test=args.smoke_test,
    )
    (output / "weight_mapping.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    neutral = {name: 1.0 for name in MULT_NAMES}
    default_cfg = make_config(base, d, neutral, args.train_data, max_steps)
    if args.default_summary:
        if not args.default_checkpoint:
            raise ValueError("--default-summary requires --default-checkpoint")
        target = output / "metrics" / "default"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(args.default_summary), target / "summary.json")
    metrics, ckpt, pred, metric_dir = run_one(
        project, output, "default", default_cfg, args.default_checkpoint, args, font_prefix
    )
    default = {
        "trial_id": "Default", **neutral,
        "lambda_mid": d["lambda_mid"], "lambda_perc": d["lambda_perc"],
        "lambda_geo": 1.0, "lambda_adv": d["lambda_adv"], "lambda_fm": d["lambda_fm"],
        **metrics, "Q": 1.0, "checkpoint": str(ckpt), "pred_dir": str(pred),
        "metric_dir": str(metric_dir),
    }
    rows = [default]
    write_csv(output / "loss_weight_sensitivity.csv", rows)

    plan = ofat_plan()
    if not 1 <= args.trials <= len(plan):
        raise ValueError(f"--trials must be between 1 and {len(plan)}")
    plan = plan[:args.trials]
    print(
        "Sensitivity design: one factor at a time; each selected weight is "
        "evaluated at 0.5x and 2.0x while all other weights remain at 1.0x."
    )

    for index, (label, mult) in enumerate(plan, start=1):
        print(
            f"\n=== Configuration {index + 1}/{len(plan) + 1}: "
            f"{label} ==="
        )
        cfg = make_config(base, d, mult, args.train_data, max_steps)
        metrics, ckpt, pred, metric_dir = run_one(
            project, output, label, cfg, None, args, font_prefix
        )
        row = {
            "trial_id": label, **mult,
            "lambda_mid": d["lambda_mid"] * mult["mid_mult"],
            "lambda_perc": d["lambda_perc"] * mult["perc_mult"],
            "lambda_geo": mult["geo_mult"],
            "lambda_adv": d["lambda_adv"] * mult["adv_mult"],
            "lambda_fm": d["lambda_fm"] * mult["fm_mult"],
            **metrics, "checkpoint": str(ckpt), "pred_dir": str(pred),
            "metric_dir": str(metric_dir),
        }
        row["Q"] = q_score(row, default)
        rows.append(row)
        write_csv(output / "loss_weight_sensitivity.csv", rows)

    write_summary(output / "loss_weight_sensitivity_summary.txt", rows)
    plot_scores(output / "sensitivity_score.png", rows)
    copy_visuals(output, rows, args.visual_samples)
    print(f"\nCompleted: {output}")


if __name__ == "__main__":
    main()
