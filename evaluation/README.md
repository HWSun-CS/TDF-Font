# UFUC ablation evaluation

This directory evaluates the Full model and seven ablated variants on the
fixed UFUC test split.

## Metrics

- SSIM: higher is better.
- RMSE: lower is better.
- LPIPS (AlexNet): lower is better.
- FID (InceptionV3 pool3 by default): lower is better.
- Stroke-IoU: higher is better.
- Symmetric Chamfer distance: lower is better. Distances are normalized by the
  image diagonal.
- Connected-component error (`beta0_error`): lower is better. It measures the
  absolute difference in the number of 8-connected foreground components.
- Hole-count error (`beta1_error`): lower is better.
- Euler-characteristic error (`euler_error`): lower is better, where
  `Euler = beta0 - beta1`.
- Topology consistency rate (`topology_consistency_rate`): higher is better. A
  glyph counts as consistent only when both beta0 and beta1 exactly match its
  paired ground truth.

Stroke masks use a threshold of 0.5. The foreground polarity is inferred from
the image border by default, which supports both dark-on-light and
light-on-dark glyph images. Before topology measurement, foreground components
and enclosed holes smaller than 4 pixels are removed as raster noise from both
prediction and ground truth. Keep `--stroke_threshold`, `--foreground`, and
`--topology_min_area` identical for every variant and report these settings.

## 1. Install metric dependencies

```bash
pip install lpips pytorch-fid scipy
```

The first LPIPS/FID run may download pretrained weights. Use the same cached
weights and software environment for all runs.

## 2. Generate UFUC predictions

Use the same fixed UFUC input directory and the same eight style references for
all variants. Example:

```bash
python -u infer.py \
  --mode batch \
  --config config/ablation/full.yaml \
  --checkpoint /path/to/full.pth.tar \
  --input_dir /path/to/ufuc_data \
  --output_dir results/full \
  --split test \
  --style-split test \
  --num-style-chars 8 \
  --exclude-style-targets \
  --device cuda:0
```

This protocol selects the same first eight sorted characters from each unseen
test font as style references. `--exclude-style-targets` removes these eight
reference characters from the generated/evaluated targets, preventing them
from leaking into the quantitative results. The remaining test characters are
used for UFUC evaluation. Keep the selected references and sorting convention
identical for every variant.

`infer.py` saves paired ground truth under
`results/full/_gt_from_input`, so no separate ground-truth argument
is normally required.

## 3. Evaluate one variant

```bash
python -u evaluation/evaluate_ufuc.py \
  --pred_dir results/full \
  --output_dir metrics/ufuc/full \
  --variant full \
  --device cuda:0 \
  --batch_size 8 \
  --topology_min_area 4
```

Outputs:

- `per_image.csv`: paired image-quality and topology measurements for every
  glyph, including predicted/GT beta0, beta1 and Euler values.
- `summary.json`: run-level metric averages, topology consistency rate and FID.

Repeat this step for the Full model and all seven ablated variants:

- `full`
- `no_mid`
- `no_perceptual`
- `no_adversarial`
- `no_feature_matching`
- `no_style_condition`
- `no_multiscale`
- `no_discriminator`

## Trajectory geometric consistency

`evaluate_trajectory_consistency.py` projects the source, ten Log-Domain
intermediate frames and target into the same frozen-keypoint space. It compares
the observed intermediate keypoints with both the teacher endpoint-linear path
and the normal TDF-Font predicted path. The frozen detector provides a common
measurement space in this analysis, while the intermediate Log-Domain images
define the reference trajectory.

The strict UFUC protocol uses the first eight sorted test characters of each
unseen font as style references and excludes those characters from the target
set. With 10 fonts and 108 test characters per font, the expected evaluated
count is therefore `10 * (108 - 8) = 1000`, not 1080.

```bash
python -u evaluation/evaluate_trajectory_consistency.py \
  --config config/default.yaml \
  --checkpoint /path/to/main_final.pth.tar \
  --input-dir /path/to/ufuc_data \
  --protocol UFUC \
  --style-split test \
  --output-dir metrics/trajectory/main_model \
  --device cuda:0 \
  --batch-size 8 \
  --num-style-chars 8 \
  --expected-sequences 1000
```

For the seen-font/unseen-character (SFUC) trajectory experiment used here, use
the same few-shot protocol: take the first eight sorted `test` characters of
each seen font as style references and exclude those support characters from
the query targets. With 300 fonts and 108 test characters per font, this gives
`300 * (108 - 8) = 30000` evaluated sequences:

```bash
python -u evaluation/evaluate_trajectory_consistency.py \
  --config config/default.yaml \
  --checkpoint /path/to/main_final.pth.tar \
  --input-dir /path/to/seen_font_data \
  --protocol SFUC \
  --style-split test \
  --num-style-chars 8 \
  --output-dir metrics/trajectory/main_model_sfuc \
  --device cuda:0 \
  --batch-size 2
```

The target root must contain the 12-frame Log-Domain sequences for the test
characters. The eight selected support characters are not included in the
reported query-set metrics.

If a separate original FOMM teacher checkpoint is available, add:

```bash
  --teacher-checkpoint /path/to/teacher.pth.tar
```

Without that option, the script uses the frozen `kp_detector` stored in the
TDF-Font checkpoint. Add `--compute-jacobian` only after the coordinate results
and debug overlays have been checked.

The main reviewer-facing trajectory experiment must use `config/default.yaml`
and the final main-model checkpoint. Do not compare that 300-font main model
directly with the 50-font `no_mid` ablation. If a causal Full-versus-no-mid
trajectory comparison is desired, evaluate the matched 50-font
`config/ablation/full.yaml` and `config/ablation/no_mid.yaml` checkpoints as a
separate supplementary comparison.

In addition to raw, by-time and global CSV files, the trajectory-consistency
script writes `trajectory_errors_by_difficulty.csv`. Its deformation groups
are based on teacher endpoint displacement, and its nonlinearity groups are
based on the Log-Domain keypoint path-length/chord-length ratio. These groups
directly cover large and highly nonlinear transformations in the reviewer
response.

## Hard-case analysis for intermediate supervision

`evaluate_mid_hard_cases.py` compares matched Full and w/o Mid checkpoints on
the same immutable sample/style-reference manifest. Difficulty groups are
defined independently of either evaluated model using frozen-teacher endpoint
displacement and Log-Domain trajectory nonlinearity. The script reports
endpoint FID, SSIM, LPIPS, Stroke-IoU, Chamfer distance and TCR for each
difficulty group; it does not use a training-loss log as an evaluation metric.

UFUC example:

```bash
python -u evaluation/evaluate_mid_hard_cases.py \
  --full-config config/ablation/full.yaml \
  --no-mid-config config/ablation/no_mid.yaml \
  --full-checkpoint /path/to/full.pth.tar \
  --no-mid-checkpoint /path/to/no_mid.pth.tar \
  --input-dir /path/to/12_frame_ufuc_data \
  --output-dir /path/to/metrics/mid_hard_cases_ufuc \
  --protocol UFUC \
  --target-split test \
  --device cuda:0 \
  --batch-size 4 \
  --num-style-chars 8 \
  --expected-samples 1000 \
  --expected-frames 12
```

Use checkpoints trained on the same 50-font subset. For
an UFSC run, set `--protocol UFSC`, `--target-split train`, and the matching
expected sample count. Reuse the generated `evaluation_manifest.json` when
comparing additional matched checkpoints.
