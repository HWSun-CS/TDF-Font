# Font Stylization Model

A deep learning model for font style transfer driven by keypoints.

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Format

### Training Data Structure

```
dataset/
├── font_001/
│   ├── train/
│   │   ├── char_A/
│   │   │   ├── 00.png       # source (skeleton image)
│   │   │   ├── 01.png       # Log-Domain intermediate frame 1
│   │   │   ├── ...
│   │   │   ├── 10.png       # Log-Domain intermediate frame 10
│   │   │   └── 11.png       # target (fully stylized image)
│   │   ├── char_B/
│   │   └── ...
│   └── test/
│       ├── char_X/
│       └── ...
├── font_002/
└── ...
```

**Requirements:**

- The default training protocol uses 12 frames: source + 10 Log-Domain intermediate frames + target.
- Intermediate frames do not provide direct pixel-loss targets for generated images; a frozen teacher keypoint detector extracts coordinate and Jacobian targets from them.
- Preprocess input images to 256 x 256. Supported image formats include PNG, JPG, and BMP.

### Inference Data Structure

Use the same structure as the training data, with only two frames required per character:

```
test_data/
├── font_001/
│   └── test/
│       ├── A/
│       │   ├── 01.png       # source image
│       │   └── 02.png       # ground truth
│       └── B/
│           ├── 01.png
│           └── 02.png
└── ...
```

---

## Training

### 1. Configure the Dataset

Edit the dataset path in `config/default.yaml`:

```yaml
dataset_params:
  root_dir: /path/to/your/dataset    # Replace with your dataset path
```

### 2. Start Training

```bash
python run.py --config config/default.yaml --log_dir logs
```

**Notes:**

- Ensure that the configured dataset directory exists and contains the data.
- The dataset must contain at least `batch_size` samples.
- Reduce `batch_size` in the configuration for smaller datasets.

---

## Inference

Inference supports four modes:

### 1. Single Image Inference

```bash
python infer.py --mode single \
  --config config/default.yaml \
  --checkpoint logs/model.pth.tar \
  --source source.png \
  --style style.png \
  --output result.png
```

### 2. Batch Inference

```bash
python infer.py --mode batch \
  --config config/default.yaml \
  --checkpoint logs/model.pth.tar \
  --input_dir data/test_set \
  --output_dir results
```

Batch inference randomly selects eight reference characters per font by default and averages their style vectors for all characters of that font.
Reference characters are also generated. Omit `--exclude-style-targets` to include them. References are sampled again on each run.
The selected reference image paths are saved in `style_references.json` in the output directory.

- `--num-style-chars 4`: Randomly select four reference characters per font.
- `--style-selection random`: Select references randomly (default).
- `--style-selection sorted`: Select the first N characters in sorted order.
- `--style-split test`: Set the reference character split; defaults to `--split`.

The current `infer.py` does not support `--batch-size` or `--num-workers`.
Batch mode runs the model one image at a time. The evaluation script has its own `--batch_size` parameter.

For example, randomly select four references and generate all test characters:

```bash
python -u infer.py --mode batch \
  --config config/default.yaml \
  --checkpoint "/path/to/model.pth.tar" \
  --input_dir "/path/to/unseen_font_data" \
  --output_dir results/UFUC_style4_all \
  --split test --style-split test \
  --num-style-chars 4 --style-selection random \
  --device cuda:0
```

### 3. Interpolation Inference (Intermediate Frames)

```bash
python infer.py --mode interpolation \
  --config config/default.yaml \
  --checkpoint logs/model.pth.tar \
  --input_dir data/test_set \
  --output_dir results \
  --alphas 0.25,0.5,0.75
```

### 4. Cross-Script Inference

```bash
python infer.py --mode cross-script \
  --config config/default.yaml \
  --checkpoint logs/model.pth.tar \
  --input_dir chinese_fonts \
  --skeleton_dir foreign_chars \
  --output_dir results
```

---

## 📝 Common Arguments

| Argument | Description |
|------|------|
| `--config` | Configuration file path |
| `--checkpoint` | Model checkpoint path |
| `--split` | Dataset split (`test` / `train` / `all`) |
| `--device` | Device (`cuda:0` or `cpu`) |

---

## Multi-task loss-weight sensitivity

The robustness runner reuses the normal training, batch-inference, and UFUC
evaluation entry points. It applies a controlled one-factor-at-a-time design:
each of the five main task-level weights is set
to `0.5x` and `2.0x` while all other weights remain at `1.0x`. Together with
the Default baseline, this gives 11 configurations. Every configuration is
trained for exactly 25% of the Full step budget.

Inspect the loss mapping and real defaults without starting training:

```bash
python -u experiments/run_loss_weight_sensitivity.py --inspect-only
```

Run the complete experiment on the server:

```bash
cd /path/to/TDF-Font
python -u experiments/run_loss_weight_sensitivity.py \
  --config config/ablation/full.yaml \
  --train-data /path/to/50_font_training_data \
  --ufuc-data /path/to/10_font_ufuc_data \
  --output-dir /path/to/loss_weight_sensitivity \
  --device cuda:0 \
  --device-ids 0 \
  --eval-batch-size 8
```

An existing Default checkpoint may be reused only when it was trained with
the same exact step budget. Pass it with `--default-checkpoint`; optionally
pass its matching UFUC `summary.json` with `--default-summary`. If these are
omitted, the runner trains and evaluates the Default baseline itself.

The current project has no native scalar named `lambda_geo`. For this
experiment, `geo_mult` is an optional group multiplier (neutral value 1.0)
over the complete keypoint geometric block; the native `kp_value_reg` and
`kp_jac_reg` values and their ratio remain unchanged. Existing training
configs are unaffected because the new multiplier defaults to 1.0.
