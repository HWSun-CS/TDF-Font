# 字体风格化生成模型

基于关键点驱动的字体风格迁移深度学习模型。

---

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

---

## 📁 数据集格式

### 训练数据结构

```
dataset/
├── font_001/
│   ├── train/
│   │   ├── 字符_A/
│   │   │   ├── 00.png       # source（骨架图）
│   │   │   ├── 01.png       # Log-Domain 中间帧 1
│   │   │   ├── ...
│   │   │   ├── 10.png       # Log-Domain 中间帧 10
│   │   │   └── 11.png       # target（完全风格化）
│   │   ├── 字符_B/
│   │   └── ...
│   └── test/
│       ├── 字符_X/
│       └── ...
├── font_002/
└── ...
```

**要求**：
- 默认训练协议使用12帧：source + 10个Log-Domain中间帧 + target。
- 中间帧不会直接与生成图像计算像素损失；冻结的教师关键点检测器从其中提取坐标和Jacobian目标。
- 输入图像应预处理为256×256；图片格式为PNG/JPG/BMP。

### 推理数据结构

**和训练数据相同**，但每个字符只需2帧即可：

```
test_data/
├── font_001/
│   └── test/
│       ├── A/
│       │   ├── 01.png       # 源图
│       │   └── 02.png       # 真值
│       └── B/
│           ├── 01.png
│           └── 02.png
└── ...
```

---

## 🚀 训练

### 1. 修改配置文件

编辑 `config/default.yaml`，修改数据集路径：

```yaml
dataset_params:
  root_dir: /path/to/your/dataset    # 改成你的数据集路径
```

### 2. 开始训练

```bash
python run.py --config config/default.yaml --log_dir logs
```

**注意**：
- 确保 `data/` 文件夹存在且包含数据
- 数据集样本数应大于等于 `batch_size`（默认21）
- 如果数据集较小，需要在配置文件中降低 `batch_size`

---

## 🎯 推理

推理支持4种模式：

### 1. 单图推理

```bash
python infer.py --mode single \
  --config config/default.yaml \
  --checkpoint logs/model.pth.tar \
  --source source.png \
  --style style.png \
  --output result.png
```

### 2. 批量推理

```bash
python infer.py --mode batch \
  --config config/default.yaml \
  --checkpoint logs/model.pth.tar \
  --input_dir data/test_set \
  --output_dir results
```

### 3. 插值推理（生成中间帧）

```bash
python infer.py --mode interpolation \
  --config config/default.yaml \
  --checkpoint logs/model.pth.tar \
  --input_dir data/test_set \
  --output_dir results \
  --alphas 0.25,0.5,0.75
```

### 4. 跨脚本推理（跨语言）

```bash
python infer.py --mode cross-script \
  --config config/default.yaml \
  --checkpoint logs/model.pth.tar \
  --input_dir chinese_fonts \
  --skeleton_dir foreign_chars \
  --output_dir results
```

---

## 📝 常用参数

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径 |
| `--checkpoint` | 模型检查点路径 |
| `--split` | 数据分割（`test`/`train`/`all`） |
| `--device` | 设备（`cuda:0`或`cpu`） |

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
