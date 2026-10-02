"""
Unified inference script for font stylization model.

Supports multiple inference modes:
  - single: Single image pair inference
  - batch: Batch inference on dataset
  - interpolation: Generate intermediate frames
  - cross-script: Cross-lingual style transfer

Usage examples:
  # Single image
  python infer.py --mode single \
    --source source.png --style style.png --output result.png

  # Batch inference
  python infer.py --mode batch \
    --input_dir data/test --output_dir results

  # With interpolation
  python infer.py --mode interpolation \
    --input_dir data/test --output_dir results --alphas 0.25,0.5,0.75

  # Cross-script transfer
  python infer.py --mode cross-script \
    --input_dir chinese_data --skeleton_dir foreign_chars --output_dir results
"""

import os
import sys
import yaml
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from skimage import io
from skimage.color import gray2rgb
from tqdm import tqdm
from collections import defaultdict

import torch

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.transformer_kp_head import TransformerKPHead
from modules.style_encoder_adapter import build_style_encoder
from modules.font_student import FontStudent
from log.logger import Logger


# ============================================================================
# Shared utility functions
# ============================================================================

def _ensure_rgb_float(img_np: np.ndarray) -> np.ndarray:
    """Ensure image is RGB float format [0, 1]"""
    if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
        img_np = gray2rgb(img_np)
    if img_np.ndim == 3 and img_np.shape[2] == 4:
        img_np = img_np[..., :3]
    img_np = img_np.astype(np.float32)
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    return img_np


def load_image_as_tensor(path: str, device: torch.device, target_hw=None) -> torch.Tensor:
    """Load image and convert to tensor, optionally resize"""
    img = _ensure_rgb_float(io.imread(path))
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    if target_hw is not None:
        h, w = target_hw
        if img_t.shape[-2:] != (h, w):
            img_t = torch.nn.functional.interpolate(
                img_t, size=(h, w), mode='bilinear', align_corners=False
            )
    return img_t.to(device)


def save_tensor_image(img: torch.Tensor, path: str) -> None:
    """Save tensor image to file"""
    x = img.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    io.imsave(path, (x * 255.0).astype(np.uint8))


def list_image_files(root: str, use_test_split: bool = False, use_train_split: bool = False, use_all: bool = False) -> list:
    """List image files in directory, optionally filtering to test/train split."""
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = []
    root_path = os.path.normpath(root)
    
    has_flat_split = os.path.exists(os.path.join(root_path, 'train')) and os.path.exists(os.path.join(root_path, 'test'))
    has_nested_split = False
    
    if not has_flat_split:
        try:
            font_dirs = [d for d in os.listdir(root_path) 
                        if os.path.isdir(os.path.join(root_path, d)) and not d.startswith('.')]
            if len(font_dirs) > 0:
                sample_font_dir = os.path.join(root_path, font_dirs[0])
                has_nested_split = (
                    os.path.exists(os.path.join(sample_font_dir, 'train')) or 
                    os.path.exists(os.path.join(sample_font_dir, 'test'))
                )
        except (OSError, PermissionError):
            pass
    
    has_split = has_flat_split or has_nested_split
    
    for dp, dirnames, fns in os.walk(root):
        # Do not descend into notebook checkpoints or any other hidden
        # directory. Otherwise a path such as
        # <font>/test/<char>/.ipynb_checkpoints can be mistaken for an
        # independent character folder and consume one of the N style slots.
        dirnames[:] = [name for name in dirnames if not name.startswith('.')]

        if use_all:
            pass
        elif (use_test_split or use_train_split) and has_split:
            norm_dp = os.path.normpath(dp)
            parts = norm_dp.split(os.sep)
            if use_test_split:
                if 'train' in parts:
                    continue
                if 'test' not in parts:
                    continue
            elif use_train_split:
                if 'test' in parts:
                    continue
                if 'train' not in parts:
                    continue
        
        for fn in fns:
            if fn.startswith('.'):
                continue
            if os.path.splitext(fn.lower())[1] in exts:
                abs_p = os.path.join(dp, fn)
                rel_p = os.path.relpath(abs_p, root)
                # Defensive check for unusual walkers/filesystems: reject a
                # file whenever any relative path component is hidden.
                if any(part.startswith('.') for part in Path(rel_p).parts):
                    continue
                files.append(rel_p)
    
    files.sort()
    return files


def build_models(config, checkpoint_path, device, style_encoder_ckpt=None):
    """Build all required models and load checkpoint"""
    common_params = dict(config['model_params']['common_params'])
    gen_common_params = dict(common_params)
    gen_common_params.pop('style_dim', None)
    kp_common_params = dict(common_params)
    kp_common_params.pop('style_dim', None)
    
    generator = OcclusionAwareGenerator(
        **config['model_params']['generator_params'], 
        **gen_common_params
    ).to(device).eval()
    
    kp_detector = KPDetector(
        **config['model_params']['kp_detector_params'], 
        **kp_common_params
    ).to(device).eval()
    
    ckpt = Logger.load_checkpoint(
        checkpoint_path,
        generator=generator,
        discriminator=None,
        kp_detector=kp_detector,
        optimizer_generator=None,
        optimizer_discriminator=None,
        optimizer_kp_detector=None,
    )
    
    se_cfg = config.get('style_encoder', {})
    style_dim = se_cfg.get('style_dim', common_params.get('style_dim', 128))
    num_fonts = config.get('font_student', {}).get('num_fonts', 0)
    if num_fonts == 0:
        num_fonts = config.get('model_params', {}).get('discriminator_params', {}).get('num_fonts', 0)
    
    style_encoder = build_style_encoder(
        ckpt_path=style_encoder_ckpt or se_cfg.get('ckpt_path'),
        freeze=True,
        img_size=se_cfg.get('img_size', 64),
        style_dim=style_dim,
        backbone=se_cfg.get('backbone', 'vgg11'),
        normalize=se_cfg.get('normalize', False),
        num_fonts=num_fonts,
    ).to(device).eval()
    
    transformer_cfg = config.get('font_student', {}).get('transformer', {})
    kp_transformer = TransformerKPHead(
        num_kp=common_params.get('num_kp', 10),
        style_dim=style_dim,
        d_model=int(transformer_cfg.get('d_model', 256)),
        nhead=int(transformer_cfg.get('nhead', 4)),
        num_layers=int(transformer_cfg.get('num_layers', 4)),
        dim_feedforward=int(transformer_cfg.get('dim_feedforward', 1024)),
        dropout=float(transformer_cfg.get('dropout', 0.1)),
        max_disp=float(transformer_cfg.get('max_disp', 0.3)),
        use_delta_jac=bool(transformer_cfg.get('use_delta_jac', False)),
    ).to(device).eval()
    
    use_transformer = False
    if ckpt is not None and isinstance(ckpt, dict):
        if 'style_encoder' in ckpt:
            try:
                style_encoder.load_state_dict(ckpt['style_encoder'], strict=False)
            except Exception as e:
                print(f"Warning: Failed to load style_encoder: {e}")
        if 'kp_transformer' in ckpt:
            try:
                kp_transformer.load_state_dict(ckpt['kp_transformer'], strict=False)
                use_transformer = True
            except Exception as e:
                print(f"Warning: Failed to load kp_transformer: {e}")
    
    return generator, kp_detector, style_encoder, kp_transformer, use_transformer


def extract_style_vector(style_images, style_encoder, device, target_hw, silent=False):
    """Extract and average style vectors from multiple images"""
    if not isinstance(style_images, list):
        style_images = [style_images]
    
    style_vectors = []
    for img_path in style_images:
        try:
            img = load_image_as_tensor(img_path, device, target_hw)
            vec = style_encoder(img, sty=True)
            style_vectors.append(vec)
        except Exception as e:
            if not silent:
                print(f"Warning: Failed to load {img_path}: {e}")
    
    if len(style_vectors) == 0:
        return None
    
    style_tensor = torch.cat(style_vectors, dim=0)
    return style_tensor.mean(dim=0, keepdim=True)


# ============================================================================
# Mode 1: Single image inference
# ============================================================================

@torch.no_grad()
def infer_single(args, config, device):
    """Single image pair inference"""
    print("\n" + "="*60)
    print("Mode: Single Image Inference")
    print("="*60)
    
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source image not found: {args.source}")
    if not os.path.exists(args.style):
        raise FileNotFoundError(f"Style image not found: {args.style}")
    
    frame_shape = config.get('dataset_params', {}).get('frame_shape', [256, 256, 3])
    target_hw = (int(frame_shape[0]), int(frame_shape[1]))
    
    generator, kp_detector, style_encoder, kp_transformer, use_transformer = build_models(
        config, args.checkpoint, device, args.style_encoder_ckpt
    )
    
    print(f"Loading images...")
    source_img = load_image_as_tensor(args.source, device, target_hw)
    style_img = load_image_as_tensor(args.style, device, target_hw)
    
    print(f"Extracting keypoints...")
    kp_source = kp_detector(source_img)
    
    if use_transformer:
        print(f"Using Transformer for style transfer...")
        style_vec = style_encoder(style_img, sty=True)
        kp_driving = kp_transformer(kp_source, style_vec)
    else:
        print(f"Using direct keypoint extraction...")
        kp_driving = kp_detector(style_img)
    
    print(f"Generating stylized image...")
    output = generator(source_img, kp_source=kp_source, kp_driving=kp_driving)
    
    save_tensor_image(output['prediction'], args.output)
    print(f"\n✅ Done! Result saved to: {args.output}\n")


# ============================================================================
# Mode 2: Batch inference (with optional interpolation)
# ============================================================================

@torch.no_grad()
def infer_batch(args, config, device):
    """Batch inference on dataset with optional interpolation"""
    print("\n" + "="*60)
    print(f"Mode: {'Interpolation' if args.interpolate else 'Batch'} Inference")
    print("="*60)
    
    frame_shape = config.get('dataset_params', {}).get('frame_shape', [256, 256, 3])
    target_hw = (int(frame_shape[0]), int(frame_shape[1]))
    
    generator, kp_detector, style_encoder, kp_transformer, use_transformer = build_models(
        config, args.checkpoint, device, args.style_encoder_ckpt
    )
    
    use_test_split = (args.split == 'test')
    use_train_split = (args.split == 'train')
    use_all = (args.split == 'all')
    
    target_rel_all = list_image_files(args.input_dir, use_test_split, use_train_split, use_all)
    if len(target_rel_all) == 0:
        print(f'No images found in {args.input_dir}')
        return

    style_split = args.style_split or args.split
    style_rel_all = list_image_files(
        args.input_dir,
        use_test_split=(style_split == 'test'),
        use_train_split=(style_split == 'train'),
        use_all=(style_split == 'all'),
    )
    if len(style_rel_all) == 0:
        print(f'No style-reference images found in {args.input_dir} for split={style_split}')
        return
    
    folder_to_files = defaultdict(list)
    for rel in target_rel_all:
        folder_to_files[os.path.dirname(rel)].append(rel)

    style_folder_to_files = defaultdict(list)
    for rel in style_rel_all:
        style_folder_to_files[os.path.dirname(rel)].append(rel)
    
    font_to_char_folders = defaultdict(list)
    for subdir, files in style_folder_to_files.items():
        top = subdir.split(os.sep)[0] if len(subdir) > 0 else subdir
        if args.font_prefix and not top.startswith(args.font_prefix):
            continue
        font_id = top
        font_to_char_folders[font_id].append(subdir)
    
    # Build style vectors (average from first N chars)
    font_to_style_images = defaultdict(list)
    selected_style_folders = set()
    for font_id, char_folders in font_to_char_folders.items():
        char_folders.sort()
        for cf in char_folders[:args.num_style_chars]:
            files = style_folder_to_files.get(cf, [])
            if len(files) > 0:
                files.sort()
                font_to_style_images[font_id].append(os.path.join(args.input_dir, files[-1]))
                selected_style_folders.add(cf)
    
    # Process each character
    pairs = []
    for subdir, files in folder_to_files.items():
        if args.exclude_style_targets and subdir in selected_style_folders:
            continue
        top = subdir.split(os.sep)[0] if len(subdir) > 0 else subdir
        if args.font_prefix and not top.startswith(args.font_prefix):
            continue
        files.sort()
        if len(files) < 2:
            continue
        src_rel = files[0]
        gt_rel = files[-1]
        font_id = top
        base = os.path.splitext(os.path.basename(src_rel))[0]
        pairs.append((os.path.join(args.input_dir, src_rel), font_id, 
                     os.path.join(args.input_dir, gt_rel), subdir, base))
    
    if len(pairs) == 0:
        print("No valid pairs found")
        return
    
    print(f"Found {len(pairs)} samples to process")
    if args.interpolate:
        alphas = [float(x) for x in args.alphas.split(',') if x.strip()]
        alphas = [a for a in alphas if 0 < a < 1]
        print(f"Will generate interpolations at alphas: {alphas}")
    
    gt_dir = os.path.join(args.output_dir, "_gt_from_input")
    os.makedirs(gt_dir, exist_ok=True)
    
    for src_path, font_id, gt_path, subdir, base in tqdm(pairs, desc="Processing", disable=args.silent):
        src = load_image_as_tensor(src_path, device, target_hw)
        
        # Get style vector
        style_imgs = font_to_style_images.get(font_id, [])
        if len(style_imgs) > 0 and use_transformer:
            style_vec = extract_style_vector(style_imgs, style_encoder, device, target_hw, args.silent)
        else:
            style_img = load_image_as_tensor(gt_path, device, target_hw)
            style_vec = style_encoder(style_img, sty=True) if use_transformer else None
        
        kp_source = kp_detector(src)
        if use_transformer and style_vec is not None:
            kp_driving = kp_transformer(kp_source, style_vec)
        else:
            style_img = load_image_as_tensor(gt_path, device, target_hw)
            kp_driving = kp_detector(style_img)
        
        # Generate final result
        out = generator(src, kp_source=kp_source, kp_driving=kp_driving)
        out_path = os.path.join(args.output_dir, subdir, f"{base}_styled.png")
        save_tensor_image(out['prediction'], out_path)
        
        # Save GT
        try:
            gt_np = _ensure_rgb_float(io.imread(gt_path))
            if (gt_np.shape[0], gt_np.shape[1]) != target_hw:
                from skimage.transform import resize
                gt_np = resize(gt_np, target_hw, preserve_range=True, anti_aliasing=True).astype(np.float32)
            gt_save_path = os.path.join(gt_dir, subdir, f"{base}_gt.png")
            os.makedirs(os.path.dirname(gt_save_path), exist_ok=True)
            io.imsave(gt_save_path, (gt_np * 255.0).astype(np.uint8))
        except Exception as e:
            if not args.silent:
                print(f'Failed to save GT: {e}')
        
        # Generate interpolations if requested
        if args.interpolate:
            for alpha in alphas:
                alpha_t = torch.tensor([alpha], device=device, dtype=torch.float32)
                kp_mid = FontStudent._ode_interp_kp(kp_source, kp_driving, alpha_t, steps=args.ode_steps)
                mid_out = generator(src, kp_source=kp_source, kp_driving=kp_mid)
                mid_path = os.path.join(args.output_dir, subdir, f"{base}_mid{alpha:.2f}.png")
                save_tensor_image(mid_out['prediction'], mid_path)
    
    print(f"\n✅ Done! Results saved to: {args.output_dir}\n")


# ============================================================================
# Mode 3: Cross-script inference
# ============================================================================

@torch.no_grad()
def infer_cross_script(args, config, device):
    """Cross-script style transfer"""
    print("\n" + "="*60)
    print("Mode: Cross-Script Inference")
    print("="*60)
    
    frame_shape = config.get('dataset_params', {}).get('frame_shape', [256, 256, 3])
    target_hw = (int(frame_shape[0]), int(frame_shape[1]))
    
    generator, kp_detector, style_encoder, kp_transformer, use_transformer = build_models(
        config, args.checkpoint, device, args.style_encoder_ckpt
    )
    
    if not use_transformer:
        print("⚠ Warning: Transformer not loaded. Cross-script inference requires transformer!")
        return
    
    # Extract style vectors from source dataset
    print("\nStep 1: Extracting style vectors from source dataset...")
    use_test_split = (args.split == 'test')
    use_train_split = (args.split == 'train')
    use_all = (args.split == 'all')
    
    rel_all = list_image_files(args.input_dir, use_test_split, use_train_split, use_all)
    folder_to_files = defaultdict(list)
    for rel in rel_all:
        folder_to_files[os.path.dirname(rel)].append(rel)
    
    font_to_char_folders = defaultdict(list)
    for subdir, files in folder_to_files.items():
        top = subdir.split(os.sep)[0] if len(subdir) > 0 else subdir
        if args.font_prefix and not top.startswith(args.font_prefix):
            continue
        font_to_char_folders[top].append(subdir)
    
    font_to_style_vectors = {}
    for font_id, char_folders in font_to_char_folders.items():
        valid_folders = []
        for cf in char_folders:
            parts = cf.split(os.sep)
            if use_all or (use_test_split and ('test' in parts or 'train' not in parts)) or \
               (use_train_split and ('train' in parts or 'test' not in parts)):
                valid_folders.append(cf)
        
        valid_folders.sort()
        style_imgs = []
        for cf in valid_folders[:args.num_style_chars]:
            files = folder_to_files.get(cf, [])
            if len(files) > 0:
                files.sort()
                style_imgs.append(os.path.join(args.input_dir, files[-1]))
        
        if len(style_imgs) > 0:
            vec = extract_style_vector(style_imgs, style_encoder, device, target_hw, args.silent)
            if vec is not None:
                font_to_style_vectors[font_id] = vec
                print(f"  ✓ Font '{font_id}': {len(style_imgs)} chars")
    
    print(f"✓ Extracted {len(font_to_style_vectors)} font styles")
    
    # Collect skeleton images
    print("\nStep 2: Collecting skeleton images...")
    skeleton_dir = Path(args.skeleton_dir)
    if not skeleton_dir.exists():
        raise FileNotFoundError(f"Skeleton directory not found: {skeleton_dir}")
    
    language_to_images = {}
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    for lang_dir in skeleton_dir.iterdir():
        if not lang_dir.is_dir():
            continue
        images = [(f.stem, str(f)) for f in lang_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in exts]
        if images:
            images.sort()
            language_to_images[lang_dir.name] = images
            print(f"  ✓ {lang_dir.name}: {len(images)} images")
    
    # Generate results
    print("\nStep 3: Generating stylized results...")
    total = sum(len(imgs) * len(font_to_style_vectors) for imgs in language_to_images.values())
    pbar = tqdm(total=total, desc="Generating", disable=args.silent)
    
    for lang_name, images in language_to_images.items():
        for char_name, skel_path in images:
            try:
                skel_img = load_image_as_tensor(skel_path, device, target_hw)
                kp_source = kp_detector(skel_img)
            except Exception as e:
                if not args.silent:
                    print(f"⚠ Failed to load {skel_path}: {e}")
                pbar.update(len(font_to_style_vectors))
                continue
            
            for font_id, style_vec in font_to_style_vectors.items():
                try:
                    kp_driving = kp_transformer(kp_source, style_vec)
                    out = generator(skel_img, kp_source=kp_source, kp_driving=kp_driving)
                    out_path = os.path.join(args.output_dir, lang_name, font_id, f"{char_name}_styled.png")
                    save_tensor_image(out['prediction'], out_path)
                except Exception as e:
                    if not args.silent:
                        print(f"⚠ Failed: {lang_name}/{char_name}/{font_id}: {e}")
                pbar.update(1)
    
    pbar.close()
    print(f"\n✅ Done! Results saved to: {args.output_dir}\n")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = ArgumentParser(description="Unified font stylization inference")
    
    # Mode selection
    parser.add_argument('--mode', required=True, 
                       choices=['single', 'batch', 'interpolation', 'cross-script'],
                       help='Inference mode')
    
    # Common arguments
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--device', default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--style-encoder-ckpt', default=None, help='Optional style encoder checkpoint')
    parser.add_argument('--silent', action='store_true', help='Suppress detailed output')
    
    # Single mode arguments
    parser.add_argument('--source', help='Source image (for single mode)')
    parser.add_argument('--style', help='Style image (for single mode)')
    parser.add_argument('--output', help='Output image (for single mode)')
    
    # Batch/interpolation mode arguments
    parser.add_argument('--input_dir', help='Input directory (for batch/interpolation modes)')
    parser.add_argument('--output_dir', help='Output directory (for batch/interpolation/cross-script modes)')
    parser.add_argument('--split', default='test', choices=['test', 'train', 'all'],
                       help='Dataset split to use')
    parser.add_argument('--style-split', default=None, choices=['test', 'train', 'all'],
                       help='Split used only for style references; defaults to --split')
    parser.add_argument('--font-prefix', default=None, help='Filter fonts by prefix')
    parser.add_argument('--num-style-chars', type=int, default=8,
                       help='Number of characters for style averaging')
    parser.add_argument('--exclude-style-targets', action='store_true',
                       help='Exclude selected style-reference characters from generated/evaluated targets')
    
    # Interpolation mode arguments
    parser.add_argument('--interpolate', action='store_true',
                       help='Generate intermediate frames (for interpolation mode)')
    parser.add_argument('--alphas', default='0.25,0.5,0.75',
                       help='Comma-separated interpolation alphas')
    parser.add_argument('--ode-steps', type=int, default=8,
                       help='Kept for backward compatibility; not used by linear trajectory interpolation')
    
    # Cross-script mode arguments
    parser.add_argument('--skeleton_dir', help='Skeleton images directory (for cross-script mode)')
    
    args = parser.parse_args()
    
    # Validate mode-specific arguments
    if args.mode == 'single':
        if not all([args.source, args.style, args.output]):
            parser.error("single mode requires --source, --style, and --output")
    elif args.mode in ['batch', 'interpolation']:
        if not all([args.input_dir, args.output_dir]):
            parser.error(f"{args.mode} mode requires --input_dir and --output_dir")
        if args.mode == 'interpolation':
            args.interpolate = True
    elif args.mode == 'cross-script':
        if not all([args.input_dir, args.skeleton_dir, args.output_dir]):
            parser.error("cross-script mode requires --input_dir, --skeleton_dir, and --output_dir")
    
    # Load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = torch.device(args.device if ('cuda' in args.device and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")
    
    # Dispatch to appropriate handler
    if args.mode == 'single':
        infer_single(args, config, device)
    elif args.mode in ['batch', 'interpolation']:
        infer_batch(args, config, device)
    elif args.mode == 'cross-script':
        infer_cross_script(args, config, device)


if __name__ == '__main__':
    main()

