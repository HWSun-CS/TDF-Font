from tqdm import trange
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch import amp
from torch.cuda.amp import GradScaler
import sys
import os

from log.logger import Logger
from modules.model import StudentTeacherFontModel, GeneratorFullModel, DiscriminatorFullModel, detach_kp
from modules.style_encoder_adapter import build_style_encoder
from modules.font_student import FontStudent
from modules.transformer_kp_head import TransformerKPHead


def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']
    mode = config.get('mode', 'train')
    use_student = mode in ['student', 'hybrid', 'student_distill', 'student_temporal']
    use_teacher_generator = bool(train_params.get('use_teacher_generator', True))
    detach_kp_for_generator = bool(train_params.get('detach_kp_for_generator', False))
    resume_from_checkpoint = bool(config.get('_resume_from_checkpoint', False))

    optimizer_generator = None
    optimizer_discriminator = None
    optimizer_kp = None
    centers = None
    if checkpoint is not None:
        ckpt_dict = Logger.load_checkpoint(checkpoint, generator, None, kp_detector,
                                   None,
                                   None,
                                   None)
        if resume_from_checkpoint:
            try:
                start_epoch = int(ckpt_dict.get('epoch', -1)) + 1 if isinstance(ckpt_dict, dict) else 0
            except Exception:
                start_epoch = 0
        else:
            start_epoch = 0
    else:
        start_epoch = 0
        ckpt_dict = None

    # The pretrained keypoint detector is a fixed teacher for both the main
    # experiment and every ablation.  Load its checkpoint first, then freeze it
    # completely so student losses cannot create or accumulate teacher grads.
    kp_detector.requires_grad_(False)
    kp_detector.eval()

    scheduler_generator = None
    scheduler_discriminator = None

    batch_size = train_params['batch_size']
    num_samples = len(dataset)
    
    # Check if dataset is large enough for the batch size
    if num_samples < batch_size:
        print(f"\n⚠️  WARNING: Dataset has {num_samples} samples but batch_size={batch_size}")
        print(f"   Setting drop_last=False to avoid empty dataloader")
        drop_last = False
    else:
        drop_last = True
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_params.get('num_workers', 6),
        drop_last=drop_last,
    )
    
    # Verify dataloader is not empty
    if len(dataloader) == 0:
        print(f"\n❌ ERROR: Dataloader is empty!")
        print(f"   Dataset has {num_samples} samples, batch_size={batch_size}, drop_last={drop_last}")
        print(f"   Please reduce batch_size in config or add more data")
        sys.exit(1)
    
    print(f"Dataloader ready: {len(dataloader)} batches/epoch, {num_samples} total samples")
    
    kp_scale_schedule = train_params.get('kp_scale_schedule', None)
    mem_log_interval = int(train_params.get('memory_log_interval', 0) or 0)
    # Optional exact step budget used by controlled experiments.  Normal
    # training is unchanged when max_steps is absent (the default).
    max_steps = train_params.get('max_steps', None)
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError(f"train_params.max_steps must be positive, got {max_steps}")

    def _compute_kp_scales(curr_epoch: int):
        """
        Curriculum: gradually relax KP/Jac clipping.
        Schedule config example (train_params.kp_scale_schedule):
            value_start: 0.3, value_end: 1.0
            jac_start: 0.1, jac_end: 1.0
            warmup_epochs: 50
        """
        if not kp_scale_schedule:
            return None, None
        vs = float(kp_scale_schedule.get('value_start', 1.0))
        ve = float(kp_scale_schedule.get('value_end', vs))
        js = float(kp_scale_schedule.get('jac_start', vs))
        je = float(kp_scale_schedule.get('jac_end', js))
        warm = int(kp_scale_schedule.get('warmup_epochs', 0))
        progress = 1.0
        if warm > 0:
            progress = min(1.0, (curr_epoch + 1) / float(warm))
        value_scale = vs + (ve - vs) * progress
        jac_scale = js + (je - js) * progress
        return value_scale, jac_scale
    device = torch.device(f'cuda:{device_ids[0]}') if (torch.cuda.is_available() and len(device_ids) > 0) else torch.device('cpu')
    use_amp = torch.cuda.is_available()
    scaler_gen = GradScaler(enabled=use_amp)
    scaler_disc = GradScaler(enabled=use_amp)

    if use_student:
        common = config['model_params']['common_params']
        se_cfg = config.get('style_encoder', {})
        font_student_cfg = config.get('font_student', {})
        style_dim = se_cfg.get('style_dim', 128)

        num_fonts = int(font_student_cfg.get('num_fonts', 0)) if font_student_cfg.get('num_fonts', 0) else 0
        style_encoder = build_style_encoder(
            ckpt_path=se_cfg.get('ckpt_path'),
            freeze=se_cfg.get('freeze', False),
            img_size=se_cfg.get('img_size', 64),
            style_dim=style_dim,
            backbone=se_cfg.get('backbone', 'vgg11'),
            normalize=se_cfg.get('normalize', False),
            num_fonts=num_fonts,
        )

        font_student_cfg = config.get('font_student', {})
        transformer_cfg = font_student_cfg.get('transformer', {})
        kp_transformer = TransformerKPHead(
            num_kp=common.get('num_kp', 10),
            style_dim=style_dim,
            d_model=int(transformer_cfg.get('d_model', 256)),
            nhead=int(transformer_cfg.get('nhead', 4)),
            num_layers=int(transformer_cfg.get('num_layers', 4)),
            dim_feedforward=int(transformer_cfg.get('dim_feedforward', 1024)),
            dropout=float(transformer_cfg.get('dropout', 0.1)),
            max_disp=float(transformer_cfg.get('max_disp', 0.3)),
            use_delta_jac=bool(transformer_cfg.get('use_delta_jac', False)),
        )

        num_fonts = int(font_student_cfg.get('num_fonts', 0)) if font_student_cfg.get('num_fonts', 0) else 0
        font_cls_weight = float(font_student_cfg.get('font_cls_weight', 0.0))
        center_weight = 0.0
        encoder_has_disc = bool(getattr(style_encoder, 'has_disc', False) or getattr(getattr(style_encoder, 'backbone', None), 'has_disc', False))
        style_classifier = None
        if (not encoder_has_disc) and num_fonts > 0 and font_cls_weight > 0:
            style_classifier = torch.nn.Linear(style_dim, num_fonts)
        # An explicit config value must override the mode default so the
        # no-mid ablation genuinely disables the temporal supervision branch.
        temporal_student = bool(
            font_student_cfg.get('use_mid_supervision', mode == 'student_temporal')
        )
        lambda_mid = font_student_cfg.get('lambda_mid', font_student_cfg.get('mid_loss_weight', train_params.get('loss_weights', {}).get('mid_keypoint', 0.0)))
        mid_supervision_type = str(
            font_student_cfg.get('mid_supervision_type', 'keypoint')
        ).lower()
        mid_kp_value_weight = float(
            font_student_cfg.get('mid_kp_value_weight', 1.0)
        )
        mid_kp_jacobian_weight = float(
            font_student_cfg.get('mid_kp_jacobian_weight', 0.5)
        )
        print(
            "Intermediate supervision: "
            f"enabled={temporal_student}, type={mid_supervision_type}, "
            f"lambda={float(lambda_mid):g}, "
            f"value_weight={mid_kp_value_weight:g}, "
            f"jacobian_weight={mid_kp_jacobian_weight:g}"
        )

        if ckpt_dict is not None and isinstance(ckpt_dict, dict):
            if 'style_encoder' in ckpt_dict:
                try:
                    style_encoder.load_state_dict(ckpt_dict['style_encoder'], strict=False)
                    print("Loaded style_encoder from checkpoint")
                except Exception as e:
                    print(f"Failed to load style_encoder: {e}")
            if 'kp_transformer' in ckpt_dict:
                try:
                    kp_transformer.load_state_dict(ckpt_dict['kp_transformer'], strict=False)
                    print("Loaded kp_transformer from checkpoint")
                except Exception as e:
                    print(f"Failed to load kp_transformer: {e}")
            if style_classifier is not None and 'style_classifier' in ckpt_dict:
                try:
                    style_classifier.load_state_dict(ckpt_dict['style_classifier'], strict=False)
                    print("Loaded style_classifier from checkpoint")
                except Exception as e:
                    print(f"Failed to load style_classifier: {e}")

        font_student = FontStudent(
            kp_extractor=kp_detector,
            generator=generator,
            style_encoder=style_encoder,
            kp_transformer=kp_transformer,
            train_params=train_params,
            num_kp=common.get('num_kp', 10),
            use_mid_supervision=temporal_student,
            lambda_mid=lambda_mid,
            mid_supervision_type=mid_supervision_type,
            mid_kp_value_weight=mid_kp_value_weight,
            mid_kp_jacobian_weight=mid_kp_jacobian_weight,
            detach_kp_for_generator=detach_kp_for_generator,
        )

        teacher_gen = None
        if use_teacher_generator:
            teacher_gen = generator
            if config.get('teacher_copy_freeze', False):
                import copy
                teacher_gen = copy.deepcopy(generator)
                for p in teacher_gen.parameters():
                    p.requires_grad = False
                teacher_gen.eval()
                if torch.cuda.is_available():
                    teacher_gen.to(device)
        student_full = StudentTeacherFontModel(kp_detector, teacher_gen, font_student, train_params)

        lr_student = train_params.get('lr_generator', 2e-4)
        lr_se = train_params.get('lr_style_encoder', lr_student)
        lr_head = train_params.get('lr_style_kp_head', lr_student)
        se_params = [p for p in style_encoder.parameters() if p.requires_grad]
        head_params = [p for p in kp_transformer.parameters() if p.requires_grad]
        kp_param_groups = []
        if len(se_params) > 0:
            kp_param_groups.append({'params': se_params, 'lr': lr_se})
        if len(head_params) > 0:
            kp_param_groups.append({'params': head_params, 'lr': lr_head})
        if style_classifier is not None:
            cls_params = [p for p in style_classifier.parameters() if p.requires_grad]
            if len(cls_params) > 0:
                lr_cls = train_params.get('lr_style_classifier', lr_head)
                kp_param_groups.append({'params': cls_params, 'lr': lr_cls})
        centers = None
        if len(kp_param_groups) > 0:
            optimizer_kp = torch.optim.Adam(kp_param_groups, betas=(0.5, 0.999))

        fine_tune_gen = config.get('fine_tune_generator', False)
        gen_lr = train_params.get('lr_generator', 2e-4)
        if fine_tune_gen:
            gen_lr = train_params.get('lr_generator_finetune', gen_lr)
        gen_params = [p for p in generator.parameters() if p.requires_grad]
        if len(gen_params) > 0:
            optimizer_generator = torch.optim.Adam([{'params': gen_params, 'lr': gen_lr}], betas=(0.5, 0.999))

        if discriminator is not None:
            disc_params = [p for p in discriminator.parameters() if p.requires_grad]
            if len(disc_params) > 0:
                lr_disc = train_params.get('lr_discriminator', gen_lr)
                optimizer_discriminator = torch.optim.Adam([{'params': disc_params, 'lr': lr_disc}], betas=(0.5, 0.999))

        milestones = train_params.get('epoch_milestones', [0])
        if optimizer_generator is not None:
            scheduler_generator = MultiStepLR(optimizer_generator, milestones, gamma=0.1, last_epoch=-1)
        if optimizer_discriminator is not None:
            scheduler_discriminator = MultiStepLR(optimizer_discriminator, milestones, gamma=0.1, last_epoch=-1)

        if torch.cuda.is_available():
            student_full = student_full.to(device)
            if style_classifier is not None:
                style_classifier = style_classifier.to(device)
        discriminator_full = None
        if discriminator is not None:
            discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    else:
        generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
        discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    vis_subdir = 'train-vis'
    if use_student:
        mode_name = 'font_student'
        vis_subdir = f"train-vis-{mode_name}"
    skip_full_checkpoint = bool(config.get('skip_full_checkpoint', False))

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq'], vis_subdir=vis_subdir) as logger:
        logger.skip_full_checkpoint = skip_full_checkpoint
        logger.skip_train_visualizations = bool(train_params.get('skip_train_visualizations', False))
        global_step = start_epoch * len(dataloader)
        stop_at_step_budget = False
        for epoch in trange(start_epoch, train_params['num_epochs']):
            # Keep the teacher in inference mode even if a parent module is
            # switched to train mode in a future training-path change.
            kp_detector.eval()
            try:
                if use_student:
                    target = student_full.module if hasattr(student_full, 'module') else student_full
                    if hasattr(target, 'epoch'):
                        target.epoch = epoch
            except Exception:
                pass
            value_scale_curr, jac_scale_curr = _compute_kp_scales(epoch)
            if use_student:
                kp_transformer.set_scales(value_scale=value_scale_curr, jac_scale=jac_scale_curr)
            for step, x in enumerate(dataloader):
                if torch.cuda.is_available() and mem_log_interval > 0:
                    torch.cuda.reset_peak_memory_stats(device)
                if torch.cuda.is_available():
                    for key in ('source', 'driving', 'style', 'mid'):
                        if key in x:
                            x[key] = x[key].to(device, non_blocking=True)
                    if 'alpha' in x:
                        x['alpha'] = x['alpha'].to(device, non_blocking=True, dtype=torch.float32)
                    if 'font_id' in x:
                        x['font_id'] = x['font_id'].to(device, non_blocking=True)
                if use_student:
                    if value_scale_curr is not None:
                        x['kp_value_scale'] = value_scale_curr
                    if jac_scale_curr is not None:
                        x['kp_jac_scale'] = jac_scale_curr
                if use_student:
                    if optimizer_kp is not None:
                        optimizer_kp.zero_grad()
                    if optimizer_generator is not None:
                        optimizer_generator.zero_grad()
                    with amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                        losses_generator, generated = student_full(x)
                        if font_cls_weight > 0 and 'font_id' in x and 'style_code' in generated:
                            font_labels = x['font_id']
                            if not torch.is_tensor(font_labels):
                                font_labels = torch.tensor(font_labels, device=device) if torch.cuda.is_available() else torch.tensor(font_labels)
                            font_labels = font_labels.to(device).long() - 1
                            font_labels = torch.clamp_min(font_labels, 0)
                            logits = None
                            if 'style_logits' in generated:
                                logits = generated['style_logits']
                            elif style_classifier is not None:
                                logits = style_classifier(generated['style_code'])
                            if logits is not None:
                                loss_font = F.cross_entropy(logits, font_labels) * font_cls_weight
                                losses_generator['font_cls'] = loss_font
                        font_labels_cond = None
                        if 'font_id' in x:
                            font_labels_cond = x['font_id']
                            if not torch.is_tensor(font_labels_cond):
                                font_labels_cond = torch.tensor(font_labels_cond, device=device) if torch.cuda.is_available() else torch.tensor(font_labels_cond)
                            font_labels_cond = font_labels_cond.to(device).long()
                            font_labels_cond = torch.clamp_min(font_labels_cond - 1, 0)
                        if (discriminator_full is not None) and ('prediction' in generated):
                            pyramid_real = discriminator_full.pyramid(x['driving'])
                            pyramid_generated = discriminator_full.pyramid(generated['prediction'])
                            kp_for_disc = generated.get('kp_driving') or generated.get('kp_hat')
                            loss_w = train_params.get('loss_weights', {})
                            if loss_w.get('generator_gan', 0) != 0:
                                discriminator_maps_generated = discriminator_full.discriminator(pyramid_generated, kp=detach_kp(kp_for_disc), y=font_labels_cond)
                                discriminator_maps_real = discriminator_full.discriminator(pyramid_real, kp=detach_kp(kp_for_disc), y=font_labels_cond)
                                value_total = 0
                                for scale in discriminator_full.scales:
                                    key = 'prediction_map_%s' % scale
                                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                                    value_total += loss_w['generator_gan'] * value
                                losses_generator['gen_gan'] = value_total
                            fm_w = loss_w.get('feature_matching', [])
                            if sum(fm_w) != 0:
                                discriminator_maps_generated = discriminator_full.discriminator(
                                    pyramid_generated, kp=detach_kp(kp_for_disc), y=font_labels_cond
                                )
                                discriminator_maps_real = discriminator_full.discriminator(
                                    pyramid_real, kp=detach_kp(kp_for_disc), y=font_labels_cond
                                )
                                value_total = 0
                                for scale in discriminator_full.scales:
                                    key = 'feature_maps_%s' % scale
                                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                                        if i >= len(fm_w) or fm_w[i] == 0:
                                            continue
                                        value = torch.abs(a - b).mean()
                                        value_total += fm_w[i] * value
                                losses_generator['feature_matching'] = value_total
                else:
                    raise SystemExit('Only student training is supported in the slimmed setup.')

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                if use_amp:
                    scaler_gen.scale(loss).backward()
                    if use_student:
                        if optimizer_kp is not None:
                            scaler_gen.step(optimizer_kp)
                        if optimizer_generator is not None:
                            scaler_gen.step(optimizer_generator)
                    scaler_gen.update()
                else:
                    loss.backward()
                    if use_student:
                        if optimizer_kp is not None:
                            optimizer_kp.step()
                        if optimizer_generator is not None:
                            optimizer_generator.step()
                losses_discriminator = {}
                if (discriminator_full is not None) and train_params['loss_weights'].get('discriminator_gan', 0) != 0:
                    optimizer_discriminator.zero_grad()
                    with amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                        losses_discriminator = discriminator_full(x, generated)
                        loss_values = [val.mean() for val in losses_discriminator.values()]
                        loss_d = sum(loss_values)

                    if use_amp:
                        scaler_disc.scale(loss_d).backward()
                        scaler_disc.step(optimizer_discriminator)
                        scaler_disc.update()
                    else:
                        loss_d.backward()
                        optimizer_discriminator.step()

                losses_generator.update(losses_discriminator)
                losses = {key: float(value.mean().detach().cpu()) for key, value in losses_generator.items()}
                if torch.cuda.is_available() and mem_log_interval > 0 and ((step + 1) % mem_log_interval == 0):
                    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                    cur_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    cur_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                    print(f"[mem] epoch={epoch} step={step} "
                          f"peak_alloc={peak_alloc:.2f}GB peak_reserved={peak_reserved:.2f}GB "
                          f"cur_alloc={cur_alloc:.2f}GB cur_reserved={cur_reserved:.2f}GB")
                logger.log_iter(losses=losses)
                global_step += 1
                if max_steps is not None and global_step >= max_steps:
                    stop_at_step_budget = True
                    print(f"Reached exact training budget: {global_step}/{max_steps} steps")
                    break

            # Log epoch after all batches are processed
            if 'x' in locals() and 'generated' in locals():
                logger_models = {'kp_detector': kp_detector}
                if use_student:
                    logger_models.update({
                        'style_encoder': style_encoder,
                        'kp_transformer': kp_transformer
                    })
                    logger_models['generator'] = generator
                    if discriminator is not None:
                        logger_models['discriminator'] = discriminator
                    if 'style_classifier' in locals() and style_classifier is not None:
                        logger_models['style_classifier'] = style_classifier
                logger_state = {'optimizer_generator': optimizer_generator,
                                'optimizer_discriminator': optimizer_discriminator,
                                'optimizer_kp': optimizer_kp}
                logger.log_epoch(epoch, logger_models, inp=x, out=generated)

            if scheduler_generator is not None:
                scheduler_generator.step()
            if scheduler_discriminator is not None:
                scheduler_discriminator.step()

            if stop_at_step_budget:
                break

        if skip_full_checkpoint:
            logger.models = None


if __name__ == "__main__":
    sys.exit("Use run.py to launch training.")
