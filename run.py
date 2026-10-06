import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime

from dataset.frames_dataset import FramesDataset

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.discriminator import MultiScaleDiscriminator

import torch

from train import train


if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config.get('env', {}).items():
        if v is None:
            continue
        os.environ[str(k)] = str(v)
        print(f"Set env {k}={v}")

    resume_from_checkpoint = opt.checkpoint is not None
    checkpoint_to_use = opt.checkpoint
    if checkpoint_to_use is None and 'teacher_checkpoint' in config:
        checkpoint_to_use = config['teacher_checkpoint']
        print(f"Using teacher checkpoint from config: {checkpoint_to_use}")

    config['_resume_from_checkpoint'] = bool(resume_from_checkpoint)

    log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
    log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
    print(f"Log directory: {log_dir}")

    gen_common_params = dict(config['model_params']['common_params'])
    gen_common_params.pop('style_dim', None)
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **gen_common_params)

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    use_discriminator = bool(config.get('train_params', {}).get('use_discriminator', True))
    discriminator = None
    if use_discriminator:
        disc_params = dict(config['model_params']['discriminator_params'])
        if 'conditional' not in disc_params:
            disc_params['conditional'] = True
        if 'num_fonts' not in disc_params:
            if bool(disc_params.get('conditional', False)):
                raise ValueError(
                    "Conditional discriminator enabled by default, but `model_params.discriminator_params.num_fonts` "
                    "is missing; please set it explicitly (e.g. num_fonts: 300)."
                )

        discriminator = MultiScaleDiscriminator(**disc_params, **gen_common_params)
        if torch.cuda.is_available():
            discriminator.to(opt.device_ids[0])
        if opt.verbose:
            print(discriminator)
    else:
        loss_weights = config.get('train_params', {}).get('loss_weights', {})
        feature_matching = loss_weights.get('feature_matching', [])
        if (
            loss_weights.get('generator_gan', 0) != 0
            or loss_weights.get('discriminator_gan', 0) != 0
            or sum(feature_matching) != 0
        ):
            raise ValueError(
                "train_params.use_discriminator=false requires generator_gan=0, "
                "discriminator_gan=0, and all feature_matching weights=0."
            )
        print("Discriminator disabled: GAN and feature-matching branches will be skipped.")

    kp_common_params = dict(config['model_params']['common_params'])
    kp_common_params.pop('style_dim', None)
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **kp_common_params)

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])

    if opt.verbose:
        print(kp_detector)
        
    dataset_params = dict(config['dataset_params'])
    mode = config.get('mode', 'train')
    if mode == 'student_temporal' and 'use_mid_frame' not in dataset_params:
        dataset_params['use_mid_frame'] = True
    dataset = FramesDataset(is_train=(opt.mode == 'train'), **dataset_params)
    config['dataset_params'] = dataset_params
    
    # Check dataset size
    print(f"Dataset loaded: {len(dataset)} samples")
    if len(dataset) == 0:
        print(f"\n❌ ERROR: No data found in '{dataset_params['root_dir']}'")
        print(f"Please check:")
        print(f"  1. Dataset path is correct in config file")
        print(f"  2. Dataset directory exists and contains data")
        print(f"  3. Data follows the required structure (see README.md)")
        sys.exit(1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    effective_config_path = os.path.join(log_dir, os.path.basename(opt.config))
    with open(effective_config_path, 'w') as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False, allow_unicode=True)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, kp_detector, checkpoint_to_use, log_dir, dataset, opt.device_ids)
