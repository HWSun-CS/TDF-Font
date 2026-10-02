import os
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# VGG architecture configurations (VGG11 and VGG16)
_VGG_CFG: Dict[str, Any] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


def _make_layers(cfg, use_batch_norm: bool = True) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if use_batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


class FontStyleEncoder(nn.Module):
    """
    Font style encoder (based on VGG11), outputs global style vector (and optional font classification logits).
    """

    def __init__(
        self,
        img_size: int = 64,
        style_dim: int = 128,
        max_pools: int = 3,
        backbone: str = 'vgg11',
        num_fonts: int = 0,
    ):
        super().__init__()
        cfg = _VGG_CFG['vgg11']
        self.features = _make_layers(cfg, True)
        self.cont = nn.Linear(512, style_dim)
        self.has_disc = int(num_fonts) > 0
        disc_dim = max(1, int(num_fonts))
        self.disc = nn.Linear(512, disc_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, sty: bool = False):
        feat = self.features(x)  # (B,512,H/32,W/32)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))
        flat = pooled.view(pooled.size(0), -1)
        cont = self.cont(flat)  # (B, style_dim)
        if sty or (not getattr(self, 'has_disc', False)):
            return cont
        disc = self.disc(flat)
        return {'cont': cont, 'disc': disc}

    def moco(self, x: torch.Tensor):
        return self.forward(x, sty=True)

    def iic(self, x: torch.Tensor):
        feat = self.features(x)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))
        flat = pooled.view(pooled.size(0), -1)
        return self.disc(flat)


class StyleEncoderAdapter(nn.Module):
    """
    Wrapper for style encoder: returns global style vector (and optional font classification logits).
    """

    def __init__(
        self,
        img_size: int = 64,
        style_dim: int = 128,
        max_pools: int = 3,
        backbone: str = 'vgg11',
        ckpt_path: Optional[str] = None,
        freeze: bool = False,
        normalize: bool = False,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        num_fonts: int = 0,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.normalize = normalize
        if mean is None:
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        if std is None:
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        self.backbone = FontStyleEncoder(
            img_size=img_size,
            style_dim=style_dim,
            max_pools=max_pools,
            backbone=backbone,
            num_fonts=num_fonts,
        )
        self.has_disc = bool(getattr(self.backbone, 'has_disc', False))

        if ckpt_path is not None and os.path.isfile(ckpt_path):
            self._load_checkpoint(ckpt_path)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def _load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt

        cleaned = {}
        for k, v in state.items():
            k_new = k
            if k_new.startswith('module.'):
                k_new = k_new[len('module.'):]
            if k_new.startswith('backbone.'):
                k_new = k_new[len('backbone.'):]
            cleaned[k_new] = v

        self.backbone.load_state_dict(cleaned, strict=False)

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return (x - self.mean) / self.std

    def forward(self, style_image: torch.Tensor, sty: bool = False):
        x = style_image
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x = self._maybe_normalize(x)
        return self.backbone(x, sty=sty)


def build_style_encoder(
    ckpt_path: Optional[str] = None,
    freeze: bool = False,
    img_size: int = 64,
    style_dim: int = 128,
    max_pools: int = 3,
    normalize: bool = False,
    backbone: str = 'vgg11',
    num_fonts: int = 0,
) -> StyleEncoderAdapter:
    return StyleEncoderAdapter(
        img_size=img_size,
        style_dim=style_dim,
        max_pools=max_pools,
        backbone=backbone,
        ckpt_path=ckpt_path,
        freeze=freeze,
        normalize=normalize,
        num_fonts=num_fonts,
    )
