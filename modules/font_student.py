from typing import Dict, Any

import torch
from torch import nn

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.transformer_kp_head import TransformerKPHead


class FontStudent(nn.Module):
    """
    Transformer predicts driving KP from source KP + style vector; optional mid-frame supervision.
    """

    def __init__(
        self,
        kp_extractor: KPDetector,
        generator: OcclusionAwareGenerator,
        style_encoder: nn.Module,
        kp_transformer: TransformerKPHead,
        train_params: Dict[str, Any],
        num_kp: int = 10,
        use_mid_supervision: bool = False,
        lambda_mid: float = 0.0,
        mid_supervision_type: str = 'keypoint',
        mid_kp_value_weight: float = 1.0,
        mid_kp_jacobian_weight: float = 0.5,
        detach_kp_for_generator: bool = False,
    ) -> None:
        super().__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.style_encoder = style_encoder
        self.kp_transformer = kp_transformer
        self.train_params = train_params
        self.num_kp = num_kp
        self.use_mid_supervision = use_mid_supervision
        self.lambda_mid = float(lambda_mid)
        self.mid_supervision_type = str(mid_supervision_type).lower()
        if self.mid_supervision_type != 'keypoint':
            raise ValueError(
                "Only keypoint-level intermediate supervision is supported; "
                f"got {mid_supervision_type!r}"
            )
        self.mid_kp_value_weight = float(mid_kp_value_weight)
        self.mid_kp_jacobian_weight = float(mid_kp_jacobian_weight)
        self.detach_kp_for_generator = bool(detach_kp_for_generator)

    @staticmethod
    def _ode_interp_kp(
        kp_start: Dict[str, torch.Tensor],
        kp_end: Dict[str, torch.Tensor],
        alpha: torch.Tensor,
        steps: int = 8,
    ) -> Dict[str, torch.Tensor]:
        kp_mid: Dict[str, torch.Tensor] = {}
        # `steps` is kept for backward compatibility with older call sites.
        for key, v_start in kp_start.items():
            if key not in kp_end:
                continue
            v_end = kp_end[key]
            a = alpha
            while a.dim() < v_start.dim():
                a = a.unsqueeze(-1)
            kp_mid[key] = v_start + a * (v_end - v_start)
        return kp_mid

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        assert 'source' in x and 'driving' in x, "FontStudent requires 'source' and 'driving'"

        # The pretrained keypoint detector is a frozen teacher.  Its outputs
        # are fixed supervision/conditioning signals, so no autograd graph is
        # needed through the detector.
        kp_mid_true = None
        with torch.no_grad():
            kp_source_true = self.kp_extractor(x['source'])
            kp_driving_true = self.kp_extractor(x['driving'])
            if (
                self.use_mid_supervision
                and ('mid' in x)
                and ('alpha' in x)
                and self.lambda_mid > 0
            ):
                # Extract this target before the endpoint Generator graph is
                # built so the teacher's temporary activations do not overlap
                # with the peak trainable-branch memory.
                kp_mid_true = self.kp_extractor(x['mid'])

        style_img = x['style'] if 'style' in x else x['driving']
        style_logits = None
        enc_out = self.style_encoder(style_img, sty=False)  # returns cont or {'cont','disc'}
        if isinstance(enc_out, dict):
            style_global = enc_out['cont']
            style_logits = enc_out.get('disc')
        else:
            style_global = enc_out

        kp_hat = self.kp_transformer(
            kp_source_true,
            style_global,
            value_scale=x.get('kp_value_scale', None),
            jac_scale=x.get('kp_jac_scale', None),
        )
        kp_driving_pred = kp_hat
        if self.detach_kp_for_generator:
            kp_driving_pred_for_gen = {
                k: (v.detach() if torch.is_tensor(v) else v) for k, v in kp_driving_pred.items()
            }
        else:
            kp_driving_pred_for_gen = kp_driving_pred

        out: Dict[str, Any] = self.generator(
            x['source'], kp_source=kp_source_true, kp_driving=kp_driving_pred_for_gen
        )

        if (
            self.use_mid_supervision
            and ('mid' in x)
            and ('alpha' in x)
            and self.lambda_mid > 0
        ):
            alpha_tensor = x['alpha']
            if not torch.is_tensor(alpha_tensor):
                alpha_tensor = torch.tensor(
                    alpha_tensor, device=x['source'].device, dtype=x['source'].dtype
                )
            alpha_tensor = alpha_tensor.to(x['source'].device, dtype=x['source'].dtype)
            alpha_tensor = alpha_tensor.view(alpha_tensor.shape[0], 1)

            kp_mid = self._ode_interp_kp(kp_source_true, kp_driving_pred, alpha_tensor)
            out['kp_mid'] = kp_mid

            # Convert the Log-Domain middle image into a fixed teacher
            # keypoint target. No intermediate Generator or DenseMotion pass
            # is constructed, so this objective updates only the Style
            # Encoder and TransformerKPHead.
            if kp_mid_true is None:
                raise RuntimeError("Missing frozen-teacher middle keypoints")

            value_loss = torch.abs(
                kp_mid['value'] - kp_mid_true['value']
            ).mean()
            loss_mid = self.mid_kp_value_weight * value_loss

            if 'jacobian' in kp_mid and 'jacobian' in kp_mid_true:
                jacobian_loss = torch.abs(
                    kp_mid['jacobian'] - kp_mid_true['jacobian']
                ).mean()
                loss_mid = (
                    loss_mid
                    + self.mid_kp_jacobian_weight * jacobian_loss
                )

            out['mid_keypoint'] = self.lambda_mid * loss_mid

        out_base = {
            'kp_source': kp_source_true,
            'kp_driving_true': kp_driving_true,
            'kp_driving_mix': kp_driving_pred,
            'kp_driving': kp_driving_pred,
            'kp_hat': kp_hat,
            'style_code': style_global,
            'lambda': torch.tensor(
                1.0,
                device=out['prediction'].device if 'prediction' in out else x['source'].device,
                dtype=x['source'].dtype if torch.is_tensor(x['source']) else None,
            ),
        }
        if style_logits is not None:
            out_base['style_logits'] = style_logits

        out.update(out_base)
        return out
