from typing import Optional

import torch
from torch import nn


class TransformerKPHead(nn.Module):
    """
    Transformer predicts driving KP from source KP + style vector.
    Supports curriculum scaling for value/jacobian deltas.
    """

    def __init__(
        self,
        num_kp: int,
        style_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_disp: float = 0.3,
        use_delta_jac: bool = False,
    ) -> None:
        super().__init__()
        self.num_kp = num_kp
        self.max_disp = float(max_disp)
        self.use_delta_jac = bool(use_delta_jac)

        # Each KP token: [x, y] + jacobian(4) -> 6 dims
        self.kp_proj = nn.Linear(6, d_model)
        self.style_proj = nn.Linear(style_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.randn(1, num_kp + 1, d_model) * 0.02)

        # Predict delta value, then clamp by tanh
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
        )

        # Optional: predict delta jacobian
        self.jac_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

        # Runtime curriculum scales (can be updated during training)
        self.register_buffer("curr_value_scale", torch.tensor(1.0))
        self.register_buffer("curr_jac_scale", torch.tensor(1.0))

    def set_scales(self, value_scale: Optional[float] = None, jac_scale: Optional[float] = None) -> None:
        """Update stored curriculum scales in-place."""
        if value_scale is not None:
            self.curr_value_scale.data.fill_(float(value_scale))
        if jac_scale is not None:
            self.curr_jac_scale.data.fill_(float(jac_scale))

    def forward(
        self,
        kp_source: dict,
        style_vec: torch.Tensor,
        value_scale: Optional[float] = None,
        jac_scale: Optional[float] = None,
    ) -> dict:
        """
        kp_source: {'value': (B,K,2), 'jacobian': (B,K,2,2)}
        style_vec: (B, style_dim)
        value_scale / jac_scale: optional runtime scale factors (for curriculum)
        return: {'value': (B,K,2), 'jacobian': (B,K,2,2)}
        """
        value = kp_source["value"]
        jac = kp_source.get("jacobian", None)
        if jac is None:
            b, k, _ = value.shape
            eye = torch.eye(2, device=value.device, dtype=value.dtype).view(1, 1, 2, 2)
            jac = eye.repeat(b, k, 1, 1)
        else:
            b, k, _, _ = jac.shape
        jac_flat = jac.view(b, k, 4)

        # Tokens
        kp_feat = torch.cat([value, jac_flat], dim=-1)  # (B,K,6)
        kp_tokens = self.kp_proj(kp_feat)  # (B,K,d_model)

        style_token = self.style_proj(style_vec).unsqueeze(1)  # (B,1,d_model)

        tokens = torch.cat([style_token, kp_tokens], dim=1)  # (B,K+1,d_model)
        pos = self.pos_embed[:, : tokens.shape[1], :]
        tokens = tokens + pos

        enc = self.encoder(tokens)  # (B,K+1,d_model)
        kp_out = enc[:, 1:, :]  # drop style token

        # Delta value with clamp; scale can be updated per-epoch/step
        scale_v = float(value_scale) if value_scale is not None else float(self.curr_value_scale)
        delta_value = self.value_head(kp_out)
        delta_value = scale_v * self.max_disp * torch.tanh(delta_value)
        pred_value = value + delta_value

        # Jacobian: default copy source; optionally learn delta
        if self.use_delta_jac:
            scale_j = float(jac_scale) if jac_scale is not None else float(self.curr_jac_scale)
            delta_jac = self.jac_head(kp_out).view(b, k, 2, 2)
            delta_jac = scale_j * 0.1 * torch.tanh(delta_jac)
            pred_jac = jac + delta_jac
        else:
            pred_jac = jac

        return {
            "value": pred_value,
            "jacobian": pred_jac,
        }
