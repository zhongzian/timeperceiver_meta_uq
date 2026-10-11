"""Capacity-matched Gaussian and fixed-df Student-t heads for PatchTST."""
from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.patchtst_backbone import PatchTSTBackbone


SUPPORTED_DISTRIBUTIONS = {"gaussian", "fixed_t8"}


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class ProbabilisticPatchTST(nn.Module):
    """Frozen PatchTST encoder plus matched location and scale projections."""

    def __init__(
        self,
        backbone: PatchTSTBackbone,
        distribution: str,
        initial_scale: float = 0.5,
        scale_floor: float = 1e-4,
        fixed_df: float = 8.0,
    ):
        super().__init__()
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"unsupported Gate 5b distribution: {distribution}")
        if initial_scale <= scale_floor:
            raise ValueError("initial_scale must exceed scale_floor")
        if distribution == "fixed_t8" and fixed_df != 8.0:
            raise ValueError("Gate 5b fixed Student-t requires df exactly 8.0")

        self.backbone = backbone
        self.distribution = distribution
        self.initial_scale = float(initial_scale)
        self.scale_floor = float(scale_floor)
        self.fixed_df = float(fixed_df)
        source = backbone.deterministic_projection
        self.loc_projection = nn.Linear(source.in_features, source.out_features)
        self.scale_projection = nn.Linear(source.in_features, source.out_features)
        self.loc_dropout = nn.Dropout(backbone.config["head_dropout"])
        self.scale_dropout = nn.Dropout(backbone.config["head_dropout"])
        self.initialize_from_deterministic_head()
        self.freeze_backbone()

    def initialize_from_deterministic_head(self) -> None:
        source = self.backbone.deterministic_projection
        self.loc_projection.load_state_dict(source.state_dict(), strict=True)
        nn.init.zeros_(self.scale_projection.weight)
        raw_initial = _inverse_softplus(self.initial_scale - self.scale_floor)
        nn.init.constant_(self.scale_projection.bias, raw_initial)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Freezing PatchTST also freezes BatchNorm/Dropout behavior and buffers.
        self.backbone.eval()
        return self

    def forward(self, batch_x: Tensor) -> Dict[str, Tensor | None]:
        hidden = self.backbone.encode(batch_x)
        raw_loc = self.loc_dropout(self.loc_projection(hidden))
        raw_scale = self.scale_dropout(self.scale_projection(hidden))
        loc = self.backbone.denormalize_location(raw_loc)
        scale = F.softplus(raw_scale).permute(0, 2, 1) + self.scale_floor
        df = torch.full_like(loc, self.fixed_df) if self.distribution == "fixed_t8" else None
        return {"loc": loc, "scale": scale, "df": df, "hidden": hidden}

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def backbone_trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.backbone.parameters() if parameter.requires_grad)

    def df_parameter_count(self) -> int:
        return sum(parameter.numel() for name, parameter in self.named_parameters() if "df" in name.lower())
