import math

import torch
from torch import nn


SUPPORTED_VARIANTS = {
    "gaussian_head",
    "laplace_head",
    "student_t_fixed_df8_head",
    "student_t_head",
}


def inverse_softplus(value):
    return math.log(math.expm1(value))


class ProbabilisticHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        patch_len,
        target_channels,
        variant,
        scale_floor=1e-4,
        initial_scale=0.5,
        initial_df=8.0,
    ):
        super().__init__()
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"invalid probabilistic variant: {variant}")
        if initial_df <= 2.01:
            raise ValueError("initial_df must be greater than 2.01")
        self.variant = variant
        self.scale_floor = scale_floor
        self.loc_projection = nn.Linear(hidden_dim, patch_len)
        self.scale_projection = nn.Linear(hidden_dim, patch_len)
        nn.init.zeros_(self.scale_projection.weight)
        nn.init.constant_(self.scale_projection.bias, inverse_softplus(initial_scale - scale_floor))
        if variant == "student_t_head":
            self.raw_df = nn.Parameter(
                torch.full((1, 1, target_channels), inverse_softplus(initial_df - 2.01))
            )
        else:
            self.register_parameter("raw_df", None)
        if variant == "student_t_fixed_df8_head":
            self.register_buffer("fixed_df", torch.full((1, 1, target_channels), 8.0))
        else:
            self.register_buffer("fixed_df", None)

    def initialize_location(self, official_projection):
        with torch.no_grad():
            self.loc_projection.weight.copy_(official_projection.weight)
            self.loc_projection.bias.copy_(official_projection.bias)

    def forward(self, hidden):
        loc = self.loc_projection(hidden)
        scale = torch.nn.functional.softplus(self.scale_projection(hidden)) + self.scale_floor
        if self.raw_df is not None:
            df = 2.01 + torch.nn.functional.softplus(self.raw_df)
        else:
            df = self.fixed_df
        return loc, scale, df
