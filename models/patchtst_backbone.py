"""Project adapter around the official supervised PatchTST architecture.

The official source remains untouched.  This adapter exposes the flattened
channel representation immediately before PatchTST's deterministic forecast
projection so deterministic and probabilistic heads share the same features.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
PATCHTST_SUPERVISED = ROOT / "upstream" / "PatchTST" / "PatchTST_supervised"
if str(PATCHTST_SUPERVISED) not in sys.path:
    sys.path.insert(0, str(PATCHTST_SUPERVISED))

from layers.PatchTST_backbone import PatchTST_backbone as OfficialPatchTSTBackbone  # noqa: E402


ARCHITECTURE_KEYS = (
    "enc_in",
    "seq_len",
    "pred_len",
    "patch_len",
    "stride",
    "e_layers",
    "d_model",
    "n_heads",
    "d_ff",
    "dropout",
    "fc_dropout",
    "head_dropout",
    "padding_patch",
    "revin",
    "affine",
    "subtract_last",
    "individual",
)


def architecture_config(values: Dict[str, Any]) -> Dict[str, Any]:
    """Return a complete, typed PatchTST architecture configuration."""
    defaults: Dict[str, Any] = {
        "patch_len": 16,
        "stride": 8,
        "e_layers": 3,
        "d_model": 128,
        "n_heads": 16,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "head_dropout": 0.0,
        "padding_patch": "end",
        "revin": True,
        "affine": False,
        "subtract_last": False,
        "individual": False,
    }
    defaults.update({key: values[key] for key in ARCHITECTURE_KEYS if key in values})
    integer_keys = {
        "enc_in", "seq_len", "pred_len", "patch_len", "stride", "e_layers",
        "d_model", "n_heads", "d_ff",
    }
    float_keys = {"dropout", "fc_dropout", "head_dropout"}
    bool_keys = {"revin", "affine", "subtract_last", "individual"}
    for key in integer_keys:
        defaults[key] = int(defaults[key])
    for key in float_keys:
        defaults[key] = float(defaults[key])
    for key in bool_keys:
        defaults[key] = bool(defaults[key])
    return defaults


class PatchTSTBackbone(nn.Module):
    """Official PatchTST with an audited pre-projection feature interface."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = architecture_config(config)
        if self.config["individual"]:
            raise ValueError("Gate 5b requires the official shared PatchTST forecast head")
        self.model = OfficialPatchTSTBackbone(
            c_in=self.config["enc_in"],
            context_window=self.config["seq_len"],
            target_window=self.config["pred_len"],
            patch_len=self.config["patch_len"],
            stride=self.config["stride"],
            n_layers=self.config["e_layers"],
            d_model=self.config["d_model"],
            n_heads=self.config["n_heads"],
            d_ff=self.config["d_ff"],
            dropout=self.config["dropout"],
            fc_dropout=self.config["fc_dropout"],
            head_dropout=self.config["head_dropout"],
            padding_patch=self.config["padding_patch"],
            revin=self.config["revin"],
            affine=self.config["affine"],
            subtract_last=self.config["subtract_last"],
            individual=False,
        )
        if not hasattr(self.model.head, "linear") or not hasattr(self.model.head, "flatten"):
            raise RuntimeError("official shared Flatten_Head interface was not found")

    @property
    def hidden_size(self) -> int:
        return int(self.model.head_nf)

    @property
    def deterministic_projection(self) -> nn.Linear:
        return self.model.head.linear

    def encode(self, batch_x: Tensor) -> Tensor:
        """Return `[batch, channel, d_model * patch_count]` features."""
        if batch_x.ndim != 3:
            raise ValueError(f"expected [B,L,C], got {tuple(batch_x.shape)}")
        if batch_x.shape[1] != self.config["seq_len"]:
            raise ValueError(f"expected seq_len={self.config['seq_len']}, got {batch_x.shape[1]}")
        if batch_x.shape[2] != self.config["enc_in"]:
            raise ValueError(f"expected channels={self.config['enc_in']}, got {batch_x.shape[2]}")

        z = batch_x.permute(0, 2, 1)
        if self.model.revin:
            z = self.model.revin_layer(z.permute(0, 2, 1), "norm").permute(0, 2, 1)
        if self.model.padding_patch == "end":
            z = self.model.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.model.patch_len, step=self.model.stride)
        z = z.permute(0, 1, 3, 2)
        encoded = self.model.backbone(z)
        hidden = self.model.head.flatten(encoded)
        if hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(f"hidden width mismatch: {hidden.shape[-1]} != {self.hidden_size}")
        return hidden

    def denormalize_location(self, normalized_forecast: Tensor) -> Tensor:
        """Apply official RevIN denormalization and return `[B,pred,C]`."""
        forecast = normalized_forecast.permute(0, 2, 1)
        if self.model.revin:
            forecast = self.model.revin_layer(forecast, "denorm")
        return forecast

    def forecast_from_hidden(self, hidden: Tensor) -> Tensor:
        normalized = self.model.head.dropout(self.deterministic_projection(hidden))
        return self.denormalize_location(normalized)

    def forward_with_hidden(self, batch_x: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = self.encode(batch_x)
        return hidden, self.forecast_from_hidden(hidden)

    def forward(self, batch_x: Tensor) -> Tensor:
        return self.forward_with_hidden(batch_x)[1]

