"""Loss functions used by the public GDF-Restormer training path."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim


class SSIMLoss(nn.Module):
    """Structural similarity loss: 1 - SSIM."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        value = ssim(pred, target, data_range=1.0)
        value = torch.clamp(value, min=0.0, max=1.0)
        return 1.0 - value
