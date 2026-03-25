import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import SSIMLoss


class MultiScaleSSIMKD(nn.Module):
    """
    A knowledge distillation loss that computes the multi-scale SSIM between student and teacher feature maps.
    It normalizes the feature maps, averages across channels, and computes the SSIM loss for specified keys and weights.
    """
    def __init__(self, keys=("2",), weights=(1.0,), window_size=7):

        """
        Args:
            keys: A tuple of keys corresponding to the feature maps to compare (e.g., "2", "3", "4" for FPN levels).
            weights: A tuple of weights for each key to balance their contributions to the total loss.
            window_size: The window size for the SSIM computation (default is 7).
        """
        super().__init__()
        self.keys = keys
        self.weights = weights
        self.ssim = SSIMLoss(window_size=window_size, reduction="mean")

    @staticmethod
    def normalize_feat(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        # Add a small epsilon to std to avoid division by zero
        std = x.std(dim=(2, 3), keepdim=True) + 1e-6
        return (x - mean) / std

    def forward(self, student_feats, teacher_feats):
        total = 0.0
        for k, w in zip(self.keys, self.weights):
            fs = student_feats[k]
            ft = teacher_feats[k]

            if fs.shape[-2:] != ft.shape[-2:]:
                fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)

            fs = self.normalize_feat(fs).mean(dim=1, keepdim=True)
            ft = self.normalize_feat(ft).mean(dim=1, keepdim=True)

            total = total + w * self.ssim(fs, ft)
        return total