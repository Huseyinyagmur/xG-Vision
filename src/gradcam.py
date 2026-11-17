"""Grad-CAM for the xG models (Selvaraju et al., 2017).

For a feature map A^k of a chosen layer, the weights are the spatially averaged gradients
of the goal logit, and the map is ReLU(sum_k w_k A^k), upsampled to the input size.
The extra input (body part) is held fixed; only the image path is explained.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src import config as C


def smallcnn_layer(model: nn.Module, block: int) -> nn.Module:
    """Output of SmallCNN block `block` (0-based, after its max-pool).
    Sizes for a 160x120 input: 0 -> 80x60, 1 -> 40x30, 2 -> 20x15, 3 -> 10x7, 4 -> 5x3."""
    return model.features[block]


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model.eval()
        self._acts = None
        self._grads = None
        self._h = [layer.register_forward_hook(self._save_act)]

    def _save_act(self, module, inp, out):
        self._acts = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad

    def remove(self):
        for h in self._h:
            h.remove()

    def __call__(self, image: torch.Tensor, extra: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        image (B, C, H, W), extra (B, n_extra), both on the model's device, float32.
        Returns (cams (B, H, W) in [0, 1] (each map divided by its max), P(goal) (B,)).
        Runs in float32 without autocast so the gradients are exact.
        """
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(image, extra)
            logits.sum().backward()                     # samples are independent in eval mode
        w = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(C.IMG_H, C.IMG_W), mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).max(1).values.clamp_min(1e-12)[:, None, None]
        return cam.detach().cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy()
