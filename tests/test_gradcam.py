"""Grad-CAM shape and consistency (no data needed)."""
import numpy as np
import torch

from src import config as C
from src.gradcam import GradCAM, smallcnn_layer
from src.models import SmallCNN


def test_gradcam_shapes_and_probabilities():
    torch.manual_seed(0)
    model = SmallCNN(prior=0.1).eval()
    img = torch.rand(3, C.N_CHANNELS, C.IMG_H, C.IMG_W)
    extra = torch.eye(C.N_EXTRA_FEATURES)[:3]
    cam = GradCAM(model, smallcnn_layer(model, 2))
    maps, probs = cam(img, extra)
    cam.remove()
    assert maps.shape == (3, C.IMG_H, C.IMG_W)
    assert maps.min() >= 0 and np.allclose(maps.reshape(3, -1).max(1)[maps.reshape(3, -1).max(1) > 0], 1)
    with torch.no_grad():
        np.testing.assert_allclose(probs, torch.sigmoid(model(img, extra)).numpy(), rtol=1e-5)
    assert not model._forward_hooks and not model.features[2]._forward_hooks      # hooks removed
