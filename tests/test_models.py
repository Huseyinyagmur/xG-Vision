"""Model and feature tests. They need no data and no pretrained-weight download."""
import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src import config as C
from src.features import body_part_onehot
from src.models import SmallCNN, ResNet18xG, adapt_first_conv


def _batch(b=4):
    img = torch.rand(b, C.N_CHANNELS, C.IMG_H, C.IMG_W)
    extra = torch.eye(C.N_EXTRA_FEATURES)[torch.arange(b) % C.N_EXTRA_FEATURES]
    return img, extra


@pytest.mark.parametrize("model", [SmallCNN(prior=0.1),
                                   ResNet18xG(pretrained=False, maxpool=True, prior=0.1),
                                   ResNet18xG(pretrained=False, maxpool=False, prior=0.1)])
def test_forward_shape_and_extra_is_used(model):
    model.eval()
    img, extra = _batch()
    out = model(img, extra)
    assert out.shape == (4,)
    # changing only the body part must change the output (the extra path is wired in)
    assert not torch.allclose(out, model(img, extra.roll(1, dims=1)))


def test_head_starts_at_prior():
    # the output bias is initialised to logit(prior)
    model = SmallCNN(prior=0.104)
    assert math.isclose(model.head.mlp[-1].bias.item(), math.log(0.104 / 0.896), rel_tol=1e-6)


def test_adapt_first_conv_keeps_scale():
    conv = nn.Conv2d(3, 8, 7, stride=2, padding=3, bias=False)
    new = adapt_first_conv(conv, 6)
    assert new.weight.shape == (8, 6, 7, 7)
    # a uniform 6-channel input gives the same response as a uniform (grey) RGB input
    # through the channel-averaged RGB filters
    grey_rgb_filters = conv.weight.mean(1, keepdim=True).repeat(1, 3, 1, 1)
    expected = nn.functional.conv2d(torch.ones(1, 3, 16, 16), grey_rgb_filters, stride=2, padding=3)
    torch.testing.assert_close(new(torch.ones(1, 6, 16, 16)), expected)


def test_body_part_onehot():
    s = pd.Series(["Head", "Right Foot", "Left Foot", "Other"])
    np.testing.assert_array_equal(body_part_onehot(s), [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        body_part_onehot(pd.Series(["Knee"]))
