"""xG models: image backbone -> global pooling -> concat(extra features) -> MLP head -> logit.

Every model has the signature `forward(image, extra) -> logit` with
    image : (B, N_CHANNELS, IMG_H, IMG_W)
    extra : (B, N_EXTRA_FEATURES)       e.g. body part one-hot
    logit : (B,)                        P(goal) = sigmoid(logit)
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from src import config as C


class XGHead(nn.Module):
    """concat(pooled image features, extra features) -> hidden -> 1 logit."""

    def __init__(self, in_features: int, n_extra: int = C.N_EXTRA_FEATURES,
                 hidden: int = C.HEAD_HIDDEN, dropout: float = C.HEAD_DROPOUT,
                 prior: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features + n_extra, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Start at the base goal rate: avoids a large, uninformative loss in the first steps.
        nn.init.constant_(self.mlp[-1].bias, math.log(prior / (1 - prior)))

    def forward(self, feats: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feats, extra.to(feats.dtype)], dim=1)).squeeze(1)


# ---------------------------------------------------------------------------
# Small CNN trained from scratch
# ---------------------------------------------------------------------------
def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ChannelSelect(nn.Module):
    """
    Keeps only some input channels (for ablations). Models always receive the full
    N_CHANNELS tensor, so data, evaluation and Grad-CAM code stay the same for every model.
    The index is a non-persistent buffer: checkpoints without it still load.
    """

    def __init__(self, channels=None):
        super().__init__()
        self.channels = None if channels is None else tuple(int(c) for c in channels)
        idx = torch.arange(C.N_CHANNELS) if channels is None else torch.tensor(self.channels)
        self.register_buffer("idx", idx, persistent=False)

    @property
    def n_out(self) -> int:
        return len(self.idx)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image if self.channels is None else image.index_select(1, self.idx)


class SmallCNN(nn.Module):
    """5 x [conv-bn-relu x2 + maxpool]: 160x120 -> 5x3, receptive field ~ 130 px (65 yd)."""

    def __init__(self, widths=(16, 32, 64, 128, 256), prior: float = 0.1, channels=None):
        super().__init__()
        self.select = ChannelSelect(channels)
        layers, c = [], self.select.n_out
        for w in widths:
            layers.append(_conv_block(c, w))
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = XGHead(c, prior=prior)
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, image: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(self.select(image))).flatten(1), extra)

    def param_groups(self, lr: float, backbone_lr: float | None = None):
        return [{"params": self.parameters(), "lr": lr}]


# ---------------------------------------------------------------------------
# ImageNet-pretrained ResNet18 with a 6-channel first conv
# ---------------------------------------------------------------------------
def adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    New conv with `in_channels` inputs. Every input channel gets the mean of the
    pretrained RGB filters, scaled by 3 / in_channels so that the activation scale
    for a 'grey' input stays the same.
    """
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        w = conv.weight.mean(dim=1, keepdim=True) * (3.0 / in_channels)
        new.weight.copy_(w.repeat(1, in_channels, 1, 1))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class ResNet18xG(nn.Module):
    """
    torchvision ResNet18 (IMAGENET1K_V1). maxpool=False keeps 2x more spatial
    resolution (160x120 -> 10x8 instead of 5x4) at ~4x the compute.
    """

    def __init__(self, pretrained: bool = True, maxpool: bool = True, prior: float = 0.1, channels=None):
        super().__init__()
        self.select = ChannelSelect(channels)
        # imported here: DataLoader workers re-import the training script and do not need torchvision
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.conv1 = adapt_first_conv(net.conv1, self.select.n_out)
        if not maxpool:
            net.maxpool = nn.Identity()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.body = nn.Sequential(net.layer1, net.layer2, net.layer3, net.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = XGHead(net.fc.in_features, prior=prior)

    def forward(self, image: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.body(self.stem(self.select(image)))).flatten(1), extra)

    def param_groups(self, lr: float, backbone_lr: float | None = None):
        """New layers (6-ch conv1, head) use `lr`; pretrained layers use `backbone_lr`."""
        backbone_lr = lr if backbone_lr is None else backbone_lr
        new = [*self.stem[0].parameters(), *self.head.parameters()]
        new_ids = {id(p) for p in new}
        pretrained = [p for p in self.parameters() if id(p) not in new_ids]
        return [{"params": new, "lr": lr}, {"params": pretrained, "lr": backbone_lr}]


# ---------------------------------------------------------------------------
def build_model(name: str, prior: float = 0.1, pretrained: bool | None = None) -> nn.Module:
    """
    Model from its MODEL_CONFIGS entry. `pretrained=None` uses the config value;
    load_checkpoint passes False because the weights come from the checkpoint anyway.
    """
    cfg = C.MODEL_CONFIGS[name]
    channels = cfg.get("channels")
    if cfg["arch"] == "small_cnn":
        return SmallCNN(prior=prior, channels=channels)
    if cfg["arch"] == "resnet18":
        pretrained = cfg.get("pretrained", True) if pretrained is None else pretrained
        return ResNet18xG(pretrained=pretrained, maxpool=cfg.get("maxpool", True), prior=prior,
                          channels=channels)
    raise ValueError(f"unknown architecture: {cfg['arch']}")


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a model from a best.pt written by src.train. Returns (model in eval mode, checkpoint)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], prior=ckpt["prior"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
