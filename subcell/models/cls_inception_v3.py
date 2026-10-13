"""The bestfitting baseline: Inception v3 with CBAM attention and a multi-label head.

The architecture of the "bestfitting" model that won a Kaggle competition for
protein localization prediction in Human Protein Atlas images. ``feature_vector``
(the 2,048-d penultimate layer) is the embedding used in the analyses. Built and
loaded by ``subcell.models.load.get_bestfitting_model``.

Adapted from the winning solution of the HPA Single Cell Classification challenge
(Shubin Dai, "bestfitting"), as released in
https://github.com/CellProfiling/HPA_SingleCellClassification_Analysis
(``team1_bestfitting/predict/net/cls_inception_v3.py``), MIT License.
"""
from dataclasses import dataclass

import torch
from torch import nn

from subcell.models.inception_v3 import inception_v3


@dataclass
class BestFittingOutput:
    probs: torch.Tensor
    features: torch.Tensor
    feature_vector: torch.Tensor


class CBAM_Module(nn.Module):
    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(
            2, 1, kernel_size=3, stride=1, padding=3 // 2
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        att = self.sigmoid_spatial(x)
        ret = module_input * att
        return ret, att


class InceptionV3(nn.Module):
    """``args`` holds ``in_channels`` and ``num_classes``; ``att_type="cbam"`` adds
    the attention module before pooling."""

    def __init__(self, args, att_type=None):
        super().__init__()
        self.args = args
        self.att_type = att_type

        self.backbone = inception_v3()

        if self.args["in_channels"] > 3:
            w = self.backbone.Conv2d_1a_3x3.conv.weight
            self.backbone.Conv2d_1a_3x3.conv = nn.Conv2d(
                self.args["in_channels"],
                32,
                kernel_size=(3, 3),
                stride=(2, 2),
                bias=False,
            )
            self.backbone.Conv2d_1a_3x3.conv.weight = torch.nn.Parameter(
                torch.cat(
                    [w] * int(self.args["in_channels"] // 3)
                    + [w[:, : int(self.args["in_channels"] % 3), :, :]],
                    dim=1,
                )
            )

        self.backbone.layer0 = nn.Sequential(
            self.backbone.Conv2d_1a_3x3,
            self.backbone.Conv2d_2a_3x3,
            self.backbone.Conv2d_2b_3x3,
        )
        self.backbone.layer1 = nn.Sequential(
            self.backbone.Conv2d_3b_1x1,
            self.backbone.Conv2d_4a_3x3,
        )
        self.backbone.layer2 = nn.Sequential(
            self.backbone.Mixed_5b,
            self.backbone.Mixed_5c,
            self.backbone.Mixed_5d,
        )
        self.backbone.layer3 = nn.Sequential(
            self.backbone.Mixed_6a,
            self.backbone.Mixed_6b,
            self.backbone.Mixed_6c,
            self.backbone.Mixed_6d,
            self.backbone.Mixed_6e,
        )
        self.backbone.layer4 = nn.Sequential(
            self.backbone.Mixed_7a,
            self.backbone.Mixed_7b,
            self.backbone.Mixed_7c,
        )

        feature_dim = 2048
        if self.att_type in ["cbam"]:
            self.att_module = CBAM_Module(channels=feature_dim, reduction=32)
        else:
            self.att_module = None

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        feature_nums = 2 * feature_dim
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(feature_nums),
            nn.Dropout(p=0.5),
            nn.Linear(feature_nums, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=0.5),
        )
        self.logit = nn.Linear(
            in_features=feature_dim, out_features=self.args["num_classes"]
        )
    def forward(self, image):
        x = self.backbone.layer0(image)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        features = x
        if self.att_type in ["cbam"]:
            x, _ = self.att_module(x)
        if x.size() == features.size():
            features = x

        x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        logits = self.logit(x)
        probs = torch.sigmoid(logits)

        return BestFittingOutput(probs=probs, features=features, feature_vector=x)
