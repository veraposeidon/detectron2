# coding=utf-8

"""
金属区域分割
"""

import torch
from torch import nn
from torch.nn import functional as F


class SegmentationModel(nn.Module):
    """
    segmentation model, based on the second half of the FCN model.
    """
    def __init__(self, cfg, feature_strides):
        super(SegmentationModel, self).__init__()
        num_classes = 2     # metal part and background
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS    # FPN output features have the same channel numbers.
        # FIXME 计算缩放倍数，自动填写channel缩放倍数
        multiple = feature_strides[cfg.MODEL.MULTI_TASK.SEGMENTATION_IN_FEATURES[0]]    # 倍数

        # deconvolution part
        out_channels = in_channels // 2
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

        in_channels = out_channels
        out_channels = in_channels // 2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

        in_channels = out_channels
        out_channels = num_classes
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, features, targets):
        pass


def build_metal_segmentation_model(cfg, feature_strides):
    return SegmentationModel(cfg, feature_strides)
