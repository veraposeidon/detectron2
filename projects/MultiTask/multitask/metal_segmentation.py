# coding=utf-8

"""
金属区域分割
"""

import torch
from torch import nn
from torch.nn import functional as F
import math


class SegmentationModel(nn.Module):
    """
    segmentation model, based on the second half of the FCN model.
    """

    def __init__(self, cfg, feature_strides):
        super(SegmentationModel, self).__init__()
        num_classes = 2  # metal part and background
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS  # FPN output features have the same channel numbers.

        multiple = feature_strides[cfg.MODEL.MULTI_TASK.SEGMENTATION_IN_FEATURES[0]]  # 缩放倍数 从数据输入到特征图
        count = int(math.log(multiple, 2))

        self.deconvs = nn.Sequential()
        for i in range(count):
            out_channels = in_channels // 2
            self.deconvs.add_module(
                "upsample_" + str(i),
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1,
                                       output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels)
                )
            )
            in_channels = out_channels

        out_channels = num_classes
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, features, targets):
        output = self.deconvs(features)
        predict_logits = self.classifier(output)

        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(predict_logits, targets.tensor.float())
        else:
            loss = None
        return predict_logits, dict(sem_seg_loss=loss)


def build_metal_segmentation_model(cfg, feature_strides):
    return SegmentationModel(cfg, feature_strides)
