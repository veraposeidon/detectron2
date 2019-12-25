# coding=utf-8

import torch
from torch import nn
from torch.nn import functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)


class MultiLabelClassifier(nn.Module):
    """
    用于对提取的特征层进行多标签分类，预测图像级别的缺陷分类
    """

    def __init__(self, cfg):
        super(MultiLabelClassifier, self).__init__()

        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES  # numbers of foreground class
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS

        # 仿造fastai做法，用cat maxpool 和 avg pool 的做法
        # https://forums.fast.ai/t/what-is-the-distinct-usage-of-the-adaptiveconcatpool2d-layer/7600
        # 原因在于：对于脏点，最好用max以确认哪里,对于大面积，最好用avg以确定区域

        # 由于size未知，直接定义输出为1。注意通道数翻倍
        output_sz = 1
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=output_sz)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=output_sz)
        self.bn_drop_lin = nn.Sequential(
            # nn.BatchNorm1d(2*in_channels),  # 这个需要Batch不为1时的Normalization。
            nn.Dropout(0.5),
            nn.Linear(2 * in_channels, 512),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # foreground classes
        )

        # 初始化权重
        self.bn_drop_lin.apply(weights_init)

    def forward(self, features, targets):
        # max pooling and avg pooling, then concat two
        output = torch.cat([self.avgpool(features), self.maxpool(features)], 1)
        # flatten for fully connect
        output = output.view(output.size(0), -1)  # Flatten操作
        # predict
        predict_logits = self.bn_drop_lin(output)  # 全连接层

        # one hot code
        if targets is not None:
            y = torch.zeros_like(predict_logits)
            for i in range(y.shape[0]):
                for o in targets[i]:
                    y[i][o] = 1
            # TODO: 此处可加soft loss
            loss = F.binary_cross_entropy_with_logits(predict_logits, y)
        else:
            loss = None
        # TODO: 模仿roi_heads.fast_rcnn 的 softmax_cross_entropy_loss，进行log_accuracy

        # 返回
        return predict_logits, dict(mulbl_cls_loss=loss)


def build_multilabel_classifier(cfg):
    return MultiLabelClassifier(cfg)
