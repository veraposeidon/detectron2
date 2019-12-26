# coding=utf-8
"""
multi loss layer
"""

import torch
from torch import nn


class MultiTaskLoss(nn.Module):
    """
    多任务损失层
    """

    def __init__(self, cfg):
        super(MultiTaskLoss, self).__init__()
        # 记得要讲这些参数注册到module的named_parameters

        if cfg.MODEL.MULTI_TASK.DETECTION_ON:
            self.is_detection = True
            self.log_var_object = torch.nn.Parameter(torch.ones(1, requires_grad=True), requires_grad=True)
        else:
            self.is_detection = False
            self.log_var_object = None

        if cfg.MODEL.MULTI_TASK.CLASSIFICATION_ON:
            self.is_classification = True
            self.log_var_classify = torch.nn.Parameter(torch.ones(1, requires_grad=True), requires_grad=True)
        else:
            self.is_classification = False
            self.log_var_classify = None

        if cfg.MODEL.MULTI_TASK.SEGMENTATION_ON:
            self.is_segmentation = True
            self.log_var_segmentation = torch.nn.Parameter(torch.ones(1, requires_grad=True), requires_grad=True)
        else:
            self.is_segmentation = False
            self.log_var_segmentation = None

    def forward(self, loss_dict):
        if self.is_detection:
            detection_loss = loss_dict['loss_cls'] + loss_dict['loss_box_reg'] + loss_dict['loss_rpn_cls'] + loss_dict[
                'loss_rpn_loc']
            detection_weighted = torch.exp(-self.log_var_object) * detection_loss + self.log_var_object
        else:
            detection_weighted = 0

        if self.is_classification:
            classification_loss = loss_dict['mulbl_cls_loss']
            classification_weighted = torch.exp(-self.log_var_classify) * classification_loss + self.log_var_classify
        else:
            classification_weighted = 0

        if self.is_segmentation:
            segmentation_loss = loss_dict['sem_seg_loss']
            segmentation_weighted = torch.exp(
                -self.log_var_segmentation) * segmentation_loss + self.log_var_segmentation
        else:
            segmentation_weighted = 0

        multi_loss = detection_weighted + classification_weighted + segmentation_weighted

        loss_dict['detection_weighted'] = detection_weighted
        loss_dict['classification_weighted'] = classification_weighted
        loss_dict['segmentation_weighted'] = segmentation_weighted

        return multi_loss


def build_multitask_loss_layer(cfg):
    return MultiTaskLoss(cfg)
