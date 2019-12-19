# coding=utf-8
"""
设置多任务实验的专属参数
"""

from detectron2.config import CfgNode as CN


def add_multitask_config(cfg):
    """
    Add config for multi task project.
    """
    _C = cfg

    # 使用多任务模式
    _C.MODEL.MULTI_TASK_ON = True
    _C.MODEL.MULTI_TAK = CN()
    # 开启多标签分类任务
    _C.MODEL.MULTI_TASK.CLASSIFICATION_ON = True
    # 开启目标检测任务
    _C.MODEL.MULTI_TASK.DETECTION_ON = True
    # 开启分割任务
    _C.MODEL.MULTI_TASK.SEGMENTATION_ON = True
