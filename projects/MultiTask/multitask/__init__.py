# coding=utf-8
from . import dataset
from .dataset_mapper import DatasetMapper
from .config import add_multitask_config

from .trainer import MultiTaskTrainer     # 重载Trainer
from .multiLabelClassifier import build_multilabel_classifier   # 多标签分类器
from .metal_segmentation import build_metal_segmentation_model  # 金属分割模型
from .multi_task_loss import build_multitask_loss_layer         # 多任务损失
from .multitask_RCNN import GeneralizedRCNNMultiTask    # R-CNN 基本结构
