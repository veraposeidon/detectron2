# coding=utf-8
from . import dataset
from .dataset_mapper import DatasetMapper
from .config import add_multitask_config
from .multitask_RCNN import GeneralizedRCNNMultiTask    # R-CNN 基本结构
from .trainerwithComet import DefaultTrainerCometWriter     # 重载Trainer
from .multiLabelClassifier import build_multilabel_classifier   # 多标签分类器
