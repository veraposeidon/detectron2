# Multi Task Project
旨在用多任务学习的方式进行目标检测任务。

## 文件介绍

- `configs/`：存放配置文件的文件夹
- `datasets/`：默认数据集文件夹
- `multitask/`：multitask为该项目下的核心函数的实现，包括重写detectron2的函数和模型以及自建的模型定义和函数
    - `__init__.py`：各组件初始化
    - `config.py`：multitask任务中单独的配置参数
    - `dataset.py`：multitask任务中数据集样本信息的加载，PASCAL_VOC format
    - `dataset_mapper.py`：mapper用于将样本信息，转换为模型输入，包括：read image, image augmentation, transfer to cuda tensors
    - `multitask_RCNN.py`：模型主框架 meta architecture，重写自GeneralizedRCNN，因为需要在此处添加除目标检测之外的任务，因此需要做修改
    - `trainerwithComet.py`：重写自DefaultTrainer，因为需要添加一项comet.ml实验管理时的过程记录，需要重写build_writers,在CommonMetricPrinter基础上添加comet.ml实验记录
- `train_net.py`：训练脚本
- `comet_experiment.py`：comet.ml进行实验管理的初始化脚本
- `dataloader_test.ipynb`：jupyter notebook 进行dataloader的测试
- `dataset_test.ipynb`：jupyter notebook 进行dataset的测试
