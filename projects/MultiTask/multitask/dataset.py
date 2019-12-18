# coding=utf-8
"""
数据集加载，使用PASCAL VOC格式
"""

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

CLASS_NAMES = ["BuDaoDian", "CaHua", "JiaoWeiLouDi", "JuPi", "LouDi", "PengLiu", "QiPao", "QiKeng", "ZaSe",
               "ZangDian", ]


# 重写VOC实例加载，因为除了目标检测，还要加载多标签和分割样本
def load_al_voc_instances(dir_name: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dir_name: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dir_name, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dir_name, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dir_name, "JPEGImages", fileid + ".jpg")
        segm_file = os.path.join(dir_name, "SegmentationClassPNG", fileid + ".png")     # 铝材表面分割图

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "seg_file_name": segm_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []
        multi_labels = []    # 多标签整理
        multi_label_names = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            # 我的数据集内没有这个问题
            # bbox[0] -= 1.0
            # bbox[1] -= 1.0
            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )

            if CLASS_NAMES.index(cls) not in multi_labels:
                multi_labels.append(CLASS_NAMES.index(cls))
                multi_label_names.append(cls)

        r["annotations"] = instances
        r["multi_labels"] = multi_labels
        r["multi_label_names"] = multi_label_names
        dicts.append(r)
    return dicts


def register_pascal_voc_al(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_al_voc_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, split=split, year=2012  # 2012 是因为PASCAL evaluator中有断言2007或2012
    )


# name, dir, split
SPLITS = [
    ("voc_al_train", "VOC_AL", "train"),
    ("voc_al_test", "VOC_AL", "test"), ]

# 　此处注册
root = "datasets"
for name, dirname, split in SPLITS:
    register_pascal_voc_al(name, os.path.join(root, dirname), split)
    MetadataCatalog.get(name).evaluator_type = "pascal_voc"
