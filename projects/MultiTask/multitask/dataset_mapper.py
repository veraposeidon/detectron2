# coding=utf-8
"""
用于对dataset的样本，进行读取图像，数据增强，转换为tensor等操作。
"""

import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


class DatasetMapper:
    """
    A customized version of `detectron2.data.DatasetMapper`
    """

    def __init__(self, cfg, is_train=True):
        self.num_seg_class = 2  # segmentation: foreground and background
        # crop
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        # get basic transform generator, includes resizing and flipping.
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        self.img_format = cfg.INPUT.FORMAT  # default type is "BGR"
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT

        # no mask, no key point, no densepose, mo proposals
        assert not cfg.MODEL.LOAD_PROPOSALS, "not supported yet"

        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # get sample information and read image
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # check real image size match the sample description in dataset_dict
        utils.check_image_size(dataset_dict, image)

        # IMAGE AUGMENTATION
        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

        # these information no needed in train mode
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("multi_labels", None)
            dataset_dict.pop("multi_label_names", None)
            dataset_dict.pop("seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = utils.annotations_to_instances(annos, image_shape)

            # 虽然用不上，但还是保留吧
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if "seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            # TODO: one-hot encoding after transformation
            sem_gt = np.zeros((self.num_seg_class, sem_seg_gt.shape[0], sem_seg_gt.shape[1]), dtype="uint8")
            for c in range(self.num_seg_class):
                sem_gt[c][sem_seg_gt == c] = 1
            # sem_gt = np.transpose(sem_gt, (1, 2, 0))  # for later transform
            sem_gt = torch.as_tensor(sem_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_gt

        if "multi_labels" in dataset_dict:
            dataset_dict["multi_labels"] = dataset_dict.pop("multi_labels")

        return dataset_dict
