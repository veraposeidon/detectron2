# coding=utf-8
"""
multi task learning meta architecture.
"""

import logging
import numpy as np
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.utils.visualizer import Visualizer

from detectron2.modeling import (
    META_ARCH_REGISTRY,
    build_backbone,
    build_proposal_generator,
    build_roi_heads,
    detector_postprocess,
)

from .multiLabelClassifier import build_multilabel_classifier
from .metal_segmentation import build_metal_segmentation_model
from .multi_task_loss import build_multitask_loss_layer

__all__ = ["GeneralizedRCNNMultiTask", ]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNMultiTask(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # build feature extraction backbone
        self.backbone = build_backbone(cfg)

        # build classification model
        if cfg.MODEL.MULTI_TASK.CLASSIFICATION_ON:
            self.classifier_in_features = cfg.MODEL.MULTI_TASK.CLASSIFICATION_IN_FEATURES
            self.classifier = build_multilabel_classifier(cfg)
        else:
            self.classifier = None

        # build segmentation model
        if cfg.MODEL.MULTI_TASK.SEGMENTATION_ON:
            self.metal_segmentation_in_features = cfg.MODEL.MULTI_TASK.SEGMENTATION_IN_FEATURES
            self.metal_segmentation = build_metal_segmentation_model(cfg, self.backbone.out_feature_strides)
        else:
            self.metal_segmentation = None

        # build object detection model
        if cfg.MODEL.MULTI_TASK.DETECTION_ON:
            self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
            self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        else:
            self.proposal_generator = None
            self.roi_heads = None

        # TODO: build multi-task layer
        self.multi_loss_layer = build_multitask_loss_layer(cfg)

        # other setting
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)    # all to cuda

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """

        inputs = [x for x in batched_inputs]
        prop_boxes = [p for p in proposals]
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(inputs, prop_boxes):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = " 1. GT bounding boxes  2. Predicted proposals"
            storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        # preprocess data
        # images for input
        images = self.preprocess_image(batched_inputs)
        # instance for object detection
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        # classification data
        if "multi_labels" in batched_inputs[0]:
            classifier_targets = [o['multi_labels'] for o in batched_inputs]
        else:
            classifier_targets = None
        # segmentation data
        if 'sem_seg' in batched_inputs[0]:
            segmentation_targets = self.preprocess_semseg_image(batched_inputs)
        else:
            segmentation_targets = None

        # backbone extract features
        features = self.backbone(images.tensor)

        # task: object detection
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # task: multi-label classification
        if self.classifier is not None:
            # classifier_features = [features[f] for f in self.classifier_in_features]
            classifier_features = features[self.classifier_in_features[0]]
            _, multi_label_losses = self.classifier(classifier_features, classifier_targets)
        else:
            multi_label_losses = {}

        # task: metal segmentation model
        if self.metal_segmentation is not None:
            segmentation_features = features[self.metal_segmentation_in_features[0]]
            _, segmentation_losses = self.metal_segmentation(segmentation_features, segmentation_targets)
        else:
            segmentation_losses = {}

        # visualize
        if self.vis_period > 0:  # vis_period > 0, 就添加图像可视化
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        # loss dict
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(multi_label_losses)
        losses.update(segmentation_losses)

        # TODO: multi_loss layer computation.
        multi_loss = self.multi_loss_layer(losses)

        return losses, multi_loss

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_semseg_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["sem_seg"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
