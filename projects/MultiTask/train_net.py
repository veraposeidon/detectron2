# coding=utf-8
"""
MultiTask Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

from comet_experiment import Experiment

import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import PascalVOCDetectionEvaluator, DatasetEvaluators, verify_results
from detectron2.utils.logger import setup_logger

from multitask import DatasetMapper, add_multitask_config, DefaultTrainerCometWriter


class Trainer(DefaultTrainerCometWriter):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [PascalVOCDetectionEvaluator(dataset_name)]  # FIXME: evaluator需要再改
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))


def setup(args):
    cfg = get_cfg()  # 默认配置
    add_multitask_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "multitask" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="multitask")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        Experiment.experiment.validate()
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
