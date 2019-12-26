# coding=utf-8

from comet_experiment import Experiment
import datetime
import time
import logging
import os
import torch
from detectron2.engine import SimpleTrainer, DefaultTrainer
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventWriter, get_event_storage


class DefaultTrainerCometWriter(DefaultTrainer, SimpleTrainer):
    """
    用于添加comet.ml 添加记录信息
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        留给trainer实现
        :param cfg:
        :param dataset_name:
        :return:
        """
        pass

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinterWithComet(self.max_iter),    # 重载实现
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    # override run_step to control the pipe line. Mainly for the loss_dict.
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()


class CommonMetricPrinterWithComet(CommonMetricPrinter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.

    To print something different, please implement a similar printer by yourself.
    """

    def __init__(self, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """

        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter
        self.expriment = Experiment.experiment

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter

        data_time, time = None, None
        eta_string = "N/A"
        try:
            data_time = storage.history("data_time").avg(20)
            time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:  # they may not exist in the first few iterations (due to warmup)
            pass

        try:
            lr = "{:.6f}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            """\
eta: {eta}  iter: {iter}  {losses}  \
{time}  {data_time}  \
lr: {lr}  {memory}\
""".format(
                eta=eta_string,
                iter=iteration,
                losses="  ".join(
                    [
                        "{}: {:.3f}".format(k, v.median(20))
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                time="time: {:.4f}".format(time) if time is not None else "",
                data_time="data_time: {:.4f}".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )

        # comet.ml记录
        for k,v in storage.histories().items():
            self.expriment.log_metric(k, v.median(20), step=iteration)
        self.expriment.log_metric("lr", lr, step=iteration)
        self.expriment.log_metric("max_mem(M)", max_mem_mb, step=iteration)