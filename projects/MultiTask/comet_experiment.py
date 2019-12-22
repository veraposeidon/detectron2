# coding=utf-8
# comet.ml 实验管理

from comet_ml import Experiment
# from comet_ml import OfflineExperiment


class Experiment:
    # experiment = Experiment(project_name="pytorch")
    experiment = Experiment()

    # 离线实验
    # experiment = OfflineExperiment(
    #     workspace="WORKSPACE_NAME",
    #     project_name="PROJECT_NAME",
    #     offline_directory="/tmp")
    # 上传指令
    # /usr/bin/python -m comet_ml.scripts.comet_upload /tmp/comet/a3e24ed1ed07477693c1c5c05507f216.zip
