# from fvcore.common.config import CfgNode
from detectron2.config.config import CfgNode

# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


def add_mlflow_config(cfg):
    """
    Add config for MLflow.
    """
    cfg.MLFLOW = CfgNode()
    cfg.MLFLOW.EXPERIMENT_NAME = "Default Experiment"
    cfg.MLFLOW.RUN_DESCRIPTION = "Default Description"
    cfg.MLFLOW.RUN_NAME = "Default Run"
    cfg.MLFLOW.TRACKING_URI = "http://localhost:5000"
    cfg.MLFLOW.ARTIFACT_URI = "./artifacts"
