import os
import time
from itertools import chain

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils
import mlflow.pytorch
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER, MLFLOW_SOURCE_NAME

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    if len(cfg["logger"]) > 0 and list(cfg["logger"].keys())[0] == "wandb":
        cfg.logger.wandb.name += f"{cfg.model.arch_name}_{cfg.model.encoder_name}_scheduler_{cfg.model.scheduler._target_.split('.')[-1]}_lr_{str(cfg.model.optimizer.lr)}_batchsize_{str(cfg.datamodule.batch_size)}"

    if len(cfg["logger"]) > 0 and list(cfg["logger"].keys())[0] == "mlflow":
        cfg.logger.mlflow.run_name = f"{cfg.model.arch_name}_{cfg.model.encoder_name}_scheduler_{cfg.model.scheduler._target_.split('.')[-1]}_lr_{str(cfg.model.optimizer.lr)}_batchsize_{str(cfg.datamodule.batch_size)}"
        cfg.logger.mlflow.tags = {MLFLOW_USER: "주헌", "model": cfg.model.arch_name}

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics
    if "Smp" in cfg.datamodule._target_.split(".")[-1].lower():
        print("Segmentation end !")

    ckpt_path = trainer.checkpoint_callback.best_model_path

    if cfg.get("test"):
        log.info("Starting predicting!")

        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        log.info(f"Instantiating trainer <{cfg.test_trainer._target_}>")
        test_trainer: Trainer = hydra.utils.instantiate(
            cfg.test_trainer, callbacks=callbacks, logger=logger
        )
        predictions = test_trainer.predict(
            model=model, datamodule=datamodule, ckpt_path=ckpt_path
        )
        predictions = list(chain.from_iterable(predictions))

    now = time.strftime("%m%d_%H:%M")
    path_submit = f"../submit/{now}"

    os.makedirs(path_submit, exist_ok=True)
    os.chdir(path_submit)

    log.info("Starting making prediction files")

    utils.make_submit(predictions, path_submit)

    log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    return utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )


if __name__ == "__main__":
    main()
