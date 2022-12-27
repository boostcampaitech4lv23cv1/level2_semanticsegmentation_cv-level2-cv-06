from detectron2.engine import HookBase
import mlflow
import torch
import os
from detectron2.config.config import CfgNode


class MLflowHook(HookBase):
    """
    A custom hook class that logs artifacts, metrics, and parameters to MLflow.
    """

    def param_logger(self, base_name_list, cfg):
        for k, v in cfg.items():
            if type(v) == CfgNode:
                self.param_logger(base_name_list + [k], v)
            else:
                mlflow.log_param(".".join(base_name_list + [k]), v)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

    def before_train(self):
        with torch.no_grad():
            mlflow.set_tracking_uri(self.cfg.MLFLOW.TRACKING_URI)

            existing_exp = mlflow.get_experiment_by_name(
                self.cfg.MLFLOW.EXPERIMENT_NAME
            )
            if not existing_exp:
                mlflow.create_experiment(
                    self.cfg.MLFLOW.EXPERIMENT_NAME,
                    artifact_location=self.cfg.MLFLOW.ARTIFACT_URI,
                )

            mlflow.set_experiment(self.cfg.MLFLOW.EXPERIMENT_NAME)
            mlflow.start_run(run_name=self.cfg.MLFLOW.RUN_NAME)
            mlflow.set_tag("mlflow.note.content", self.cfg.MLFLOW.RUN_DESCRIPTION)

            self.param_logger([], self.cfg)

    def after_step(self):
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()
            output_metrics = {}
            for k, v in latest_metrics.items():
                output_metrics[k] = v[0]
            mlflow.log_metrics(metrics=output_metrics, step=v[1])

    def after_train(self):
        with torch.no_grad():
            with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
                f.write(self.cfg.dump())
            mlflow.log_artifacts(self.cfg.OUTPUT_DIR)
        mlflow.end_run()
