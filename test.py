import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
from mlflow import (
    log_metric,
    log_param,
    log_artifact,
    set_experiment,
    create_experiment,
    start_run,
)

mlflow.set_tracking_uri("http://211.114.51.32:5000/")

if __name__ == "__main__":
    ARTIFACT_URI = "sftp://noops:zhflsdl3@211.114.51.32:5005/noops_storage/noops-mlflow-tracking-server/artifacts"
    EXPERIMENT_NAME = "example_experiment3"
    # create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_URI)
    set_experiment(EXPERIMENT_NAME)
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print(mlflow.get_tracking_uri())
    print(mlflow.get_artifact_uri())
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    # mlflow.log_artifacts("input/data/batch_01_vt", "hong")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
