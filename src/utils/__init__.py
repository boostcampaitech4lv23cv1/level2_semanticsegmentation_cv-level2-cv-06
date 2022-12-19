from src.utils.pylogger import get_pylogger
from src.utils.rich_utils import enforce_tags, print_config_tree
from src.utils.utils import (
    close_loggers,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    save_file,
    task_wrapper,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def get_confmat(df):
    df = pd.DataFrame(df.detach().cpu().numpy())
    plt.clf()  # ADD THIS LINE
    plt.figure(figsize=(10, 10))
    confmat_heatmap = sns.heatmap(
        data=df,
        cmap="RdYlGn",
        annot=True,
        fmt=".3f",
        cbar=False,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted label")

    confmat_heatmap.xaxis.set_label_position("top")
    plt.yticks(rotation=0)
    confmat_heatmap.tick_params(axis="x", which="both", bottom=False)

    return confmat_heatmap.get_figure()
