import sys

import cv2

sys.path.insert(0, "Mask2Former")

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# import Mask2Former project
from mask2former import add_maskformer2_config
from mlflow_config import add_mlflow_config
from train_dinat import add_dinat_config

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--weight", type=str)
    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_dinat_config(cfg)  # this is only for dinat model
    add_mlflow_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weight
    cfg.freeze()
    return cfg


def main():
    args = get_parser()
    cfg = setup_cfg(args)
    with open("/opt/ml/input/data/test.json") as f:
        test_files = json.load(f)
    images = test_files["images"]
    predictor = DefaultPredictor(cfg)
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    image_id = []
    preds_array = np.empty((0, size**2), dtype=np.long)
    for image_info in tqdm(images, total=len(images)):
        file_name = image_info["file_name"]
        image_id.append(file_name)
        path = Path("/opt/ml/input/data") / file_name
        img = read_image(path, format="BGR")
        pred = predictor(img)
        output = pred["sem_seg"].argmax(dim=0).detach().cpu().numpy()
        temp_img = np.zeros((3, 512, 512))
        transformed = transform(image=temp_img, mask=output)
        mask = transformed["mask"]
        temp_mask = [mask]
        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], size**2]).astype(int)
        preds_array = np.vstack((preds_array, oms))

    submission = pd.read_csv(
        "/opt/ml/input/code/submission/sample_submission.csv", index_col=None
    )

    for file_name, string in zip(image_id, preds_array):
        submission = submission.append(
            {
                "image_id": file_name,
                "PredictionString": " ".join(str(e) for e in string.tolist()),
            },
            ignore_index=True,
        )
    create_dir("./submission")
    sub_name = (
        cfg.MODEL.BACKBONE.NAME[2:]
        + "_iter_"
        + re.findall("\d+", cfg.MODEL.WEIGHTS.split("/")[-1])[0].lstrip("0")
    )
    submission.to_csv(f"./submission/{sub_name}.csv", index=False)


def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


if __name__ == "__main__":
    main()
