import sys

import cv2

sys.path.insert(0, "Mask2Former")

from pathlib import Path
import numpy as np
import argparse
import json
from tqdm import tqdm
import albumentations as A
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image

# import Mask2Former project
from mask2former import add_maskformer2_config
from mlflow_config import add_mlflow_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--weight", type=str)
    arg = parser.parse_args()
    return arg


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
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
    preds_array = np.empty((0, size * size), dtype=np.long)
    for index, image_info in enumerate(tqdm(images, total=len(images))):
        file_name = image_info["file_name"]
        image_id.append(file_name)
        path = Path("/opt/ml/input/data") / file_name
        img = read_image(path, format="BGR")
        pred = predictor(img)
        output = pred["sem_seg"].argmax(dim=0).detach().cpu().numpy()
        temp_mask = []
        temp_img = np.zeros((3, 512, 512))
        transformed = transform(image=temp_img, mask=output)
        mask = transformed["mask"]
        temp_mask.append(mask)

        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], size * size]).astype(int)
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
    submission.to_csv("./submission/no_back_submission.csv", index=False)


def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


if __name__ == "__main__":
    main()
