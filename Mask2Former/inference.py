import sys

sys.path.insert(0, "Mask2Former")

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from typing import List, Optional, Tuple

import albumentations as A
import cv2
import torch
import torch.nn as nn

# import Mask2Former project
from mask2former import add_maskformer2_config
from mlflow_config import add_mlflow_config
from torch.utils.data import DataLoader, Dataset

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
    add_mlflow_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weight
    cfg.freeze()
    return cfg


class ImageDataset(Dataset):
    def __init__(self, image: List[Path]):
        self.image = image

    def __getitem__(self, index) -> np.ndarray:
        return self.image[index]

    def __len__(self):
        return len(self.image)


class BatchPredictor(DefaultPredictor):
    """Run batch inference with D2"""

    def __collate(self, batch):
        data = []
        for image in batch:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]

            image = self.aug.get_transform(image).apply_image(image)
            image = image.astype("float32").transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data.append({"image": image, "height": height, "width": width})
        return data

    def __call__(self, img_infos: List[dict]) -> List[dict]:
        """Run d2 on a list of images.

        Args:
            images (list): BGR images of the expected shape: 720x1280
        """

        images = []
        image_id = []
        for img_info in img_infos:
            path = Path("/opt/ml/input/data") / img_info["file_name"]
            img = read_image(path, format="BGR")
            images.append(img)
            image_id.append(img_info["file_name"])
        dataset = ImageDataset(images)
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=self.__collate,
            pin_memory=True,
        )
        size = 256
        transform = A.Resize(
            size, size, p=1.0, always_apply=True, interpolation=cv2.INTER_NEAREST
        )
        preds_array = np.empty((0, size**2), dtype=np.long)
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader)):
                outs = self.model(batch)
                outs = torch.cat(
                    [out["sem_seg"].unsqueeze(0) for out in outs], 0
                ).argmax(1)
                outs = outs.detach().cpu().numpy()
                for out in outs:
                    out = out.astype(np.uint8)
                    mask = transform(image=out)["image"]
                    oms = mask.reshape([1, -1]).astype(int)
                    preds_array = np.vstack((preds_array, oms))
                del mask, oms, outs
                torch.cuda.empty_cache()

        return image_id, preds_array


def main():
    args = get_parser()
    cfg = setup_cfg(args)
    with open("/opt/ml/input/data/test.json") as f:
        test_files = json.load(f)
    images = test_files["images"]
    predictor = BatchPredictor(cfg)
    size = 256
    image_id = []
    preds_array = np.empty((0, size**2), dtype=np.long)
    print("Start inference")
    image_id, preds_array = predictor(images)
    print("End inference")
    submission = pd.read_csv(
        "/opt/ml/input/code/submission/sample_submission.csv", index_col=None
    )
    print("Start submission")
    for file_name, string in zip(tqdm(image_id), preds_array):
        submission = submission.append(
            {
                "image_id": file_name,
                "PredictionString": " ".join(map(str, string.tolist())),
            },
            ignore_index=True,
        )
    create_dir("./submission")
    weight_nm = cfg.MODEL.WEIGHTS.split("/")[-1].split(".")[0]
    regex_nm = re.findall("\d+", weight_nm)
    nm = regex_nm[0].lstrip("0") or f"{weight_nm}"
    sub_name = cfg.MODEL.BACKBONE.NAME[2:] + "_iter_" + nm
    submission.to_csv(f"./submission/{sub_name}.csv", index=False)
    print("End submission! file name: ", f"{sub_name}.csv")


def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


if __name__ == "__main__":
    main()
