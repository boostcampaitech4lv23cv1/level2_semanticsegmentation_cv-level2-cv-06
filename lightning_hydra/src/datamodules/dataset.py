from typing import Any, Dict, Optional, Tuple
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import os.path as osp
import numpy as np


cat2label = {
    "Background": 0,
    "General trash": 1,
    "Paper": 2,
    "Paper pack": 3,
    "Metal": 4,
    "Glass": 5,
    "Plastic": 6,
    "Styrofoam": 7,
    "Plastic bag": 8,
    "Battery": 9,
    "Clothing": 10,
}


class SmpDataset(Dataset):
    def __init__(self, data_path: str, mode="train", transform=None):
        """
        Parameters:
            data_path (string): data directory path
            mode (string): (train, val, test)
            transform (A.transform): transform
        """
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.coco = COCO(osp.join(data_path, f"{mode}.json"))

    def __getitem__(self, idx: int):
        img_id = self.coco.getImgIds(idx)
        img_infos = self.coco.loadImgs(img_id)[0]

        img_path = osp.join(self.data_path, img_infos["file_name"])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == "train":
            ann_ids = self.coco.getAnnIds(imgIds=img_infos["id"])
            ann_infos = self.coco.loadAnns(ann_ids)

            cat_ids = self.coco.getCatIds()
            cat_infos = self.coco.loadCats(cat_ids)

            mask = np.zeros((img_infos["height"], img_infos["width"]))
            ann_infos = sorted(
                ann_infos, key=lambda ann_info: ann_info["area"], reverse=True
            )

            for ann_info in ann_infos:
                binary_mask = self.coco.annToMask(ann_info)

                ann_cat_id = ann_info["category_id"]
                ann_cat_name = self.get_classname(ann_cat_id, cat_infos)
                pixel_value = cat2label[ann_cat_name]
                mask[binary_mask == 1] = pixel_value

            mask = mask.astype(np.uint8)

            if self.transform is not None:
                transformed = self.transform(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"]

            return img, mask

        if self.mode == "val":
            ann_ids = self.coco.getAnnIds(imgIds=img_infos["id"])
            ann_infos = self.coco.loadAnns(ann_ids)

            cat_ids = self.coco.getCatIds()
            cat_infos = self.coco.loadCats(cat_ids)

            mask = np.zeros((img_infos["height"], img_infos["width"]))
            ann_infos = sorted(
                ann_infos, key=lambda ann_info: ann_info["area"], reverse=True
            )

            for ann_info in ann_infos:
                binary_mask = self.coco.annToMask(ann_info)

                ann_cat_id = ann_info["category_id"]
                ann_cat_name = self.get_classname(ann_cat_id, cat_infos)
                pixel_value = cat2label[ann_cat_name]
                mask[binary_mask == 1] = pixel_value

            mask = mask.astype(np.uint8)

            if self.transform is not None:
                transformed = self.transform(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"]

            return img, mask

        if self.mode == "test":
            if self.transform is not None:
                transformed = self.transform(image=img)
                img = transformed["image"]

            return img, img_infos["file_name"]

    def __len__(self):
        return len(self.coco.getImgIds())

    def get_classname(self, cat_id, cats):
        return [x["name"] for x in cats if x["id"] == cat_id][0]
