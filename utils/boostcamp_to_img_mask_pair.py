import os
from torch.utils.data import Dataset
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from shutil import copyfile

category_names = [
    "Backgroud",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]

home_path = os.path.expanduser("~")
dataset_root = os.path.join(home_path, "input", "data")
dataset_list = [("trash_train", "train.json"), ("trash_val", "val.json")]


def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


class CustomDataLoader(Dataset):
    """COCO format"""

    def __init__(self, data_dir, output_root, transform=None):
        super().__init__()
        self.transform = transform
        self.coco = COCO(data_dir)
        self.output_root = output_root

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        image_from = os.path.join(dataset_root, image_infos["file_name"])
        image_filename = "_".join(image_infos["file_name"].split("/"))
        image_to = os.path.join(self.output_root, "image", image_filename)
        mask_filename = ".".join([image_filename.split(".")[0], "png"])
        mask_to = os.path.join(self.output_root, "mask", mask_filename)

        # cv2 를 활용하여 image 불러오기
        image = cv2.imread(os.path.join(dataset_root, image_infos["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
        anns = self.coco.loadAnns(ann_ids)

        # Load the categories in a variable
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)

        # masks : size가 (height x width)인 2D
        # 각각의 pixel 값에는 "category id" 할당
        # Background = 0
        mask = np.zeros((image_infos["height"], image_infos["width"]))
        # General trash = 1, ... , Cigarette = 10
        anns = sorted(anns, key=lambda idx: idx["area"], reverse=True)
        for i in range(len(anns)):
            className = get_classname(anns[i]["category_id"], cats)
            pixel_value = category_names.index(className)
            mask[self.coco.annToMask(anns[i]) == 1] = pixel_value
        mask = mask.astype(np.int8)

        # transform -> albumentations 라이브러리 활용
        assert self.transform is None
        return image_from, image_to, mask, mask_to

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


def convert():
    for output_dir_name, json_name in dataset_list:
        json_path = os.path.join(dataset_root, json_name)
        output_root = os.path.join(dataset_root, output_dir_name)
        output_image_dir = os.path.join(output_root, "image")
        output_mask_dir = os.path.join(output_root, "mask")
        create_dir(output_root)
        create_dir(output_image_dir)
        create_dir(output_mask_dir)

        dataset = CustomDataLoader(data_dir=json_path, output_root=output_root)

        for i in tqdm(range(len(dataset))):
            image_from, image_to, mask, mask_to = dataset[i]
            copyfile(image_from, image_to)
            cv2.imwrite(mask_to, mask)


if __name__ == "__main__":
    convert()
