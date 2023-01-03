import os
import pickle
import sys

sys.path.insert(0, "Mask2Former")

from pathlib import Path
import numpy as np
import argparse
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image

# import Mask2Former project
from mask2former import add_maskformer2_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--weight', type=str, default='model_final.pth')
    arg = parser.parse_args()
    return arg


def setup_cfg(args):
    config_dir = os.path.join(args.model_dir, 'config.yaml')
    weight_dir = os.path.join(args.model_dir, args.weight)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_dir)
    cfg.MODEL.WEIGHTS = weight_dir
    cfg.freeze()
    return cfg


def main():
    args = get_parser()
    cfg = setup_cfg(args)
    with open('/opt/ml/input/data/test.json') as f:
        test_files = json.load(f)
    images = test_files['images']
    predictor = DefaultPredictor(cfg)
    outputs = []
    for index, image_info in enumerate(tqdm(images, total=len(images))):
        file_name = image_info['file_name']
        path = Path('/opt/ml/input/data') / file_name
        img = read_image(path, format="BGR")
        pred = predictor(img)
        output = pred['sem_seg'].unsqueeze(dim=0).detach().cpu().numpy().astype(np.float16)
        outputs.append(output)
    with open(f'./result_pkl/{args.model_dir}.pkl', 'wb') as f:
        pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
