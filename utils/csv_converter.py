import cv2
import numpy as np
from tqdm import tqdm
import csv
import os
import sys
csv.field_size_limit(sys.maxsize)
 

def submission_to_data(submission_path: str):
    '''submission.csv의 prediction을 image, mask 파일로 생성
    Args:
        submission_path (str): submission.csv file path

    '''
    home_path = os.path.expanduser("~")
    root = os.path.join(home_path, "input", "data", "trash_test", "masks")
    size = 512
    with open(submission_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in tqdm(reader):
            image_filename = row[0]
            mask = np.array(list(map(int, row[1].split()))).reshape([size, size])
            mask_filename = ".".join([image_filename.split(".")[0], "png"])
            mask_filename = mask_filename.replace("/", "_")
            mask_to = os.path.join(root, mask_filename)
            cv2.imwrite(mask_to, mask)

        

if __name__ == "__main__":
    submission_to_data("/opt/ml/input/ensemble_all3.csv")
    
