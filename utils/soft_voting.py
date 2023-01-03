import warnings
from glob import glob

warnings.filterwarnings('ignore')

import numpy as np
import pickle
import albumentations as A
import json
import pandas as pd
from tqdm import tqdm

pkls = glob('./result_pkl/*.pkl')
transform = A.Compose([A.Resize(256, 256)])


def main():
    with open('../data/test.json') as f:
        test_files = json.load(f)
    images = test_files['images']

    n = len(pkls)
    output = np.zeros((624, 11, 512, 512), dtype=np.float16)
    for pkl in pkls:
        output += np.array(pickle.load(open(pkl, 'rb')))
    output /= n
    output = np.argmax(output, axis=1)

    size = 256
    image_id = []
    preds_array = np.empty((0, size * size), dtype=np.long)
    for image, mask in tqdm(zip(images, output), total=624):
        file_name = image['file_name']
        image_id.append(file_name)
        temp = np.zeros((1, 1, 1))
        transformed = transform(image=temp, mask=mask)
        temp_mask = []
        transformed_mask = transformed['mask']
        temp_mask.append(transformed_mask)

        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], size * size]).astype(int)
        preds_array = np.vstack((preds_array, oms))

    submission = pd.read_csv('../code/submission/sample_submission.csv', index_col=None)

    for file_name, string in zip(image_id, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)

    submission.to_csv(f"./soft_voting.csv", index=False)


if __name__ == "__main__":
    main()
