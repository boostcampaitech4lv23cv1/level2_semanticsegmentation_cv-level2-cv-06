import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2


class AugmentTrain:
    def __init__(self, input_size: int):
        self.transform = self.train_transform(input_size)

    def train_transform(self, input_size):
        return Compose(
            [
                A.Resize(input_size, input_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)


class AugmentValid:
    def __init__(self, input_size: int):
        self.transform = self.test_transform(input_size)

    def test_transform(self, input_size):
        return Compose(
            [
                A.Resize(input_size, input_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)


class AugmentTest:
    def __init__(self, input_size: int):
        self.transform = self.test_transform(input_size)

    def test_transform(self, input_size):
        return Compose(
            [
                A.Resize(input_size, input_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def __call__(self, image):
        return self.transform(image=image)
