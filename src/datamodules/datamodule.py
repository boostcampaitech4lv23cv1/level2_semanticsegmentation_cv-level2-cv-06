from typing import Any, Dict, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from src.datamodules.dataset import SmpDataset
from src.datamodules.transform import AugmentTrain, AugmentValid, AugmentTest


class SmpDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/opt/ml/input/data/",
        # TODO fix this path
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        input_size: int = 256,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.pred_dataset: Optional[Dataset] = None

        self.train_transform = AugmentTrain(self.hparams.input_size)
        self.valid_transform = AugmentValid(self.hparams.input_size)
        self.test_transform = AugmentTest(self.hparams.input_size)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        if stage == "fit" or stage is None:

            self.train_dataset = SmpDataset(
                data_path=self.hparams.data_dir,
                transform=self.train_transform,
                mode="train",
            )
            self.val_dataset = SmpDataset(
                data_path=self.hparams.data_dir,
                transform=self.valid_transform,
                mode="val",
            )
        if stage == "predict" or stage is None:
            self.pred_dataset = SmpDataset(
                data_path=self.hparams.data_dir,
                transform=self.test_transform,
                mode="test",
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            # collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    # def collate_fn(self, batch):
    #     img, mask, img_infos = list(zip(*batch))
    #     batched_imgs = cat_list(images, fill_value=0)
    #     batched_targets = cat_list(targets, fill_value=255)

    #     return tuple(zip(*batch))


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "smp.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
