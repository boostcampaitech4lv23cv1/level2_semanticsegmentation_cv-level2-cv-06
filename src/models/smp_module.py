from typing import Any, List
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, JaccardIndex
from torchmetrics import Accuracy, Precision, Recall
import albumentations as A
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp


class SmpModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        arch_name="Unet",
        encoder_name="resnet18",
        loss_type="mae",
        module_type="depth_map",
        activation="sigmoid",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = smp.create_model(
            arch=self.hparams.arch_name,
            encoder_name=self.hparams.encoder_name,
            activation=self.hparams.activation,
            classes=11,
        )
        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_iou = JaccardIndex(num_classes=11, reduction=None)
        self.val_iou = JaccardIndex(num_classes=11, reduction=None)

        self.iou = JaccardIndex(num_classes=11, reduction="none")
        self.accuracy = Accuracy(num_classes=11)
        self.precision_ = Precision(
            num_classes=11, average="none", mdmc_average="global"
        )
        self.recall = Recall(num_classes=11, average="none", mdmc_average="global")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for logging best so far validation accuracy
        self.val_miou_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x.float())

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_miou_best.reset()

    def step(self, batch: Any):
        img, mask = batch
        logits = self.forward(img)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

        loss = self.criterion(logits, mask.long())
        return mask, logits, probs, pred, loss

    def phase_step(self, batch, batch_idx, phase):
        mask, logits, probs, pred, loss = self.step(batch)

        ious = self.iou(probs, mask)
        acc = self.accuracy(probs, mask)
        precisions = self.precision_(probs, mask)
        recalls = self.recall(probs, mask)

        # Filter nan values before averaging
        miou = ious[~ious.isnan()].mean()
        avg_precision = precisions[~precisions.isnan()].mean()
        avg_recall = recalls[~recalls.isnan()].mean()

        # Log metrics
        self.log(f"{phase}/loss", loss, sync_dist=True)
        self.log(f"{phase}/miou", miou, sync_dist=True),
        self.log(f"{phase}/acc", acc, sync_dist=True)
        self.log(f"{phase}/aP", avg_precision, sync_dist=True)
        self.log(f"{phase}/aR", avg_recall, sync_dist=True)

        label2cat = {
            0: "Background",
            1: "General_trash",
            2: "Paper",
            3: "Paper_pack",
            4: "Metal",
            5: "Glass",
            6: "Plastic",
            7: "Styrofoam",
            8: "Plastic_bag",
            9: "Battery",
            10: "Clothing",
        }

        for c in range(11):
            cls = label2cat[c]
            if not ious[c].isnan():
                self.log(f"{phase}_{cls}_iou", ious[c], sync_dist=True)
            if not precisions[c].isnan():
                self.log(f"{phase}_{cls}_precision", precisions[c], sync_dist=True)
            if not recalls[c].isnan():
                self.log(f"{phase}_{cls}_recall", recalls[c], sync_dist=True)

        return loss, logits, mask, pred

    def training_step(self, batch: Any, batch_idx: int):
        loss, logits, mask, pred = self.phase_step(batch, batch_idx, "train")
        self.log("LearningRate", self.optimizer.param_groups[0]["lr"])

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "logits": logits, "mask": mask}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # iou_per_class = self.train_iou.compute().detach()
        # miou = iou_per_class.mean()
        # self.log("train/miou", miou, on_step=False, on_epoch=True, prog_bar=True)
        # self.train_iou.reset()
        pass

    def validation_step(self, batch: Any, batch_idx: int):

        loss, logits, mask, pred = self.phase_step(batch, batch_idx, "val")

        return {"loss": loss, "logits": logits, "mask": mask}

    def validation_epoch_end(self, outputs: List[Any]):
        # iou_per_class = self.val_iou.compute().detach()
        # miou = iou_per_class.mean()
        # self.val_miou_best(miou)

        # self.log(
        #     "val/best_miou",
        #     self.val_miou_best.compute(),
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        # self.log(
        #     "val/miou",
        #     miou,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        size = 256
        transform = A.Compose([A.Resize(size, size)])

        imgs, file_name = batch
        outputs = self.forward(imgs)
        outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        temp_mask = []
        for img, mask in zip(imgs.detach().cpu().numpy(), outputs):
            mask = transform(image=img, mask=mask)["mask"]
            temp_mask.append(mask)
        outputs = np.array(temp_mask)
        outputs = outputs.reshape([outputs.shape[0], size**2]).astype(int)
        return outputs, file_name

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        self.optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=self.optimizer)

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/miou",
                "interval": "epoch",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "smp.yaml")
    _ = hydra.utils.instantiate(cfg)
