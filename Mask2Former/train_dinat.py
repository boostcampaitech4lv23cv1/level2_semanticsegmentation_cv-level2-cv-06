"""
Mask2Former training script + DiNAT as a backbone.

Dilated Neighborhood Attention Transformer.
https://arxiv.org/abs/2209.15001

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import detectron2.utils.comm as comm
from detectron2.engine import (
    default_argument_parser,
    launch,
    default_setup,
)
from detectron2.config import CfgNode, get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import add_maskformer2_config

# from register_trash_dataset import register_all_trash_full
from register_trash_dataset_noops import register_all_trash_full
from mlflow_config import add_mlflow_config
from train_net_mlflow import Trainer
from dinat import *
from hooks_mlflow import MLflowHook
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import verify_results


def add_dinat_config(cfg):
    cfg.MODEL.DINAT = CfgNode()
    cfg.MODEL.DINAT.EMBED_DIM = 192
    cfg.MODEL.DINAT.DEPTHS = [3, 4, 18, 5]
    cfg.MODEL.DINAT.NUM_HEADS = [6, 12, 24, 48]
    cfg.MODEL.DINAT.KERNEL_SIZE = 7
    cfg.MODEL.DINAT.DILATIONS = None
    cfg.MODEL.DINAT.MLP_RATIO = 2.0
    cfg.MODEL.DINAT.QKV_BIAS = True
    cfg.MODEL.DINAT.QK_SCALE = None
    cfg.MODEL.DINAT.DROP_RATE = 0.0
    cfg.MODEL.DINAT.ATTN_DROP_RATE = 0.0
    cfg.MODEL.DINAT.DROP_PATH_RATE = 0.3
    cfg.MODEL.DINAT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]


def setup(args):
    """
    Modified version of the original;
    Just adds DiNAT args.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_dinat_config(cfg)
    add_mlflow_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former"
    )
    return cfg


def main(args):
    cfg = setup(args)

    mlflow_hook = MLflowHook(cfg)

    if args.eval_only:
        return test_mode(cfg, args)
    trainer = Trainer(cfg)
    trainer.register_hooks(hooks=[mlflow_hook])
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def test_mode(cfg, args):
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
    return res


if __name__ == "__main__":
    register_all_trash_full()
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
