from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.config import get_cfg
import torch
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


args = default_argument_parser().parse_args()

checkpoint_paths = ["output/model_0079999.pth", "output/model_0074999.pth"]
model = DefaultTrainer.build_model(setup(args))
model.load_state_dict(torch.load(checkpoint_paths[0])["model"])
model_dict = model.state_dict()
soups = {key: [] for key in model_dict}

for i, checkpoint_path in enumerate(checkpoint_paths):
    # model=torch.load(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path)["model"])
    weight_dict = model.state_dict()
    for k, v in weight_dict.items():
        soups[k].append(v)
if 0 < len(soups):
    soups = {
        k: (torch.sum(torch.stack(v), axis=0) / len(v)).type(v[0].dtype)
        for k, v in soups.items()
        if len(v) != 0
    }
    model_dict.update(soups)
    model.load_state_dict(model_dict)
torch.save(model.state_dict(), "soup.pth")
