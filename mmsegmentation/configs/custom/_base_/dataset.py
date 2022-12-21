# dataset settings
dataset_type = "CustomDataset"
data_train = "/opt/ml/aistage/data/trash_train"
data_val = "/opt/ml/aistage/data/trash_val"
data_test = "/opt/ml/aistage/trash_test_mmseg"
classes = (
    "Background",
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
)

classes = [
    "Background",
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
palette = [
    [0, 0, 0],
    [192, 0, 128],
    [0, 128, 192],
    [0, 128, 64],
    [128, 0, 0],
    [64, 0, 128],
    [64, 0, 192],
    [192, 128, 64],
    [192, 192, 128],
    [64, 64, 128],
    [128, 0, 192],
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375], to_rgb=True
)
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        palette=palette,
        img_dir=data_train + "/images",
        ann_dir=data_train + "/masks",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        palette=palette,
        img_dir=data_val + "/images",
        ann_dir=data_val + "/masks",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        palette=palette,
        img_dir=data_test + "/images",
        pipeline=test_pipeline,
    ),
)


evaluation = dict(interval=1, metric="mIoU", classwise=True, save_best="mIoU")
