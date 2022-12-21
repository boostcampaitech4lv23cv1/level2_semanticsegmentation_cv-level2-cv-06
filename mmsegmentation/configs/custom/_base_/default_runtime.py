# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        #     # dict(type='TensorboardLoggerHook')
        #     # dict(type='PaviLoggerHook') # for internal services
        #     dict(
        #         type="MlflowLoggerHook",
        #         exp_name="upernet_swin_tiny",
        #         uri="http://211.114.51.32:5000",
        #     ),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True

checkpoint_config = dict(interval=1)
