_base_ = [
    '../motifs/segment_mask2former_r50_sgdet_psg.py',
]

expt_name = 'd2str_segment_panoptic_0.8_mask2former_r50_bs_8x2_det_sample'

model = dict(
    relation_head=dict(
        type='D2STRHead',
        head_config=dict(
            # NOTE: Evaluation type
            use_gt_box=False,
            use_gt_label=False,
            obj_layer=4,
            rel_layer=2,
            obj_dim=512,
            rel_dim=1024,
            num_head=8,
            drop=0.1,
            attn_drop=0.1,
            drop_path=0.1,
            rel_mlp_ratio=4.,
        ),
    ),
    test_cfg=dict(
        object_mask_thr=0.8,
        max_per_image=10,
        postprocess='panoptic',
    )
)

evaluation = dict(interval=1,
                  metric='sgdet',
                  relation_mode=True,
                  classwise=True,
                  iou_thrs=0.5,
                  detection_method='pan_seg')

# Change batch size and learning rate
data = dict(samples_per_gpu=8,
            workers_per_gpu=2
            )
optimizer = dict(type='SGD', lr=0.01, momentum=0.9)

lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=1.0 / 3,
                 step=[7, 10])

# Log config
project_name = 'ICME-2023'
work_dir = f'./work_dirs/{expt_name}'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project=project_name,
                name=expt_name,
            ),
        ),
    ],
)
