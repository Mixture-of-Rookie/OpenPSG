_base_ = [
    '../motifs/segment_mask2former_r50_sgdet_psg.py',
]

expt_name = 'vctree_segment_instance_30_mask2former_r50_bs_8x2'

model = dict(
    relation_head=dict(
        type='Mask2FormerVCTreeHead',
        head_config=dict(
            # NOTE: Evaluation type
            use_gt_box=False,
            use_gt_label=False,
        ),
        expt_name=expt_name,
    ),
    test_cfg=dict(
        object_mask_thr=0.8,
        max_per_image=30,
        postprocess='instance',
    )
)

evaluation = dict(
    interval=1,
    metric='sgdet',
    relation_mode=True,
    classwise=True,
    iou_thrs=0.5,
    detection_method='pan_seg',
)

# Change batch size and learning rate
data = dict(samples_per_gpu=8,
            workers_per_gpu=2)

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
