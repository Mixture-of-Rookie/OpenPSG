_base_ = [
    '../motifs/panoptic_fpn_r50_fpn_1x_predcls_psg.py',
]

expt_name = 'd2str_panoptic_fpn_r50_bs_8x2_det_sample'

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
            rel_mlp_ratio=1.,
        ),
        loss_attention=dict(type='AttnMarginLoss'),
    ),
    roi_head=dict(bbox_head=dict(type='SceneGraphBBoxHead'), ),
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
