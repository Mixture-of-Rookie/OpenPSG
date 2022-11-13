_base_ = [
    '../motifs/panoptic_fpn_r50_fpn_1x_predcls_psg.py',
]

expt_name = 'transformer_panoptic_fpn_r50_bs_8x2_det_sample'

model = dict(
    relation_head=dict(
        type='TransformerHead',
        head_config=dict(
            # NOTE: Evaluation type
            use_gt_box=False,
            use_gt_label=False,
            dropout_rate=0.1,
            context_object_layer=4,
            context_edge_layer=2,
            num_head=8,
            k_dim=64,
            v_dim=64,
            inner_dim=1024,
        ),
        expt_name=expt_name,
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
optimizer = dict(type='SGD', lr=0.002, momentum=0.9)

# Log config
project_name = 'ICME-2023'
work_dir = f'./work_dirs/{expt_name}'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(
           type='WandbLoggerHook',
           init_kwargs=dict(
               project=project_name,
               name=expt_name,
               # config=work_dir + "/cfg.yaml"
           ),
        ),
    ],
)
