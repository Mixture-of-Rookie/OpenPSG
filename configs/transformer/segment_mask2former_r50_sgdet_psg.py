_base_ = [
    '../motifs/segment_mask2former_r50_sgdet_psg.py',
]

expt_name = 'transformer_segment_panoptic_0.4_mask2former_r50_bs_8x2_det_sample'

model = dict(
    relation_head=dict(
        type='Mask2FormerTransformerHead',
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
    ),
    test_cfg=dict(
        object_mask_thr=0.4,
        max_per_image=30,
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
optimizer = dict(type='SGD', lr=0.02, momentum=0.9)

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
