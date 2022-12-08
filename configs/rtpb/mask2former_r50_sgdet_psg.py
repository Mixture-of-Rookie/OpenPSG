_base_ = [
    '../motifs/mask2former_r50_sgdet_psg.py',
]

expt_name = 'rtpb_mask2former_r50_bs_8x2_det_sample'

model = dict(
    relation_head=dict(
        type='Mask2FormerRTPBHead',
        head_config=dict(
            # NOTE: Evaluation type
            use_gt_box=False,
            use_gt_label=False,
            dropout_rate=0.1,
            test_nms_thres=0.5,
            obj_layer=4,
            rel_layer=2,
            num_head=8,
            key_dim=64,
            val_dim=64,
            inner_dim=2048,
            use_rel_graph=True,
            use_graph_encode=True,
            graph_encode_strategy='trans',
            use_bias=False,
            remove_bias=False,
        ),
    ),
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
