_base_ = [
    '../_base_/models/mask2former_r50.py',
    '../_base_/datasets/psg.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/custom_runtime.py',
]

find_unused_parameters = True
custom_hooks = []


# HACK:
object_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged'
]

predicate_classes = [
    'over',
    'in front of',
    'beside',
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]


num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='SceneGraphMask2Former',
    relation_head=dict(
        type='Mask2FormerMotifHead',
        object_classes=object_classes,
        predicate_classes=predicate_classes,
        num_classes=len(object_classes) + 1,  # with background class
        num_predicates=len(predicate_classes) + 1,
        use_bias=False,  # NOTE: whether to use frequency bias
        head_config=dict(
            # NOTE: Evaluation type
            use_gt_box=False,
            use_gt_label=False,
            use_vision=True,
            embed_dim=200,
            hidden_dim=512,
            roi_dim=1024,
            context_pooling_dim=4096,
            dropout_rate=0.2,
            context_object_layer=1,
            context_edge_layer=1,
            glove_dir='data/glove/',
            causal_effect_analysis=False,
        ),
        relation_sampler=dict(
            type='Motif',
            pos_iou_thr=0.5,
            require_overlap=False,  # for sgdet training, not require
            num_sample_per_gt_rel=4,
            num_rel_per_image=1024,
            pos_fraction=0.25,
            # NOTE: To only include overlapping bboxes?
            test_overlap=False,  # for testing
        ),
        loss_object=dict(type='CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=1.0),
        loss_relation=dict(type='CrossEntropyLoss',
                           use_sigmoid=False,
                           loss_weight=1.0),
    ),
)

# Dataset settings
image_size = (1024, 1024)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='LoadPanopticSceneGraphAnnotations',
        with_bbox=True,
        with_rel=True,
        with_mask=True,
        with_seg=True,
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True,
    ),
    dict(
        type='RelRandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=False,
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(type='SceneGraphFormatBundle', img_to_float=True),
    dict(
        type='Collect',
        keys=[
            'img',
            'gt_bboxes',
            'gt_labels',
            'gt_rels',
            'gt_relmaps',
            'gt_masks',
            'gt_semantic_seg',
        ]
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline
    ),
    val=dict(
        pipeline=test_pipeline,
    ),
    test=dict(
        pipeline=test_pipeline,
    )
)

# To freeze modules
freeze_modules = [
    'backbone',
    'panoptic_head',
]

optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3,
                 step=[7, 10])

evaluation = dict(
    interval=1,
    metric='sgdet',
    relation_mode=True,
    classwise=True,
    iou_thrs=0.5,
    detection_method='pan_seg')

# Log config
project_name = 'ICME-2023'
expt_name = 'motifs_mask2former_r50_bs_8x2'
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

load_from = 'work_dirs/checkpoints/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'
