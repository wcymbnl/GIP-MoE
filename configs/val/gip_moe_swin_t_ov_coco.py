_base_ = '../grounding_dino_swin_t.py'  # noqa


model = dict(
    lmm=None,
    test_cfg=dict(max_per_img=300, chunked_size=-1)
)


dataset_type = 'CocoDataset'
data_root = '../grounding_data/coco/'
base_class_name = (
    'person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat',
    'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe',
    'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite',
    'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair',
    'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave', 'oven',
    'toaster', 'refrigerator', 'book', 'clock', 'vase', 'toothbrush')
novel_class_name = (
    'airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie',
    'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard',
    'sink', 'scissors')
class_name = base_class_name + novel_class_name
metainfo = dict(
    classes=class_name,
    base_classes=base_class_name,
    novel_classes=novel_class_name)

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')

dataset = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file='annotations/instances_val2017.65.json',
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True)
val_evaluator = dict(
    _delete_=True,
    type='OVCocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.65.json',
    metric='bbox',
    metric_items=['mAP', 'mAP_50'])


val_dataloader = dict(dataset=dataset)
test_dataloader = val_dataloader

test_evaluator = val_evaluator
