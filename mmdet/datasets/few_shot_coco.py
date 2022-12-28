# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
import os
import copy
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
import json
from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from .coco import CocoDataset

BASE_CLASSES = {
    1:[
        'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
        'traffic light', 'fire hydrant', 'stop sign', 'bench', 'bird',
        'cat', 'horse', 'sheep', 'cow', 'bear', 'zebra', 'giraffe',
        'umbrella', 'handbag', 'tie', 'frisbee', 'skis', 'snowboard',
        'kite', 'baseball bat', 'baseball glove', 'surfboard',
        'tennis racket', 'bottle', 'cup', 'fork', 'knife', 'bowl', 'banana',
        'apple', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'cake',
        'couch', 'potted plant', 'bed', 'toilet', 'tv', 'laptop', 'remote',
        'keyboard', 'cell phone', 'oven', 'toaster', 'sink', 'book',
        'clock', 'vase', 'teddy bear', 'hair drier', 'toothbrush'
    ],
    2:[
        'person', 'car', 'motorcycle', 'airplane', 'train', 'truck', 'boat',
        'fire hydrant', 'stop sign', 'parking meter', 'bird', 'cat', 'dog',
        'sheep', 'cow', 'elephant', 'zebra', 'giraffe', 'backpack',
        'handbag', 'tie', 'suitcase', 'skis', 'snowboard', 'sports ball',
        'baseball bat', 'baseball glove', 'skateboard', 'tennis racket',
        'bottle', 'wine glass', 'fork', 'knife', 'spoon', 'banana', 'apple',
        'sandwich', 'broccoli', 'carrot', 'hot dog', 'donut', 'cake',
        'chair', 'potted plant', 'bed', 'dining table', 'tv', 'laptop',
        'mouse', 'keyboard', 'cell phone', 'microwave', 'toaster', 'sink',
        'refrigerator', 'clock', 'vase', 'scissors', 'hair drier',
        'toothbrush'
    ],
    3:[
        'person', 'bicycle', 'motorcycle', 'airplane', 'bus', 'truck', 'boat', 'traffic light',
        'stop sign', 'parking meter', 'bench', 'cat', 'dog',
        'horse', 'cow', 'elephant', 'bear', 'giraffe',
        'backpack', 'umbrella', 'tie', 'suitcase', 'frisbee', 'snowboard', 'sports ball', 'kite',
        'baseball glove', 'skateboard', 'surfboard',
        'bottle', 'wine glass', 'cup', 'knife', 'spoon', 'bowl', 'apple', 'sandwich', 'orange', 'carrot',
        'hot dog', 'pizza', 'cake', 'chair', 'couch', 'bed', 'dining table', 'toilet', 'laptop',
        'mouse', 'remote', 'cell phone', 'microwave',
        'oven', 'sink', 'refrigerator', 'book',
        'vase', 'scissors', 'teddy bear', 'toothbrush'
    ],
    4:[
        'person', 'bicycle', 'car', 'airplane', 'bus', 'train', 'boat',
        'traffic light', 'fire hydrant', 'parking meter', 'bench', 'bird',
        'dog', 'horse', 'sheep', 'elephant', 'bear', 'zebra', 'backpack',
        'umbrella', 'handbag', 'suitcase', 'frisbee', 'skis', 'sports ball',
        'kite', 'baseball bat', 'skateboard', 'surfboard', 'tennis racket',
        'wine glass', 'cup', 'fork', 'spoon', 'bowl', 'banana', 'sandwich',
        'orange', 'broccoli', 'hot dog', 'pizza', 'donut', 'chair', 'couch',
        'potted plant', 'dining table', 'toilet', 'tv', 'mouse', 'remote',
        'keyboard', 'microwave', 'oven', 'toaster', 'refrigerator', 'book',
        'clock', 'scissors', 'teddy bear', 'hair drier'
    ],
    'voc':[
        'truck', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake',
        'bed', 'toilet', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
}

NOVEL_CLASSES = {
    1: [
        'person', 'airplane', 'boat', 'parking meter', 'dog', 'elephant',
        'backpack', 'suitcase', 'sports ball', 'skateboard', 'wine glass',
        'spoon', 'sandwich', 'hot dog', 'chair', 'dining table', 'mouse',
        'microwave', 'refrigerator', 'scissors'
    ],
    2:[
        'bicycle', 'bus', 'traffic light', 'bench', 'horse', 'bear', 'umbrella',
        'frisbee', 'kite', 'surfboard', 'cup', 'bowl', 'orange', 'pizza',
        'couch', 'toilet', 'remote', 'oven', 'book', 'teddy bear'
    ],
    3:[
        'car', 'train', 'fire hydrant', 'bird', 'sheep', 'zebra', 'handbag', 
        'skis', 'baseball bat', 'tennis racket', 'fork', 'banana', 'broccoli', 
        'donut', 'potted plant', 'tv', 'keyboard', 'toaster', 'clock', 'hair drier'
    ],
    4:[
        'motorcycle', 'truck', 'stop sign', 'cat', 'cow', 'giraffe', 'tie', 
        'snowboard', 'baseball glove', 'bottle', 'knife', 'apple', 'carrot', 
        'cake', 'bed', 'laptop', 'cell phone', 'sink', 'vase', 'toothbrush'
    ],
    'voc':[
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'boat',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 'couch',
        'potted plant', 'dining table', 'tv'
    ]
}
@DATASETS.register_module()
class FewShotCocoDataset(CocoDataset):

    def __init__(self, ann_file, shot, split, seed, **kwargs):
        self.shot = shot
        self.split = split
        self.seed = seed
        self.base_classes = BASE_CLASSES[split]
        self.novel_classes = NOVEL_CLASSES[split]
        self.CLASSES = self.base_classes + self.novel_classes
        self.ori_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        super(CocoDataset, self).__init__(ann_file=ann_file, 
                    classes=self.base_classes + self.novel_classes, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        # split_dir = 'data/datasets/cocosplit'
        # seedfn = f'seed{self.seed}'
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES


        # name = self.concat_json(split_dir, seedfn)
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {}
        for i, cat_id in enumerate(self.cat_ids):
            self.cat2label[cat_id] = self.CLASSES.index(self.ori_classes[i])   
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        # assert len(set(total_ann_ids)) == len(
        #     total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def concat_json(self, split_dir, seedfn):
        img_set = []
        for i, cls in enumerate(self.novel_classes):
            fn = f'full_box_{self.shot}shot_{cls}_trainval.json'
            f = open(os.path.join(split_dir, seedfn, fn))
            if i == 0:
                json_data = json.load(f)
                for img in json_data['images']:
                    img_set.append(img['id'])
            else:
                add_data = json.load(f)
                for img in add_data['images']:
                    if not img['id'] in img_set:
                        json_data['images'].append(img)
                for ann in add_data['annotations']:
                    json_data['annotations'].append(ann)
        name = f'full_box_{self.shot}shot_split{self.split}_trainval.json'
        new_file = open(name, 'w')
        json.dump(json_data, new_file)
        return name


@DATASETS.register_module()
class FewShotTestCocoDataset(CocoDataset):

    def __init__(self, ann_file, split, **kwargs):
        self.split = split
        self.base_classes = BASE_CLASSES[split]
        self.novel_classes = NOVEL_CLASSES[split]
        self.CLASSES = self.base_classes + self.novel_classes
        self.ori_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        super(CocoDataset, self).__init__(ann_file=ann_file, 
                    classes=self.base_classes + self.novel_classes, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.ori_cat_ids = copy.deepcopy(self.cat_ids)
        self.cat2label = {}
        for i, cat_id in enumerate(self.cat_ids):
            self.cat2label[cat_id] = self.CLASSES.index(self.ori_classes[i])
        self.cat_ids = [0] * len(self.CLASSES)
        for key in self.cat2label:
            self.cat_ids[self.cat2label[key]] = key
        
        # for key in self.coco.anns:
        #     print(self.coco.anns[key]['category_id'])
        #     print(self.CLASSES[self.cat2label[self.coco.anns[key]['category_id']]])
        #     print(self.CLASSES.index('microwave'))
        #     self.coco.anns[key]['category_id'] = self.CLASSES.index(self.ori_classes[self.coco.anns[key]['category_id']])
        # self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 novel=True):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco

        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.ori_cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if novel:
                    precisions = cocoEval.eval['precision']
                    assert len(self.cat_ids) == precisions.shape[2]
                    novel_category = []
                    base_category = []
                    results_per_novel_base = []
                    for idx, catId in enumerate(self.ori_cat_ids):
                        if self.CLASSES[self.cat2label[catId]] in self.novel_classes:
                            novel_category.append(idx)
                        else:
                            base_category.append(idx)
                            
                    precision = precisions[:, :, novel_category, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_novel_base.append(
                            ('novel', f'{float(ap):0.3f}'))
                    
                    precision = precisions[0, :, novel_category, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_novel_base.append(
                            ('novel AP50', f'{float(ap):0.3f}'))

                    precision = precisions[:, :, base_category, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_novel_base.append(
                            ('base', f'{float(ap):0.3f}'))     

                    precision = precisions[0, :, base_category, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_novel_base.append(
                            ('base AP50', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_novel_base) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_novel_base))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)


                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
