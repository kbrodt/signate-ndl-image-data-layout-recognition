import argparse
import multiprocessing
import typing as t
from collections import defaultdict
from pathlib import Path

from ensemble_boxes import *
import tqdm
import numpy as np

from .utils import load_json, load_pkl, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--preds-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)

    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--skip_box_thr', type=float, default=0.0001)

    return parser.parse_args()


def get_bboxes(ann: t.List[np.ndarray], h: int, w: int) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray], t.List[int]]:
    bboxes_list, score_list, labels_list = [], [], []
    for cls, bbox in enumerate(ann[:-1]):
        if len(bbox) > 0:
            bbox[:, [0, 2]] = bbox[:, [0, 2]] / w
            bbox[:, [1, 3]] = bbox[:, [1, 3]] / h
            bboxes_list.extend(bbox[:, :4].tolist())
            score_list.extend(bbox[:, -1])
            labels_list.extend([cls] * len(bbox))

    return bboxes_list, score_list, labels_list


def ensemble_wbf(item):
    ann, *anns_ = item
    w, h = ann['width'], ann['height']
    boxes_list, scores_list, labels_list = [], [], []
    for ann_ in anns_:
        bboxes_list_, score_list_, labels_list_ = get_bboxes(ann_, h=h, w=w)
        boxes_list.append(bboxes_list_)
        scores_list.append(score_list_)
        labels_list.append(labels_list_)

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                                  iou_thr=args.iou_thr, skip_box_thr=args.skip_box_thr)
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * w
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * h
    tmp = defaultdict(list)
    for cls, bbox, sc in zip(labels, boxes, scores):
        tmp[idx_to_categories[int(cls)]].append(bbox.tolist())
        
    return ann['filename'], dict(tmp)


def main():
    global args
    global idx_to_categories

    args = parse_args()
    path_to_data = Path(args.data_path)

    train_anns = []
    for js in (path_to_data / 'train_annotations').rglob('*.json'):
         train_anns.append(load_json(js))

    categories = set()
    for x in train_anns:
        for l in x['labels']:
            categories.add(l['category'])
    idx_to_categories = sorted([cat for cat in categories])

    test_anns = load_pkl(path_to_data / 'test_anns.pkl')
    preds_path = Path(args.preds_path)
    test_preds = [load_pkl(path) for path in preds_path.rglob('*.pkl')]

    for p in test_preds:
        assert len(p) == len(test_anns)

    if len(test_preds) > 1:
        with multiprocessing.Pool() as p:
            with tqdm.tqdm(zip(test_anns, *test_preds), total=len(test_anns)) as pbar:
                submit = dict(list(p.imap_unordered(func=ensemble_wbf, iterable=pbar)))
    else:
        submit = {
            ann['filename']: {
                idx_to_categories[cls]: bbox[np.argsort(-bbox[:, -1]), :4].tolist()
                for cls, bbox in enumerate(preds[:-1])
                if len(bbox) > 0
            }
            for ann, preds in zip(test_anns, test_preds[0])
        }

    for k in submit:
        if '9_table' in submit[k]:
            del submit[k]['9_table']

    save_json(submit, args.save_path)


if __name__ == '__main__':
    main()
