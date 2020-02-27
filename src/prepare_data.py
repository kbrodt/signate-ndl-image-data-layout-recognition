import argparse
import typing as t
from pathlib import Path
import multiprocessing

import numpy as np
import PIL.Image as Image
import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from .utils import load_json, save_pkl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'test'])
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=314159, required=False)
    parser.add_argument('--n-splits', type=int, default=5, required=False)

    return parser.parse_args()


def to_mmdet_trian(path: t.Union[str, Path]) -> t.Dict[str, t.Any]:
    img = np.array(Image.open(path))
    height, width = img.shape[:2]
    ann = load_json(path.parent.parent / 'train_annotations' / (path.stem + '.json'))

    bboxes, labels = [], []
    for x in ann['labels']:
        box2d = x['box2d']
        category = x['category']
        label = int(category.split('_')[0])
        bboxes.append([box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']])
        labels.append(label)

    return {
        'filename': path.name,
        'width': width,
        'height': height,
        'ann':
            {
                'bboxes': np.array(bboxes, dtype='float32'),
                'labels': np.array(labels, dtype='int64'),
            }
    }


def to_mmdet_test(path: t.Union[str, Path]) -> t.Dict[str, t.Any]:
    img = np.array(Image.open(path))
    height, width = img.shape[:2]

    return {
        'filename': path.name,
        'width': width,
        'height': height,
    }


def main():
    args = parse_args()
    path_to_data = Path(args.data_path)

    if args.mode == 'train':
        seed = args.seed
        n_splits = args.n_splits
        with multiprocessing.Pool() as p:
            with tqdm.tqdm(list((path_to_data / 'train_images').glob('*.jpg'))) as pbar:
                train_anns = list(p.imap_unordered(func=to_mmdet_trian, iterable=pbar))

        all_labels = [x['ann']['labels'] for x in train_anns]
        y = np.zeros((len(all_labels), max([max(x) for x in all_labels])))
        for i, labels in enumerate(all_labels):
            for l in set(labels):
                y[i, l - 1] = 1

        mskf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed)
        for train_index, val_index in mskf.split(y, y):
            train_annotation = [x for i, x in enumerate(train_anns) if i in train_index]
            val_annotation = [x for i, x in enumerate(train_anns) if i in val_index]
            print(f'train size: {len(train_annotation)}, val size: {len(val_annotation)}')
            break

        save_pkl(train_annotation, path_to_data / 'train_anns.pkl')
        save_pkl(val_annotation, path_to_data / 'dev_anns.pkl')
    elif args.mode == 'test':
        with multiprocessing.Pool() as p:
            with tqdm.tqdm(list((path_to_data / 'test_images').glob('*.jpg'))) as pbar:
                test_anns = list(p.imap_unordered(func=to_mmdet_test, iterable=pbar))

        save_pkl(test_anns, path_to_data / 'test_anns.pkl')


if __name__ == '__main__':
    main()
