# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
dataset (COCO-like) which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import json
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import numpy as np
from tqdm import tqdm

import datasets.transforms as T

#每个类别的出现频率：
#from:https://github.com/SHTUPLUS/PySGG/blob/a63942a076932b3756a477cf8919c3b74cd36207/pysgg/data/datasets/visual_genome.py#L38
# HEAD = [31, 20, 48, 30]
# BODY = [22, 29, 50, 8, 21, 1, 49, 40, 43, 23, 38, 41]
# TAIL = [6, 7, 46, 11, 33, 16, 9, 25, 47, 19, 35, 24, 5, 14, 13, 10, 44, 4, 12, 36, 32, 42, 26, 28, 45, 2, 17, 3, 18, 34,
#         37, 27, 39, 15]


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        #TODO load relationship
        with open('/'.join(ann_file.split('/')[:-1])+'/rel.json', 'r') as f:
            all_rels = json.load(f)
        if 'train' in ann_file:
            self.rel_annotations = all_rels['train']
        elif 'val' in ann_file:
            self.rel_annotations = all_rels['val']
        else:
            self.rel_annotations = all_rels['test']

        self.rel_categories = all_rels['rel_categories']

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        rel_target = self.rel_annotations[str(image_id)]

        target = {'image_id': image_id, 'annotations': target, 'rel_annotations': rel_target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if torch.isnan(boxes).any() or torch.isinf(boxes).any():
            print("Invalid values detected in bounding boxes")
            print("boxes:", boxes)
            raise ValueError("NaN or Inf detected in bounding boxes")

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # TODO add relation gt in the target
        rel_annotations = target['rel_annotations']

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # TODO add relation gt in the target
        target['rel_annotations'] = torch.tensor(rel_annotations)

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    #T.RandomSizeCrop(384, 600), # TODO: cropping causes that some boxes are dropped then no tensor in the relation part! What should we do?
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# https://github.com/suprosanna/relationformer/blob/75c24f61a81466df8f40c498e5f7aae3edd5ac6b/datasets/get_dataset_counts.py#L9
def vg_get_statistics(train_data, must_overlap=True):
    """
    Get counts of all of the relations. Used for modeling directly P(rel | o1, o2)
    :param train_data:
    :param must_overlap:
    :return:
    """
    num_classes = len(train_data.coco.cats)
    num_predicates = len(train_data.rel_categories)

    fg_matrix = np.zeros(
        (
            num_classes + 1,
            num_classes + 1,
            num_predicates,
        ),
        dtype=np.int64,
    )

    rel = train_data.rel
    for idx in tqdm(range(len(train_data))):
        image_id = train_data.ids[idx]

        target = train_data.coco.loadAnns(train_data.coco.getAnnIds(image_id))
        gt_classes = np.array(list(map(lambda x: x["category_id"], target)))
        rel_list = rel[str(image_id)]
        gt_indices = np.array(torch.Tensor(rel_list).T, dtype="int64")
        gt_indices[-1, :] -= 1

        # foreground
        o1o2 = gt_classes[gt_indices[:2, :]].T
        for (o1, o2), gtr in zip(o1o2, gt_indices[2]):
            fg_matrix[o1 - 1, o2 - 1, gtr] += 1

    return fg_matrix

def build(image_set, args):

    ann_path = args.ann_path
    img_folder = args.img_folder

    if image_set == 'train':
        ann_file = ann_path + 'train.json'
    elif image_set == 'val':
        if args.eval:
            ann_file = ann_path + 'test.json'
        else:
            ann_file = ann_path + 'val.json'

    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=False)
    return dataset
