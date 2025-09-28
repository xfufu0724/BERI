import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import yaml
import numpy as np
from collections import defaultdict, OrderedDict, Counter
class VisualGenomeTrainData:
    def __init__(self):
        statistics = self.get_statistics()

def get_VG_statistics(train_data, must_overlap=True):
    """Save the initial data distribution for the frequency bias model for COCO dataset

    Args:
        train_data: CocoDetection dataset instance
        must_overlap (bool, optional): Whether to only consider overlapping boxes. Defaults to True.

    Returns:
        fg_matrix: Foreground matrix with counts of subject-object-relation occurrences
        bg_matrix: Background matrix with counts of subject-object co-occurrences
        rel_counter: Counter of relation occurrences
    """
    # Get the number of object and relation classes
    num_obj_classes = 150  # COCO categories
    num_rel_classes = 50  # Relation categories
    
    # Initialize matrices
    fg_matrix = np.zeros((num_obj_classes+1, num_obj_classes+1, num_rel_classes+1), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes+1, num_obj_classes+1), dtype=np.int64)
    rel_counter = Counter()
    
    # Iterate through all images in the dataset
    for idx in tqdm(range(len(train_data))):
        # Get image and target
        _, target = train_data[idx]
        
        # Extract the necessary information
        gt_classes = target['labels'].numpy()
        gt_boxes = target['boxes'].numpy()
        gt_relations = target['rel_annotations'].numpy()
        
        # For the foreground, process all relations
        if len(gt_relations) > 0:
            for rel in gt_relations:
                subj_idx = rel[0]
                obj_idx = rel[1]
                rel_type = rel[2]
                
                if subj_idx < len(gt_classes) and obj_idx < len(gt_classes):
                    subj_class = gt_classes[subj_idx]
                    obj_class = gt_classes[obj_idx]
                    
                    # Update foreground matrix and relation counter
                    fg_matrix[subj_class, obj_class, rel_type] += 1
                    rel_counter[rel_type] += 1
        
        # For the background, get all things that overlap (if requested)
        if must_overlap:
            # Create all possible object pairs and check overlap
            for i in range(len(gt_boxes)):
                for j in range(len(gt_boxes)):
                    if i != j:  # Don't consider the same object
                        box1 = gt_boxes[i]
                        box2 = gt_boxes[j]
                        
                        # Check if boxes overlap (simple IoU check)
                        if boxes_overlap(box1, box2):
                            bg_matrix[gt_classes[i], gt_classes[j]] += 1
        else:
            # If overlap not required, count all object pairs
            for i in range(len(gt_classes)):
                for j in range(len(gt_classes)):
                    if i != j:  # Don't consider the same object
                        bg_matrix[gt_classes[i], gt_classes[j]] += 1
    
    return fg_matrix, bg_matrix, rel_counter

def boxes_overlap(box1, box2, threshold=0.0):
    """Check if two boxes overlap.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        threshold: IoU threshold
    
    Returns:
        bool: True if boxes overlap above threshold
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x2 < x1 or y2 < y1:
        return False
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    iou = intersection / (area1 + area2 - intersection)
    
    return iou > threshold