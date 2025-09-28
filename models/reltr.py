# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from icecream import ic
import matplotlib.pyplot as plt
import time
from torch import Tensor


class BERI(nn.Module):
    """ Triple Interactive Dual Network """
    def __init__(self, backbone, transformer, num_classes, num_rel_classes, num_entities,num_triplets, num_relations, aux_loss=False, matcher=None):
        
        super().__init__()
        self.num_entities = num_entities
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.query_embed = nn.Embedding(num_entities, hidden_dim)
        

        self.sub_embed = nn.Embedding(num_triplets, hidden_dim)
        self.obj_embed = nn.Embedding(num_triplets, hidden_dim)

        self.entity_embed = nn.Embedding(num_entities,hidden_dim)
        self.rel_embed = nn.Embedding(num_relations, hidden_dim)

        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        
        self.relation_predictor = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_rel_classes + 1)
        )      

        self.sub_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        

        src, mask = features[-1].decompose()
        assert mask is not None

        start_time = time.time()

        hs,hs_rel, hs_sub, hs_obj,_= self.transformer(self.input_proj(src), mask, self.query_embed.weight,self.sub_embed.weight,
                                                 self.obj_embed.weight, pos[-1], self.entity_embed.weight, self.rel_embed.weight)

        outputs_class = self.entity_class_embed(hs)
        outputs_coord = self.entity_bbox_embed(hs).sigmoid()

        outputs_class_sub = self.sub_class_embed(hs_sub)
        outputs_coord_sub = self.sub_bbox_embed(hs_sub).sigmoid()

        outputs_class_obj = self.obj_class_embed(hs_obj)
        outputs_coord_obj = self.obj_bbox_embed(hs_obj).sigmoid()

        rel_feat = torch.cat([hs_sub, hs_obj,hs_rel], dim=-1)
    
        outputs_class_rel = self.relation_predictor(rel_feat)

        detection_time = time.time() - start_time

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
               'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
               'rel_logits': outputs_class_rel[-1],'recognition_time': detection_time,
               
                }
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                    outputs_class_obj, outputs_coord_obj, outputs_class_rel)
        
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                      outputs_class_obj, outputs_coord_obj, outputs_class_rel):
        
        return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f,
                 'rel_logits': g}
                for a, b, c, d, e, f, g in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                               outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1],
                                               outputs_class_rel[:-1])]


class SetCriterion(nn.Module):
    def __init__(self, num_classes, num_rel_classes, matcher, weight_dict, eos_coef, losses, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.label_smoothing = label_smoothing
        empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        empty_weight_rel = torch.ones(num_rel_classes)
        empty_weight_rel[-1] = self.eos_coef
        self.register_buffer('empty_weight_rel', empty_weight_rel)

        self.valid_ratio_ema = None
        self.ema_alpha = 0.9
        self.min_loss_weight = 0.2
        
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_annotations"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes ):
        """Compute the losses related to the entity/subject/object bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        #
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_relations(self, outputs, targets, indices, num_boxes, log=True):

        assert 'rel_logits' in outputs
        src_logits = outputs['rel_logits']
        idx = self._get_src_permutation_idx(indices[1])
        #从每个目标 t 中的 rel_annotations 中获取特定索引位置（由 indices[1] 提供）处的关系类标签（[:, 2]）
        #rel_annotations 的形状通常是一个二维张量，其中每一行代表一对关系，包括：rel_annotations[:, 0]: 主体（subject）的索引\rel_annotations[:, 1]: 客体（object）的索引\rel_annotations[:, 2]: 关系类别标签
        target_classes_o = torch.cat([t["rel_annotations"][J,2] for t, (_, J) in zip(targets, indices[1])])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)

        losses = {'loss_rel': loss_ce}
        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
      

    def get_loss_with_high(self, loss, outputs, targets, match_info, num_boxes, **kwargs):

        if loss != 'relations':
            return self.get_loss_standard(loss, outputs, targets, match_info, num_boxes, **kwargs)

        indices = [match_info['entity_indices'], match_info['rel_indices']]
        
        losses = self.loss_relations(outputs, targets, indices, num_boxes, **kwargs)
        batched_k = match_info['batched_k']
        if hasattr(self, 'weight_by_k') and self.weight_by_k and len(batched_k) > 0:
            k_means = []
            for k in batched_k:
                if isinstance(k, torch.Tensor):
                    k_float = k.float()
                    if k_float.numel() > 0:
                        k_means.append(k_float.mean().item())
                    else:
                        k_means.append(1.0)
                elif isinstance(k, (int, float)):
                    k_means.append(float(k))
                else:
                    k_means.append(1.0)
            
            if k_means:
                weight_factors = [1.0 / max(k, 1.0) for k in k_means]
                if 'loss_rel' in losses:
                    avg_weight = sum(weight_factors) / len(weight_factors)
                    losses['loss_rel'] = losses['loss_rel'] * avg_weight    
                    device = losses['loss_rel'].device
                    losses['high_match_count'] = torch.tensor(sum(k_means) / len(k_means), device=device)
                    losses['high_weight_factor'] = torch.tensor(avg_weight, device=device)
    
        return losses
        

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss_standard(self, loss, outputs, targets, match_info, num_boxes, **kwargs):
       
        indices = [match_info['entity_indices'], match_info['rel_indices']]
        
        # 使用标准损失函数
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'relations': self.loss_relations
        }
        
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 获取匹配信息
        entity_indices, rel_indices, rel_indices_high, batched_k, k_mean_log= self.matcher(outputs_without_aux, targets)
        
        # 存储所有匹配信息
        match_info = {
            'entity_indices': entity_indices,        
            'rel_indices_o2o': rel_indices,        
            'rel_indices_o2m': rel_indices_high,       
            'batched_k': batched_k                 
        }
        

        num_boxes = sum(len(t["labels"]) + len(t["rel_annotations"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:

            if loss == 'relations':

                losses.update(self.get_loss_with_high(loss, outputs, targets, match_info, num_boxes))
            else:

                losses.update(self.get_loss_standard(loss, outputs, targets, match_info, num_boxes))
        

        losses.update({'k_log_layer{}'.format(len(outputs.get('aux_outputs', [])) + 1): k_mean_log})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):

                aux_entity_indices, aux_rel_indices, aux_rel_indices_high, aux_batched_k, aux_k_mean_log, aux_sub_weight, aux_obj_weight = self.matcher(aux_outputs, targets)
                
                aux_match_info = {
                    'entity_indices': aux_entity_indices,
                    'rel_indices': aux_rel_indices,
                    'rel_indices_high': aux_rel_indices_high,
                    'batched_k': aux_batched_k,
                    'k_mean_log': aux_k_mean_log,

                }
                
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'relations':

                        kwargs = {'log': False}
                    
                    if loss == 'relations':
                        l_dict = self.get_loss_with_o2m(loss, aux_outputs, targets, aux_match_info, num_boxes, **kwargs)
                    else:
                        l_dict = self.get_loss_standard(loss, aux_outputs, targets, aux_match_info, num_boxes, **kwargs)
                    
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
       
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    num_classes = 151 
    num_rel_classes = 51

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    matcher = build_matcher(args)
    model = BERI(
        backbone,
        transformer,
        num_classes=num_classes,
        num_rel_classes = num_rel_classes,
        num_entities=args.num_entities,
        num_triplets = args.num_triplets,
        num_relations=args.num_relations,
        aux_loss=args.aux_loss,
        matcher=matcher)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_rel'] = args.rel_loss_coef


    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'relations']

    criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

