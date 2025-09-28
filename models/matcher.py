import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
import torch.nn.functional as F
import copy
from icecream import ic

sorted_dict = {'on': 712409, 'has': 277936, 'in': 251756, 'of': 146339, 'wearing': 136099, 'near': 96589, 'with': 66425, 'above': 47341, 'holding': 42722, 'behind': 41356, 'under': 22596, 'sitting on': 18643, 'wears': 15457, 'standing on': 14185, 'in front of': 13715, 'attached to': 10190, 'at': 9903, 'hanging from': 9894, 'over': 9317, 'for': 9145, 'riding': 8856, 'carrying': 5213, 'eating': 4688, 'walking on': 4613, 'playing': 3810, 'covering': 3806, 'laying on': 3739, 'along': 3624, 'watching': 3490, 'and': 3477, 'between': 3411, 'belonging to': 3288, 'painted on': 3095, 'against': 3092, 'looking at': 3083, 'from': 2945, 'parked on': 2721, 'to': 2517, 'made of': 2380, 'covered in': 2312, 'mounted on': 2253, 'says': 2241, 'part of': 2065, 'across': 1996, 'flying in': 1973, 'using': 1925, 'on back of': 1914, 'lying on': 1869, 'growing on': 1853, 'walking in': 1740}
sorted_idxs_with_cnt = {30: 712409, 19: 277936, 21: 251756, 29: 146339, 47: 136099, 28: 96589, 49: 66425, 0: 47341, 20: 42722, 7: 41356, 42: 22596, 39: 18643, 48: 15457, 40: 14185, 22: 13715, 6: 10190, 5: 9903, 18: 9894, 32: 9317, 15: 9145, 37: 8856, 10: 5213, 13: 4688, 45: 4613, 36: 3810, 12: 3806, 23: 3739, 3: 3624, 46: 3490, 4: 3477, 9: 3411, 8: 3288, 33: 3095, 2: 3092, 24: 3083, 16: 2945, 34: 2721, 41: 2517, 26: 2380, 11: 2312, 27: 2253, 38: 2241, 35: 2065, 1: 1996, 14: 1973, 43: 1925, 31: 1914, 25: 1869, 17: 1853, 44: 1740}
alphabet_list = [    'above',    'across',    'against',    'along',    'and',    'at',    'attached to',    'behind',    'belonging to',    'between',    'carrying',    'covered in',    'covering',    'eating',    'flying in',    'for',    'from',    'growing on',    'hanging from',    'has',    'holding',    'in',    'in front of',    'laying on',    'looking at',    'lying on',    'made of',    'mounted on',    'near',    'of',    'on',    'on back of',    'over',    'painted on',    'parked on',    'part of',    'playing',    'riding',    'says',    'sitting on',    'standing on',    'to',    'under',    'using',    'walking in',    'walking on',    'watching',    'wearing',    'wears',    'with']
alphabet_to_frequecy = [7, 43, 33, 27, 29, 16, 15, 9, 31, 30, 21, 39, 25, 22, 44, 19, 35, 48, 17, 1, 8, 2, 14, 26, 34, 47, 38, 40, 5, 3, 0, 46, 18, 32, 36, 42, 24, 20, 41, 11, 13, 37, 10, 45, 49, 23, 28, 4, 12, 6]

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, iou_threshold: float = 0.7):
        
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.iou_threshold = iou_threshold
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.relation_order = torch.tensor([30, 19, 21, 29, 47, 28, 49, 0, 20, 7, 42, 39, 48, 40, 22, 6, 5, 18, 32, 15, 37, 10, 13, 45, 36, 12, 23, 3, 46, 4, 9, 8, 33, 2, 24, 16, 34, 41, 26, 11, 27, 38, 35, 1, 14, 43, 31, 25, 17, 44])
        self.relation_order = self.relation_order
        self.o2m_k = 4
        self.query_multiple = 1
        self.sorted_dict = sorted_dict
        self.sorted_idxs_with_cnt = sorted_idxs_with_cnt
        self.relation_freq = torch.tensor(list(sorted_dict.values()))
        self.num_mul_so_queries = 200
        self.num_groups = 4
        self.o2m_predicate_weight = -0.5
        self.size_of_groups = self.get_group_list_by_n_groups(self.num_groups)
        self.o2m_predicate_score = True
        self.use_group_mask = True

    def fill_list(self, num, n):
      quotient, remainder = divmod(num, n)
      lst = [quotient] * n
      for i in range(remainder):
          lst[-1 * (i + 1)] += 1
      return torch.tensor(lst)


    def get_group_list_by_n_groups(self, n_groups):
        total_list = list()
        last_checked_index = 0
        current_idx = 0
        size_of_whole_groups = 0
        for i in range(n_groups - 1):
            sum_of_this_group = 0
            size_of_this_group = 0
            remaining_list = self.relation_freq.numpy()[last_checked_index:]
            remaining_half_cnt = remaining_list.sum() // 2
            while sum_of_this_group + self.relation_freq.numpy()[current_idx] < remaining_half_cnt:
                sum_of_this_group += self.relation_freq.numpy()[current_idx]
                size_of_this_group += 1
                size_of_whole_groups += 1
                current_idx += 1
            total_list.append(size_of_this_group)
            last_checked_index = current_idx
        total_list.append(50 - size_of_whole_groups)
        return total_list
    
    def compute_enhanced_spatial_quality(self, sub_bbox, obj_bbox, tgt_sub_bbox, tgt_obj_bbox, bs, num_queries):
        sub_xyxy = box_cxcywh_to_xyxy(sub_bbox)
        obj_xyxy = box_cxcywh_to_xyxy(obj_bbox)
        tgt_sub_xyxy = box_cxcywh_to_xyxy(tgt_sub_bbox)
        tgt_obj_xyxy = box_cxcywh_to_xyxy(tgt_obj_bbox)
        
        sub_cx = (sub_xyxy[:, 0] + sub_xyxy[:, 2]) / 2
        sub_cy = (sub_xyxy[:, 1] + sub_xyxy[:, 3]) / 2
        obj_cx = (obj_xyxy[:, 0] + obj_xyxy[:, 2]) / 2
        obj_cy = (obj_xyxy[:, 1] + obj_xyxy[:, 3]) / 2
        
        sub_w = sub_xyxy[:, 2] - sub_xyxy[:, 0]
        sub_h = sub_xyxy[:, 3] - sub_xyxy[:, 1]
        obj_w = obj_xyxy[:, 2] - obj_xyxy[:, 0]
        obj_h = obj_xyxy[:, 3] - obj_xyxy[:, 1]

        dx = (obj_cx - sub_cx) / (sub_w + 1e-6)  
        dy = (obj_cy - sub_cy) / (sub_h + 1e-6) 
        dw = torch.log(obj_w / (sub_w + 1e-6)) 
        dh = torch.log(obj_h / (sub_h + 1e-6))  
        
        pred_vector = torch.stack([dx, dy], dim=-1)
        pred_size_ratio = torch.stack([dw, dh], dim=-1) 

        total_relations = tgt_sub_bbox.shape[0]
        tgt_sub_cx = (tgt_sub_xyxy[:, 0] + tgt_sub_xyxy[:, 2]) / 2
        tgt_sub_cy = (tgt_sub_xyxy[:, 1] + tgt_sub_xyxy[:, 3]) / 2
        tgt_obj_cx = (tgt_obj_xyxy[:, 0] + tgt_obj_xyxy[:, 2]) / 2
        tgt_obj_cy = (tgt_obj_xyxy[:, 1] + tgt_obj_xyxy[:, 3]) / 2
        
        tgt_sub_w = tgt_sub_xyxy[:, 2] - tgt_sub_xyxy[:, 0]
        tgt_sub_h = tgt_sub_xyxy[:, 3] - tgt_sub_xyxy[:, 1]
        tgt_obj_w = tgt_obj_xyxy[:, 2] - tgt_obj_xyxy[:, 0]
        tgt_obj_h = tgt_obj_xyxy[:, 3] - tgt_obj_xyxy[:, 1]
        
        tgt_dx = (tgt_obj_cx - tgt_sub_cx) / (tgt_sub_w + 1e-6)
        tgt_dy = (tgt_obj_cy - tgt_sub_cy) / (tgt_sub_h + 1e-6)
        tgt_dw = torch.log(tgt_obj_w / (tgt_sub_w + 1e-6))
        tgt_dh = torch.log(tgt_obj_h / (tgt_sub_h + 1e-6))
        
        tgt_vector = torch.stack([tgt_dx, tgt_dy], dim=-1)
        tgt_size_ratio = torch.stack([tgt_dw, tgt_dh], dim=-1) 
        
        pred_vector = pred_vector.view(bs, num_queries, 1, 2) 
        pred_size_ratio = pred_size_ratio.view(bs, num_queries, 1, 2)  
        tgt_vector = tgt_vector.view(1, 1, total_relations, 2)  
        tgt_size_ratio = tgt_size_ratio.view(1, 1, total_relations, 2) 

        pred_dist = torch.norm(pred_vector, dim=-1, keepdim=True) + 1e-6
        tgt_dist = torch.norm(tgt_vector, dim=-1, keepdim=True) + 1e-6
        pred_direction = pred_vector / pred_dist
        tgt_direction = tgt_vector / tgt_dist
        direction_sim = torch.sum(pred_direction * tgt_direction, dim=-1) 
        direction_sim = (direction_sim + 1) / 2 

        size_sim = 1 - torch.mean(torch.abs(pred_size_ratio - tgt_size_ratio), dim=-1) 
        size_sim = size_sim.clamp(0, 1) 
        
        dist_ratio = torch.min(pred_dist / (tgt_dist + 1e-6), tgt_dist / (pred_dist + 1e-6))
        dist_sim = dist_ratio.squeeze(-1) 

        spatial_quality =  direction_sim + dist_sim + size_sim 
        
        return spatial_quality
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        
        bs, num_queries = outputs["pred_logits"].shape[:2]
        num_queries_rel = outputs["rel_logits"].shape[1]
        alpha = 0.25
        gamma = 2.0

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        C_split = C.split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]

        sub_tgt_bbox = torch.cat([v['boxes'][v['rel_annotations'][:, 0]] for v in targets])
        sub_tgt_ids = torch.cat([v['labels'][v['rel_annotations'][:, 0]] for v in targets])
        obj_tgt_bbox = torch.cat([v['boxes'][v['rel_annotations'][:, 1]] for v in targets])
        obj_tgt_ids = torch.cat([v['labels'][v['rel_annotations'][:, 1]] for v in targets])
        rel_tgt_ids = torch.cat([v["rel_annotations"][:, 2] for v in targets])

        anno_len = [len(t['rel_annotations']) for t in targets]

        gt_rel_ids = torch.cat([v["rel_annotations"][:, 2] for v in targets])
        rel_groups = self.group_tensor[gt_rel_ids].long()
        invalid_mask = (rel_groups == -1) 
        rel_groups[invalid_mask] = torch.randint(0, self.n_groups, (invalid_mask.sum(),), device=rel_groups.device)

        group_mask = 1-F.one_hot(rel_groups.long(), num_classes=len(self.n_queries_per_group)).t()
        freq_list = [fl * self.query_multiple for fl in self.n_queries_per_group]
        
        for idx, freq in enumerate(freq_list):
            temp = group_mask[idx].reshape(1,1,-1).repeat(bs, freq, 1) * 1e+6  # 降低常数值
            if idx == 0:
                new_group_mask = temp
            else:
                new_group_mask = torch.cat((new_group_mask, temp), dim=1)
        
        group_mask = new_group_mask.reshape(bs*num_queries_rel, -1).to(outputs["rel_logits"].device)

        sub_prob = outputs["sub_logits"].flatten(0, 1).sigmoid()
        sub_bbox = outputs["sub_boxes"].flatten(0, 1)
        obj_prob = outputs["obj_logits"].flatten(0, 1).sigmoid()
        obj_bbox = outputs["obj_boxes"].flatten(0, 1)
        rel_prob = outputs["rel_logits"].flatten(0, 1).sigmoid()
        
        sub_iou = box_iou(box_cxcywh_to_xyxy(sub_bbox), box_cxcywh_to_xyxy(sub_tgt_bbox))[0].view(bs, num_queries_rel, -1)
        obj_iou = box_iou(box_cxcywh_to_xyxy(obj_bbox), box_cxcywh_to_xyxy(obj_tgt_bbox))[0].view(bs, num_queries_rel, -1)
        mask_iou = (group_mask.reshape(bs,num_queries_rel,-1)==0).float()#

        sub_iou *=mask_iou
        obj_iou *=mask_iou

        spatial_quality = self.compute_enhanced_spatial_quality(
                                sub_bbox, obj_bbox, 
                                sub_tgt_bbox, obj_tgt_bbox, 
                                bs, num_queries_rel
                            )
        box_quality = torch.max(sub_iou, obj_iou)
        cum_iou = box_quality * (0.3+ 0.7 * spatial_quality)
        rel_out_prob = outputs['rel_logits'].softmax(dim=-1)
        cum_iou += self.o2m_predicate_weight * rel_out_prob[..., rel_tgt_ids]

        topk_iou = torch.topk(cum_iou, self.o2m_k, dim=1)[0]
        batched_k = [(ci[i].sum(dim=0)).int().clamp_(min=1) for i, ci in enumerate(topk_iou.split(anno_len, dim=-1))]
        k_mean_log = torch.nanmean(torch.cat(batched_k).float(), dim=0).nan_to_num(0)

        neg_cost_class_sub = (1 - alpha) * (sub_prob ** gamma) * (-(1 - sub_prob + 1e-8).log())
        pos_cost_class_sub = alpha * ((1 - sub_prob) ** gamma) * (-(sub_prob + 1e-8).log())
        cost_sub_class = pos_cost_class_sub[:, sub_tgt_ids] - neg_cost_class_sub[:, sub_tgt_ids]
        cost_sub_bbox = torch.cdist(sub_bbox, sub_tgt_bbox, p=1)
        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(sub_bbox), box_cxcywh_to_xyxy(sub_tgt_bbox))

        neg_cost_class_obj = (1 - alpha) * (obj_prob ** gamma) * (-(1 - obj_prob + 1e-8).log())
        pos_cost_class_obj = alpha * ((1 - obj_prob) ** gamma) * (-(obj_prob + 1e-8).log())
        cost_obj_class = pos_cost_class_obj[:, obj_tgt_ids] - neg_cost_class_obj[:, obj_tgt_ids]
        cost_obj_bbox = torch.cdist(obj_bbox, obj_tgt_bbox, p=1)
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(obj_bbox), box_cxcywh_to_xyxy(obj_tgt_bbox))

        neg_cost_class_rel = (1 - alpha) * (rel_prob ** gamma) * (-(1 - rel_prob + 1e-8).log())
        pos_cost_class_rel = alpha * ((1 - rel_prob) ** gamma) * (-(rel_prob + 1e-8).log())
        cost_rel_class = pos_cost_class_rel[:, rel_tgt_ids] - neg_cost_class_rel[:, rel_tgt_ids]

        C_sub_obj = self.cost_class * (cost_sub_class + cost_obj_class)
        C_rel = 0.5 * cost_rel_class
        C_rel = self.cost_bbox * (cost_sub_bbox + cost_obj_bbox) + C_sub_obj + C_rel + self.cost_giou * (cost_sub_giou + cost_obj_giou)

        C_rel_one_to_one = C_rel.view(bs, num_queries_rel, -1).cpu()
        sizes1 = [len(v["rel_annotations"]) for v in targets]
        C_rel_split = C_rel_one_to_one.split(sizes1, -1)
        indices1 = [linear_sum_assignment(c[i]) for i, c in enumerate(C_rel_split)]
        
        if self.use_group_mask:
            C_rel_o2m = C_rel + group_mask
            C_rel_o2m = C_rel_o2m.view(bs, num_queries_rel, -1).cpu()
            C_rel_o2m_split = C_rel_o2m.split(sizes1, -1)
            indices2 = [linear_sum_assignment(c[i]) for i, c in enumerate(C_rel_o2m_split)]
        else:
            indices2 = indices1
        
        entity_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        rel_indices_o2o = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices1]
        rel_indices_o2m = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices2]

        return entity_indices, rel_indices_o2o, rel_indices_o2m, batched_k, k_mean_log

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, iou_threshold=args.set_iou_threshold)
