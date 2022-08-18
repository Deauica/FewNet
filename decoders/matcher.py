"""
matcher.py by Hungarian Algorithm,
Modified from: https://github.com/facebookresearch/detr/blob/main/models/matcher.py
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from collections import abc
from typing import Dict, List


class HungarianMatcher(nn.Module):
    """ 相对于 之前的改进，集中在 下面两点：
    1. 匹配准则 的计算方式，这里是动态的，而不是 固定的 NLL 以及 l1_loss 等；
    2. outputs, targets 的形式 有所改变.
    """

    def __init__(self,
                 weight_boxes, cost_boxes_func,
                 weight_logits, cost_logits_func,
                 weight_angle=None, cost_angle_func=None):
        """Creates the matcher
        """
        super(HungarianMatcher, self).__init__()
        self.weight_boxes, self.cost_boxes_func = weight_boxes, cost_boxes_func
        self.weight_logits, self.cost_logits_func = weight_logits, cost_logits_func
        self.weight_angle, self.cost_angle_func = weight_angle, cost_angle_func
        
        assert (self.weight_angle != 0 and self.weight_boxes != 0
                and self.weight_logits !=0), (
            "weight_(angle, boxes, logits) can not be 0, but your weight: {}, {}, {}".format(
                self.weight_boxes, self.weight_logits, self.weight_angle
            )
        )

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        
        Args:
            outputs (Dict[str, torch.Tensor]):
               包括三个核心的 key, 分别是 boxes, angle, logits:
               "boxes": torch.Tensor, .shape == [batch_size, num_queries, 4], 表示 (x, y, w, h);
               "angle": torch.Tensor, .shape == [batch_size, num_queries, 1], 表示角度信息，采用 弧度制;
               "logits": torch.Tensor, .shape == [batch_size, num_queries, 1], 表示 对应 box 为 positive
               的 probability.
               
            targets (List[Dict[str, torch.Tensor]]):
               包括 两个核心的 key, 分别是 boxes, angle， 也可能包括 labels:
               "boxes": torch.Tensor 类型，维度信息为 [num_target_boxes, 4], 表示 (x, y, w, h);
               "angle": torch.Tensor 类型，维度信息为 [num_target_boxes], 表示每一个 box 的角度信息,
                         采用的单位是 弧度制;
               "labels": Optional, torch.Tensor 类型，维度信息为 [num_target_boxes]。对于 二分类的
                         任务来看，这里可以不生成。

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs.shape[:2]
        
        # step 1. obtain out_{boxes, angle, logits}
        out_logits = outputs["logits"].flatten(0, 1).sigmoid()  # [bs * num_queries, 1]
        out_boxes = outputs["boxes"].flatten(0, 1)  # [bs * num_queries, 4], normalized by model.forward
        out_angle = outputs["angle"].flatten(0, 1)  # [bs * num_queries, 1]
        
        # step 2. obtain tgt_{boxes, angle, logits}
        tgt_boxes = torch.cat([tgt["boxes"] for tgt in targets])  # [num_tgt_boxes_batch, 4] -- normalized
        tgt_angle = torch.cat([tgt["angle"] for tgt in targets])  # [num_tgt_boxes_batch, 1]
        tgt_labels = torch.cat([
            tgt.get("labels", torch.full_like(tgt["angle"], 1))
            for tgt in targets])  # [num_tgt_boxes_batch, 1], just like tgt_angle
        
        # step 3. generate cost matrix
        cost_logits = self.cost_logits_func(out_logits, tgt_labels)
        cost_angle = self.cost_angle_func(out_angle, tgt_angle)
        cost_boxes = self.cost_boxes_func(out_boxes, tgt_boxes)  # [bs * num_queries, num_tgt_boxes_batch]
        cost_matrix = (self.weight_logits * cost_logits + self.weight_angle * cost_angle
                       + self.weight_boxes * cost_boxes)  # [bs * num_queries, num_tgt_boxes_batch]
        cost_matrix = cost_matrix.reshape([bs, num_queries, -1]).cpu()
        
        # step 4. perform hungarian algo
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]
