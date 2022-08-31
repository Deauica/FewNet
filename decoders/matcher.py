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
                 weight_boxes, cost_boxes_func, weight_logits, cost_logits_func,
                 angle_minmax=None):
        """Creates the matcher. In this class definition, cost_boxes can may contain the cost
        calculation for various box type, such as, bbox, rbox or bezier box.
        """
        super(HungarianMatcher, self).__init__()
        self.weight_boxes, self.cost_boxes_func = weight_boxes, cost_boxes_func
        self.weight_logits, self.cost_logits_func = weight_logits, cost_logits_func
        
        self.angle_minmax = angle_minmax

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
               
            targets (Dict[str, List[Union[Tensor, List[Tensor]]]]):
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
                
        Notes:
            - raw angle of outputs is (0, 1) scope, so we need proper pre-process.
        """
        # pre-process for outputs["angle"]
        if self.angle_minmax is not None:
            outputs["angle"] = (
                    self.angle_minmax[0] +
                    outputs["angle"] * (self.angle_minmax[1] - self.angle_minmax[0])
            )
        bs, num_queries = outputs["boxes"].shape[:2]
        
        # step 1. obtain out_{boxes, angle, logits}
        out_logits = outputs["logits"].flatten(0, 1).sigmoid()  # [bs * num_queries, 1]
        
        out_boxes = outputs["boxes"].flatten(0, 1)  # [bs * num_queries, 4], 4 for only simple bbox
        out_angle = outputs["angle"].flatten(0, 1)  # [bs * num_queries, 1], 1 for only angle
        out_boxes = torch.cat([out_boxes, out_angle], dim=1)  # [bs * num_queries, 5]
        
        # step 2. obtain tgt_{boxes, angle, logits}
        tgt_boxes = torch.cat(targets["boxes"], dim=0)  # [num_tgt_boxes_batch, 4] -- normalized
        tgt_angle = torch.cat(targets["angle"], dim=0)  # [num_tgt_boxes_batch, 1]
        tgt_boxes = torch.cat([tgt_boxes, tgt_angle], dim=1)  # [num_tgt_boxes_batch, 5]
        
        tgt_labels = (
            torch.cat(targets["label"], dim=0) if "label" in targets else
            torch.full_like(tgt_angle, 1)
        )
        
        # step 3. generate cost matrix
        cost_logits = self.cost_logits_func(out_logits, tgt_labels)
        cost_boxes = self.cost_boxes_func(out_boxes, tgt_boxes)  # [bs * num_queries, num_tgt_boxes_batch]
        cost_matrix = (  # [bs * num_queries, num_tgt_boxes_batch]
                self.weight_logits * cost_logits + self.weight_boxes * cost_boxes)
        cost_matrix = cost_matrix.reshape([bs, num_queries, -1]).cpu()
        
        # step 4. perform hungarian algo
        sizes = [len(_) for _ in targets["boxes"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]
