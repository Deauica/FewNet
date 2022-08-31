"""
Loss definition for ``Few Could be Better than All``,

有三个 Loss 需要注意到：
1. score_map_loss, For three score maps, perform smooth l1 loss;
   -- 分母 是 total size of three score maps.
2. classification loss, After matching, classification loss is just bce loss;
   -- 分母 是 total number of selection features.
3. detection loss, After matching, detection loss is (scaled gwd loss);
   -- 分母 是 matched boxes 的数量.
   
当前的 angle 方面，有一个设置 需要留意：
1. targets 在 make_fewnet_target 中，没有做 归一化，而仅仅只是在 fewnet_loss 的计算环节，做了 逆向归一化;
2. fewnet_loss 中 anti-norm 的具体环节在两个：
  - self.matcher 内部，可能会做归一化，通过传入的 self.minmax 来控制;
  - output_matched 做了归一化，这个一般是必须的，因为 需要通过 gwd_loss.
3. 当前的验证，可以初步判断，gwd_loss 与具体的 angle version 好像没有太大的关系.
"""


from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

from .matcher import *
from .utils import generalized_box_iou, box_cxcywh_to_xyxy
from typing import Union
from .gwd_loss import GDLoss


def cost_bboxes_func(out_boxes: Tensor, tgt_boxes: Tensor):
    """Simplified cost function for bbox
    忽略 angle 的 boxes 的 cost 计算，
    1. 不需要计算 Wasserstein Distance, 没有 复杂的 协方差计算等；
    2. 由于这一步骤 不需要 back propagation，所以不会对最终结果有影响。
    但是，显然，也存在不少的问题，比如，仅仅只是旋转角度不同的 box 可能无法区分
    
    Args:
        out_boxes (Tensor): shape is [bs * num_queries, 4]
        tgt_boxes (Tensor): shape is [num_tgt_boxes_batch, 4]
        
    Returns:
        cost_boxes (Tensor): shape is [bs * num_queries, num_tgt_boxes_batch]
        
    Notes:
        所有的 box 的 coordinate 都已经成功的归一化.
    """
    # Compute the L1 cost between boxes
    cost_boxes_norm1 = torch.cdist(  # [bs * num_queries, num_tgt_boxes_batch]
        out_boxes, tgt_boxes, p=1
    )
    
    # Compute the giou cost between boxes
    cost_boxes_giou = -generalized_box_iou(  # [bs * num_queries, num_tgt_boxes_batch]
        box_cxcywh_to_xyxy(out_boxes), box_cxcywh_to_xyxy(tgt_boxes)
    )
    
    return cost_boxes_norm1 + cost_boxes_giou


def cost_logits_func(out_prob: Tensor, tgt_labels: Tensor):
    """Simplified cost function for bce loss, remove log from traditional bce.
    In other words, cost = p * p + (1-p) * (1-p)
    
    Args:
        out_prob (torch.Tensor): .shape == [bs * num_queries, 1]
        tgt_labels (torch.Tensor): .shape == [num_tgt_boxes_batch]
        
    Returns:
        cost_logits (torch.Tensor): .shape == [bs * num_queries, num_tgt_boxes_batch]
        
    Notes:
        这里后面从 [bs * num_queries, 1] 到 [bs * num_queries, num_tgt_boxes_batch] 的扩展
        是 直接复制的结果。
    """
    if len(out_prob.shape) < 2:
        out_prob = out_prob.unsqueeze(dim=-1)
    cost_logits = F.binary_cross_entropy(out_prob, torch.ones_like(out_prob))
    return cost_logits.tile([len(tgt_labels), ])


def cost_angle_func(out_angle: Tensor, tgt_angle: Tensor):
    """ Simplified version for smooth l1 loss.
    这个就是 最简单的 l1 norm.
    
    Args:
        out_angle (Tensor): shape is [batch_size * num_queries, 1]
        tgt_angle(Tensor): shape is [num_tgt_boxes_batch, 1]
        
    Returns:
        cost_angle(Tensor): shape is [batch_size * num_queries, num_tgt_boxes_batch]
        
    Notes:
        这里采用是 统一的 le135 的 弧度制标准。
    """
    if len(tgt_angle.shape) < 2:
        tgt_angle.unsqueeze(dim=-1)  # [num_tgt_boxes_batch, 1]
    
    cost_angle = torch.cdist(out_angle, tgt_angle, p=1)
    return cost_angle


def cost_rbox_func(out_boxes: Tensor, tgt_boxes: Tensor, **kwargs):
    """Currently, this function utilize GWD Loss as matcher metrics directly
    
    Args:
        out_boxes (Tensor): a Tensor with shape (bs * num_queries, 5)
        tgt_boxes (Tensor): a Tensor with shape (sum_num_tgt_boxes, 5)
        
    Returns:
        cost_mat (Tensor): a two-dimensional Tensor of shape (bs * num_queries, sum_num_tgt_boxes)
    """
    assert out_boxes.shape[-1] == tgt_boxes.shape[-1] == 5, (
        "shape of out_boxes and tgt_boxes: {}, {}".format(out_boxes.shape, tgt_boxes.shape)
    )
    tiled_H, tiled_W = out_boxes.shape[0], tgt_boxes.shape[0]
    tiled_out_boxes = torch.tile(
        out_boxes.unsqueeze(dim=1), dims=(1, tiled_W, 1)
    )  # ensure the element in each row be same
    tiled_tgt_boxes = torch.tile(
        tgt_boxes.unsqueeze(dim=0), dims=(tiled_H, 1, 1)
    )  # ensure the element in each col be same
    loss_rbox_func = GDLoss(
        loss_type=kwargs.get("rbox_loss_type", "gwd"),
        fun=kwargs.get("rbox_fun", "log1p"),
        tau=kwargs.get("rbox_tau", 1.0),
        reduction="none"  # 这里仍然采用 reduction
    )
    cost_mat = loss_rbox_func(
        tiled_out_boxes.flatten(0, 1), tiled_tgt_boxes.flatten(0, 1)
    ).reshape(tiled_H, tiled_W)  #
    return cost_mat


class FewNetLoss(nn.Module):
    def __init__(self,
                 weight_cost_logits=1.0, weight_cost_boxes=1.0,
                 weight_loss_score_map=1.0, weight_loss_logits=1.0, weight_loss_rbox=1.0,
                 max_target_num=100, angle_version="le135",
                 rbox_loss_type="gwd", rbox_fun="log1p", rbox_tau=1.0):
        super(FewNetLoss, self).__init__()
        self.weight_cost_logits, self.weight_cost_boxes = (
            weight_cost_logits, weight_cost_boxes
        )
        self.weight_loss_score_map, self.weight_loss_logits, self.weight_loss_rbox = (
            weight_loss_score_map, weight_loss_logits, weight_loss_rbox
        )
        self.cost_logits_func, self.cost_boxes_func = (
            cost_logits_func, cost_rbox_func  # cost_rbox_func
        )
        
        self.loss_logits_func = self.loss_logits
        
        self.max_target_num = max_target_num
        self.angle_version = angle_version
        self.angle_minmax = torch.as_tensor(dict(
            oc=(0, np.pi / 2), le135=(-np.pi / 4, np.pi * 3 / 4),
            le90=(-np.pi / 2, np.pi / 2)
        )[self.angle_version])
        
        # During matching, for simply consider cost_boxes while ignoring cost_logits
        # for scene text detection.
        self.matcher = HungarianMatcher(
            self.weight_cost_boxes, self.cost_boxes_func,
            self.weight_cost_logits, self.cost_logits_func,
            angle_minmax=self.angle_minmax
        )

        self.rbox_fun, self.rbox_loss_type, self.rbox_tau, self.rbox_reduction = (
            rbox_fun, rbox_loss_type, rbox_tau, "sum"
        )
        self.loss_rbox_func = GDLoss(
            loss_type=self.rbox_loss_type, fun=self.rbox_fun, tau=self.rbox_tau,
            reduction="sum"  # 这里仍然采用 reduction
        )
    
    def forward(self,
                outputs: Dict[str, Union[Tensor, List[Tensor]]],
                targets: Dict[str, Union[Tensor, List[Tensor]]]):
        """Perform the Loss Calculation.
        
        Args:
            outputs (Dict[str, Union[Tensor, List[Tensor]]): This is a dict containing at least these entries:
              "logits": Tensor of dim [bs, num_selected_features, 1] with the classification logits
              "boxes": Tensor of dim [bs, num_selected_features, 4] with the normalized
                       boxes coordinates (cx, cy, w, h)
              "angle": Tensor of dim [bs, num_selected_features, 1] with the angle for each corresponding boxes,
                       format should be `le135`
              "score_map": List of Tensor with dim [bs, Hi, Wi] with point in a tensor
                       representing its importance.
             
            targets (Dict[str, Union[Tensor, List[Tensor]]]): a dict containing at least these entries:
               "boxes": Tensor of shape [B, max_tgt_boxes, 4] with (cx, cy, w, h) for each rotated box.
               "angle": Tensor of shape [B, max_tgt_boxes, 1] with angle for each rotated box.
                       Format should be `le135`
               "score_map": List of Tensor of shape [B, Hi, Wi]. The first List corresponds to the
                       batch size and the shape for each element should be [B, Hi, Wi].
               "num_tgt_boxes": Tensor of the length of batch_size. Each element represent the target
                       boxes in the corresponding element.
                       
        Returns:
            loss_dict (Dict[str, FloatTensor]): key is the name for loss and value is the corresponding
              value.
              
        Notes:
            Since label is not necessary for text detection, so we ignore the "label" key in `targets`.
        """
        loss, loss_dict = 0, {}
        
        # step 0. pre-process for targets["boxes"] and targets["angle"] for historical issue
        l_boxes, l_angles = [], []
        for i, num_tgt_box in enumerate(targets["num_tgt_boxes"]):
            l_boxes.append(targets["boxes"][i, :num_tgt_box])
            l_angles.append(targets["angle"][i, :num_tgt_box])
        targets["boxes"], targets["angle"] = l_boxes, l_angles  # List of [num_tgt_box, _]
        
        # step 1. loss for score_maps
        out_score_maps, tgt_score_maps = outputs.pop("score_map"), targets.pop("score_map")
        loss_score_map = self.loss_score_map(out_score_maps, tgt_score_maps)
        loss_dict.update(
            loss_score_map=self.weight_loss_score_map * loss_score_map)
        
        # step 2. matching between outputs and targets
        # Now, outputs and targets contain no score_map
        indices = self.matcher(outputs, targets)
        
        outputs_matched = self.gen_output_matched(outputs, indices)  # [str, [num_tgt_boxes, ...]]
        targets_matched = self.gen_target_matched(targets, indices)
        
        # step 3. loss for logits
        # the loss_logits should contain all the matched and unmatched samples
        loss_logits = self.loss_logits_func(
            outputs_matched["logits"], outputs["logits"].flatten(0, 1)
        )
        N = outputs["logits"].shape[0] * outputs["logits"].shape[1]  # B * num_selected_features
        loss_dict.update(
            loss_logits=self.weight_loss_logits * loss_logits/N)
        
        # step 4. loss for rotated boxes -- 注意 gwd_loss 的计算中，是否可以针对 normalized coords.
        N_r = outputs_matched["boxes"].shape[0]
        outputs_rbox = torch.cat(  # [num_tgt_boxes, 5]
            [outputs_matched["boxes"],
             self.angle_minmax[0] + outputs_matched["angle"] * (
                     self.angle_minmax[1] - self.angle_minmax[0]
             )], dim=-1
        )
        tgt_rbox = torch.cat(  # [num_tgt_boxes, 5]
            [targets_matched["boxes"], targets_matched["angle"]], dim=-1
        )
        loss_rbox = self.loss_rbox_func(outputs_rbox, tgt_rbox,
                                        reduction_override=self.rbox_reduction)
        loss_dict.update(
            loss_rbox=self.weight_loss_rbox * loss_rbox / N_r)
        
        for _ in loss_dict.values():
            loss += _
        return loss, loss_dict
    
    
    def gen_output_matched(self, outputs, indices):
        """
        Returns:
            t (Dict[str, Tensor]): a dict containing at least "bbox", "logits", "angle".
              dim of Tensor is [num_tgt_boxes, ...]
        """
        assert "score_map" not in outputs, (
            "Call this function after self.matcher please"
        )
        sizes = [len(elem[0]) for elem in indices]
        batch_idx = torch.cat([ torch.full((s,), i) for i, s in enumerate(sizes)])
        src_idx = torch.cat([src_indice for (src_indice, _) in indices])
        
        t = OrderedDict()  # [num_tgt_boxes, ...]
        for k in outputs.keys():
            t[k] = outputs[k][batch_idx, src_idx]
        return t
    
    def gen_target_matched(self, targets, indices, keys=("boxes", "angle")):
        """Generate matched targets based on `targets` and `indices`.
        Args:
            targets (Dict[str, Any]): source targets.
            indices (List[Tuple]): returned value of self.matcher.
            keys (Tuple[str]): containing the keys to be padded and matched.
            
        Returns:
            t (Dict[str, Tensor]):  a dict containing at least "bbox", "angle".
              dim of Tensor is [num_tgt_boxes, ...]
        """
        assert "score_map" not in targets, (
            "Call this function after self.matcher please"
        )
        _targets = OrderedDict()
        for k in targets.keys():
            if k not in keys:  # no operation is needed
                continue
            
            _targets[k] = torch.stack([
                F.pad(input=target, pad=(0, 0, 0, self.max_target_num - target.shape[0]))
                for target in targets[k]
            ])
            
        # Following code is similar to `self.gen_output_matched`
        sizes = [len(elem[1]) for elem in indices]
        batch_idx = torch.cat([torch.full((s,), i) for i, s in enumerate(sizes)])
        tgt_idx = torch.cat([tgt_indice for (_, tgt_indice) in indices])

        t = OrderedDict()  # [num_tgt_boxes, ...]
        for k in _targets.keys():
            t[k] = _targets[k][batch_idx, tgt_idx]
        return t

    def loss_score_map(self, out_score_maps: List[Tensor], tgt_score_maps: List[Tensor]):
        N_f = 0
        loss_sum = 0
        
        for _, (out_score_map, tgt_score_map) in enumerate(zip(out_score_maps, tgt_score_maps)):
            N_f += torch.numel(out_score_map)  # B * Hi * Wi
            loss_sum += F.smooth_l1_loss(
                input=out_score_map, target=tgt_score_map, reduction="sum"
            )  # shape of out_score_map should be same as tgt_score_map
        return loss_sum / N_f
    
    def _loss_logits(self, outputs_logits, target_label=1):
        """Calculate bce loss for logits in outputs.
        
        Args:
            outputs_logits (Tensor): tensor of dim [num_boxes, 1]
            target_label (int): 0 for negative and 1 for positive.
        """
        targets_logits = torch.full_like(outputs_logits, target_label)
        return F.binary_cross_entropy(  # no simgoid is needed to perform
            input=outputs_logits, target=targets_logits, reduction="sum"
        )
    
    def loss_logits(self, out_matched_logits, out_logits):
        """Calculate bce loss for logits in output. Currently,
        we think the matched boxes should be positive and the others are negative.
        Current implementation is redundant due the the historical issue.
        
        Args:
            out_matched_logits (Tensor): tensor of dim [num_tgt_boxes_batch, 1]
            out_logits (Tensor): tensor of dim [bs * num_selected_features, 1]
            
        Returns:
            l: sum of bce logits.
        """
        l_matched_pos = self._loss_logits(out_matched_logits, 1)
        l_matched_neg = self._loss_logits(out_matched_logits, 0)
        l_neg = self._loss_logits(out_logits, 0)
        return l_neg - l_matched_neg + l_matched_pos


if __name__ == "__main__":
    pass