"""
Loss definition for ``Few Could be Better than All``,

有三个 Loss 需要注意到：
1. score_map_loss, For three score maps, perform smooth l1 loss;
   -- 分母 是 total size of three score maps.
2. classification loss, After matching, classification loss is just bce loss;
   -- 分母 是 total number of selection features.
3. detection loss, After matching, detection loss is (scaled gwd loss);
   -- 分母 是 matched boxes 的数量.
"""


from torch import Tensor, nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

from .matcher import *
from .utils import generalized_box_iou, box_cxcywh_to_xyxy
from typing import Union
from .gwd_loss import GDLoss


def cost_boxes_func(out_boxes: Tensor, tgt_boxes: Tensor):
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
    
    out_neg_prob = 1 - out_prob
    return (out_prob * out_prob + out_neg_prob * out_neg_prob).tile([len(tgt_labels), ])


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


class FewNetLoss(nn.Module):
    def __init__(self,
                 weight_cost_logits=1.0, weight_cost_boxes=1.0, weight_cost_angle=1.0,
                 weight_loss_score_map=1.0, weight_loss_logits=1.0, weight_loss_rbox=1.0,
                 max_target_num=100,
                 rbox_loss_type="gwd", rbox_fun="log1p", rbox_tau=1.0):
        super(FewNetLoss, self).__init__()
        self.weight_cost_logits, self.weight_cost_boxes, self.weight_cost_angle = (
            weight_cost_logits, weight_cost_boxes, weight_cost_angle
        )
        self.weight_loss_score_map, self.weight_loss_logits, self.weight_loss_rbox = (
            weight_loss_score_map, weight_loss_logits, weight_loss_rbox
        )
        self.cost_logits_func, self.cost_boxes_func, self.cost_angle_func = (
            cost_logits_func, cost_boxes_func, cost_angle_func
        )
        
        self.loss_logits_func = self.loss_logits
        self.matcher = HungarianMatcher(
            self.weight_cost_boxes, self.cost_boxes_func,
            self.weight_cost_logits, self.cost_logits_func,
            self.weight_cost_angle, self.cost_angle_func
        )
        
        self.max_target_num = max_target_num
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
             
            targets (Dict[str, List[Union[Tensor, List[Tensor]]]]): a dict containing at least these entries:
               "boxes": List of Tensor with dim [num_tgt_boxes_i, 4] for the normalized boxes coordination
               "angle": List of Tensor with dim [num_tgt_boxes_i, 1] for the angle for each boxes.
                       Format should be `le135`
               "score_map": List of List[Tensor]. The first List corresponds to the batch size and the
                       second List corresponds the feature level. Each Tensor's shape should be [Hi, Wi].
                       
        Returns:
            loss_dict (Dict[str, FloatTensor]): key is the name for loss and value is the corresponding
              value.
              
        Notes:
            Since label is not necessary for text detection, so we ignore the "label" key in `targets`.
        """
        loss_dict = {}
        
        # step 1. loss for score_maps
        out_score_maps, tgt_score_maps = outputs.pop("score_map"), targets.pop("score_map")
        loss_score_map = self.loss_score_map(out_score_maps, tgt_score_maps)
        loss_dict.update(dict(
            loss_score_map=self.weight_loss_score_map * loss_score_map))
        
        # step 2. matching between outputs and targets
        # Now, outputs and targets contain no score_map
        indices = self.matcher(outputs, targets)
        
        outputs_matched = self.gen_output_matched(outputs, indices)  # [str, [num_tgt_boxes, ...]]
        targets_matched = self.gen_target_matched(targets, indices)
        
        # step 3. loss for logits
        loss_logits = self.loss_logits_func(outputs_matched["logits"])
        N = outputs["logits"].shape[0] * outputs["logits"].shape[1]
        loss_dict.update(dict(
            loss_logits=self.weight_loss_logits * loss_logits/N))
        
        # step 4. loss for rotated boxes -- 注意 gwd_loss 的计算中，是否可以针对 normalized coords.
        N_r = outputs_matched.shape[0]
        outputs_rbox = torch.cat(  # [num_tgt_boxes, 5]
            [outputs_matched["boxes"], outputs_matched["angle"]], dim=-1
        )
        tgt_rbox = torch.cat(  # [num_tgt_boxes, 5]
            [targets_matched["boxes"], targets_matched["angle"]], dim=-1
        )
        loss_rbox = self.loss_rbox_func(outputs_rbox, tgt_rbox,
                                        reduction_override=self.rbox_reduction)
        loss_dict.update(dict(
            loss_rbox=self.weight_loss_rbox * loss_rbox / N_r))
        return loss_dict
    
    
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
    
    def gen_target_matched(self, targets, indices):
        """Generate matched targets based on `targets` and `indices`.
        
        Returns:
            t (Dict[str, Tensor]):  a dict containing at least "bbox", "logits", "angle".
              dim of Tensor is [num_tgt_boxes, ...]
        """
        assert "score_map" not in targets, (
            "Call this function after self.matcher please"
        )
        _targets = OrderedDict()
        for k in targets.keys():
            _targets[k] = torch.stack([
                F.pad(input=target, pad=(0, 0, 0, self.max_target_num - target[k].shape[0]))
                for target in targets[k]
            ])
            
        # Following code is similar to `self.gen_output_matched`
        sizes = [len(elem[0]) for elem in indices]
        batch_idx = torch.cat([torch.full((s,), i) for i, s in enumerate(sizes)])
        tgt_idx = torch.cat([tgt_indice for (_, tgt_indice) in indices])

        t = OrderedDict()  # [num_tgt_boxes, ...]
        for k in _targets.keys():
            t[k] = _targets[k][batch_idx, tgt_idx]
        return t

    def loss_score_map(self, out_score_maps: List[Tensor], tgt_score_maps: List[Tensor]):
        N_f = 0
        loss_sum = 0
        tgt_score_maps = self.transform_tgt_score_maps(tgt_score_maps)  # add transform
        for out_score_map, tgt_score_map in zip(out_score_maps, tgt_score_maps):
            N_f += out_score_map.shape[-2] * out_score_map.shape[-1]
            loss_sum += F.smooth_l1_loss(
                input=out_score_map, target=tgt_score_map, reduction="sum"
            )  # shape of out_score_map should be same as tgt_score_map
        return loss_sum / N_f
    
    @staticmethod
    def transform_tgt_score_maps(src_tgt_score_maps, feat_level_num=3):
        r""" Perform preprocess to `src_tgt_score_maps` to keep it complied with out_score_maps
        
        Args:
            src_tgt_score_maps (List[List[Tensor]]): First list represents the batch_size and the
              second list represents feature level. Each tensor's shape should be [Hi, Wi]
              
            feat_level_num (int): number of feature level, used to check the tgt_score_maps.
        
        Returns:
            tgt_score_maps (List[Tensor]): This list represents the different feature level, and
              each tensor's shape should be [B, Hi, Wi].
        """
        tgt_score_maps = []
        for t_score_maps in zip(*src_tgt_score_maps):
            tgt_score_maps.append(
                torch.stack(t_score_maps, dim=0)  # [B, Hi, Wi]
            )
            
        assert len(tgt_score_maps) == feat_level_num, (
            "Please check your input data, since your len(tgt_score_maps): {}".format(
                len(tgt_score_maps)
            )
        )
        return tgt_score_maps
    
    def loss_logits(self, outputs_logits):
        """Calculate bce loss for logits in outputs.
        
        Args:
            outputs_logits (Tensor): tensor of dim [num_tgt_boxes, 1]
        """
        targets_logits = torch.full_like(outputs_logits, 1)
        return F.binary_cross_entropy_with_logits(  # sigmoid is perform automatically
            input=outputs_logits, target=targets_logits, reduction="sum"
        )
    

if __name__ == "__main__":
    pass