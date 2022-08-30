"""
Post-Processor for `Few Could be Better than All.`
"""

from concern.config import Configurable, State
import inspect
import torch
import numpy as np
from decoders.utils import obb2poly_np


class FewNetPostProcess(Configurable):
    """
    Detailed description for angle definition in rotated box, please refer to:
    https://github.com/open-mmlab/mmrotate/blob/main/docs/en/intro.md
    """
    logits_threshold = State(default=0.5)  # 0.45 for IC15 and 0.5 for others
    angle_version = State(default="le135")
    angle_minmax = dict(
        oc=(0, np.pi / 2), le135=(-np.pi / 4, np.pi * 3 / 4),
        le90=(-np.pi / 2, np.pi / 2)
    )
    
    def __init__(self, logits_threshold=0.5, angle_version="le135", **kwargs):
        super(FewNetPostProcess, self).__init__(**kwargs)
        
        arg_values = inspect.getfullargspec(self.__init__)
        for param, val in zip(arg_values.args[1:], arg_values.defaults):
            # update the param based on the arguments of self.__init__
            if not hasattr(self, param):
                setattr(self, param, val)
            elif val is not None:
                setattr(self, param, val)
            else:
                pass
    
    def represent(self, outputs, data, *args, **kwrags):
        r""" Generate quad version for boxes and proper scores.
        
        Args:
            outputs (Dict[str, torch.Tensor]): A dict contains at least these entries:
              "logits": Tensor of dim [bs, num_selected_features, 1] with the classification logits.
                Sigmoid should be performed here.
              "angle": Tensor of dim [bs, num_selected_features, 1] with the angle. Current angle version
                should be "le135".
              "boxes": Tensor of dim [bs, num_selected_features, 4] with the normalized
                boxes coordinates (cx, cy, w, h).
                
            data (Dict[str, Any]): a dict contains at least these entries:
              "shape": Tensor of List, with each element is a tuple, represent the shape for original image.
                height_i, width_i = data["shape"][i].
            
        Returns:
            boxes_batch (List[ndarray]): QUAD version of boxes.
              for each element, [num_boxes, 9], (x0, y0, x1, y1, x2, y2, x3, y3, theta)
            scores_batch: scores for batches. [num_boxes].
        """
        # re-format outputs["logits"] and outputs["angle]
        if len(outputs["logits"].shape) > 2:
            outputs["logits"] = outputs["logits"].squeeze(dim=-1)  # [bs, num_selected_features]
        outputs["logits"] = outputs["logits"]  # no sigmoid is needed
            
        #
        boxes_batch, scores_batch = [], []
        for i, out_logits, out_angle in enumerate(zip(
                outputs["logits"], outputs["angle"])):
            logits_mask = out_logits > self.logits_threshold
            score = out_logits[logits_mask]  # [num_boxes, 1]
            
            # boxes related, angle is scaled to angle_minmax
            img_H, img_W = data["shape"][i]
            angles = outputs["angle"][logits_mask]  # 2-dim vector, [num_boxes, 1]
            angles = self.angle_minmax[0] + angles * (self.angle_minmax[1] - self.angle_minmax[0])
            
            boxes = outputs["boxes"][logits_mask]  # 2-dim vector, [num_boxes, 4]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * img_W
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * img_H
            boxes = torch.cat(  # [num_candidates, 6], (cx, cy, w, h, theta, score)
                [boxes, angles, score.unsqueeze(dim=-1)], dim=-1)
            boxes = obb2poly_np(boxes, version=self.angle_version)  # [num_boxes, 9]
            
            boxes = boxes[:, :-1]  # 2d tensor
            score = boxes[:, -1]  # 1d vector
            
            boxes_batch.append(boxes.cpu().numpy())
            scores_batch.append(score.cpu().numpy())
        return boxes_batch, scores_batch
        
    __call__ = represent  # __call__ is represent
