"""
utils for boxes arithmetic and gaussian target generation

- boxes arithmetic is modified from: github.com:detr/
"""

from torchvision.ops.boxes import box_area
import torch
import numpy as np
import math
from shapely.geometry import Polygon as plg


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def obb2poly_np(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_np_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_np_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_np_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly_np_oc(rbboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    score = rbboxes[:, 5]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le135(rrects):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle, score = rrect[:6]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3, score],
                        dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)
    return polys


def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))


def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))


def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta, score = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates


def obb2poly(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_le90(rbboxes)
    else:
        raise NotImplementedError
    return results

def obb2poly_oc(rboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[:, 0]
    y = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    a = rboxes[:, 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)


def obb2poly_le135(rboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons with Tensor.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def poly_make_valid(poly):
    """Convert a potentially invalid polygon to a valid one by eliminating
    self-crossing or self-touching parts.
    Args:
        poly (Polygon): A polygon needed to be converted.
    Returns:
        A valid polygon.
    """
    return poly if poly.is_valid else poly.buffer(0)


def poly_intersection(poly_det, poly_gt, invalid_ret=None, return_poly=False):
    """Calculate the intersection area between two polygon.
    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.
        invalid_ret (None|float|int): The return value when the invalid polygon
            exists. If it is not specified, the function allows the computation
            to proceed with invalid polygons by cleaning the their
            self-touching or self-crossing parts.
        return_poly (bool): Whether to return the polygon of the intersection
            area.
    Returns:
        intersection_area (float): The intersection area between two polygons.
        poly_obj (Polygon, optional): The Polygon object of the intersection
            area. Set as `None` if the input is invalid.
    """
    assert isinstance(poly_det, plg)
    assert isinstance(poly_gt, plg)
    assert invalid_ret is None or isinstance(invalid_ret, float) or \
        isinstance(invalid_ret, int)

    if invalid_ret is None:
        poly_det = poly_make_valid(poly_det)
        poly_gt = poly_make_valid(poly_gt)

    poly_obj = None
    area = invalid_ret
    if poly_det.is_valid and poly_gt.is_valid:
        poly_obj = poly_det.intersection(poly_gt)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area


def poly_union(poly_det, poly_gt, invalid_ret=None, return_poly=False):
    """Calculate the union area between two polygon.
    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.
        invalid_ret (None|float|int): The return value when the invalid polygon
            exists. If it is not specified, the function allows the computation
            to proceed with invalid polygons by cleaning the their
            self-touching or self-crossing parts.
        return_poly (bool): Whether to return the polygon of the intersection
            area.
    Returns:
        union_area (float): The union area between two polygons.
        poly_obj (Polygon|MultiPolygon, optional): The Polygon or MultiPolygon
            object of the union of the inputs. The type of object depends on
            whether they intersect or not. Set as `None` if the input is
            invalid.
    """
    assert isinstance(poly_det, plg)
    assert isinstance(poly_gt, plg)
    assert invalid_ret is None or isinstance(invalid_ret, float) or \
        isinstance(invalid_ret, int)

    if invalid_ret is None:
        poly_det = poly_make_valid(poly_det)
        poly_gt = poly_make_valid(poly_gt)

    poly_obj = None
    area = invalid_ret
    if poly_det.is_valid and poly_gt.is_valid:
        poly_obj = poly_det.union(poly_gt)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area


def poly_iou(poly_det, poly_gt, zero_division=0):
    """Calculate the IOU between two polygons.
    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.
        zero_division (int|float): The return value when invalid
                                    polygon exists.
    Returns:
        iou (float): The IOU between two polygons.
    """
    assert isinstance(poly_det, plg)
    assert isinstance(poly_gt, plg)
    
    area_inters = poly_intersection(poly_det, poly_gt)
    area_union = poly_union(poly_det, poly_gt)
    return area_inters / area_union if area_union != 0 else zero_division


# This class is only used for debug
import numpy as np
import torch
from typing import List
import cv2
import os


class DebugFewNetLoss(object):
    def __init__(self, angle_version, is_plot_unmatched=True, ratio=2, step = 1):
        self.plot_call_num = 0
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.angle_version = angle_version
        self.is_plot_unmatched = is_plot_unmatched
        self.ratio = ratio
        self.step = step
        assert self.step != 0, "step can not be zero"
        
        
        if os.path.exists("debug/loss_debug"):
            import shutil
            shutil.rmtree("debug/loss_debug")
        os.mkdir("debug/loss_debug")

    def norm_tensor_2_ndarray(self, t) -> List[np.ndarray]:
        t = t * 255
    
        results = []
        for img in torch.unbind(t, dim=0):
            img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))  # [H, W, C]
            img += self.RGB_MEAN.reshape([1, 1, -1])
            img = img.clip(0, 255)
            results.append(img.astype(np.uint8))
        return results
    
    
    def plot_out_tgt_boxes(self, targets, outputs, indices):
        # step 0: environ preparation
        self.plot_call_num += 1
        if self.plot_call_num % self.step != 0:
            return  # do nothing
        
        # step 1: generate images
        tgt_images = self.norm_tensor_2_ndarray(targets["image"])  # List[ndarray]
        
        # step 2: generate {tgt, out}_boxes
        tgt_boxes, out_boxes = [], []  # len == number of images
        out_unmatched_boxes = []
        
        for (_tgt_boxes, _out_boxes, _tgt_angles, _out_angles, (out_idxes, tgt_idxes)) in zip(
                targets["boxes"], outputs["boxes"], targets["angle"], outputs["angle"], indices):
            """ for each image """
            t = torch.ones(_out_boxes.shape[0])
            out_unmatched_idxes = torch.nonzero(
                torch.scatter(t.to(out_idxes.device), 0, out_idxes, 0)
            ).flatten()[: out_idxes.shape[0] * self.ratio]
            
            tgt_boxes_with_score = torch.cat(
                [_tgt_boxes[tgt_idxes], _tgt_angles[tgt_idxes],
                 torch.ones([_tgt_boxes[tgt_idxes].shape[0], 1], device=_tgt_boxes.device)],
                dim=-1
            )
            out_boxes_with_score = torch.cat(
                [_out_boxes[out_idxes], _out_angles[out_idxes],
                 torch.ones([_out_boxes[out_idxes].shape[0], 1], device=_out_boxes.device)],
                dim=-1
            )
            out_unmatched_boxes_with_score = torch.cat(
                [_out_boxes[out_unmatched_idxes], _out_angles[out_unmatched_idxes],
                 torch.ones([_out_boxes[out_unmatched_idxes].shape[0], 1], device=_out_boxes.device)],
                dim=-1
            )
            tgt_boxes.append(tgt_boxes_with_score.detach().cpu().numpy())
            out_boxes.append(out_boxes_with_score.detach().cpu().numpy())
            out_unmatched_boxes.append(out_unmatched_boxes_with_score.detach().cpu().numpy())
        
        # transform {tgt, out}_boxes to quad version
        tgt_boxes = [
            obb2poly_np(tgt_boxes_per_img, self.angle_version)[:, :-1]
            for tgt_boxes_per_img in tgt_boxes
        ]
        out_boxes = [
            obb2poly_np(out_boxes_per_img, self.angle_version)[:, :-1]
            for out_boxes_per_img in out_boxes
        ]
        out_unmatched_boxes = [
            obb2poly_np(out_unmatched_boxes_per_img, self.angle_version)[:, :-1]
            for out_unmatched_boxes_per_img in out_unmatched_boxes
        ]
        
        # step 3: plot boxes to the images
        for i, (tgt_img, tgt_boxes_per_img, out_boxes_per_img, out_unmatched_boxes_per_img) in enumerate(
                zip(tgt_images, tgt_boxes, out_boxes, out_unmatched_boxes)):
            """
            G: gt,
            R: matched out_boxes,
            B: unmatched_out_boxes
            """
            tgt_img = cv2.polylines(tgt_img, tgt_boxes_per_img.reshape([-1, 4, 2]).astype(np.int32),
                                    True, (0, 255, 0), 2)
            tgt_img = cv2.polylines(tgt_img, out_boxes_per_img.reshape([-1, 4, 2]).astype(np.int32),
                                    True, (0, 0, 255), 2)
            if self.is_plot_unmatched:
                tgt_img = cv2.polylines(
                    tgt_img, out_unmatched_boxes_per_img.reshape([-1, 4, 2]).astype(np.int32),
                    True, (255, 0, 0), 2
                )
                
            img_root = "debug/loss_debug"
            
            if "filename" in targets:
                img_dirname = os.path.basename(targets["filename"][i])
                img_dirpath = os.path.join(img_root, img_dirname)
                if not os.path.exists(img_dirpath):
                    os.mkdir(img_dirpath)
                
                imgname = "ld_{}_{}".format(
                    self.plot_call_num, os.path.basename(targets["filename"][i])
                )
                imgpath = os.path.join(img_dirpath, imgname)
                
                cv2.imwrite(imgpath, tgt_img)
            else:
                raise KeyError("filename not exists")


# introduce loss from EAST
from torch import nn
import torch
from collections import OrderedDict


class EastLoss(nn.Module):
    def __init__(self, weight_bbox=1, weight_theta=1, reduction="none", *args, **kwargs):
        super(EastLoss, self).__init__()
        self.weight_bbox, self.weight_theta = weight_bbox, weight_theta
        self.reduction = reduction
        self.eps = 1e-10  # eps
        self._args, self._kwargs = args, kwargs
        
        assert self.reduction in ["sum", "mean", "none"], (
            "The reduction can noly be one of ['sum', 'mean', 'none'], "
            "However, your reduction is: {}".format(self.reduction)
        )
        
    def loss_theta(self,
                   pred_theta: torch.Tensor, tgt_theta: torch.Tensor, reduction_override=None):
        loss = 1 - torch.cos(pred_theta - tgt_theta + self.eps)
        reduction = self.reduction if reduction_override is None else reduction_override
        
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def loss_bbox(self, pred_bbox: torch.Tensor, tgt_bbox: torch.Tensor, reduction_override=None):
        """ Calculate the bbox loss for (cx, cy, w, h) format.
        
        Args:
            pred_bbox (torch.Tensor): Tensor with shape [num_boxes, 4], (cx, cy, w, h)
            tgt_bbox (torch.Tensor): Tensor with shape [num_bxoes, 4], (cx, cy, w, h)
        
        Note:
            - This function will utilize giou instead of the iou loss in EAST.
        """
        reduction = self.reduction if reduction_override is None else reduction_override
        
        pred_bbox = box_cxcywh_to_xyxy(pred_bbox)
        tgt_bbox = box_cxcywh_to_xyxy(tgt_bbox)
        return self.generalized_iou_loss(tgt_bbox, pred_bbox, reduction)
    
    def forward(self, pred_rbox: torch.Tensor, tgt_rbox: torch.Tensor, reduction_override=None,
                *args, **kwargs):
        """ Calculate the Loss with pred_rbox and tgt_rbox representing prediction and target
        
        Args:
            pred_rbox (torch.Tensor): prediction with shape [num_rboxes, 5]
            tgt_rbox (torch.Tensor): gt with shape [num_rboxes, 5]
            reduction_override (Optional[str]): if not None, override the reduction in self.reduction.
        
        Returns:
            loss (torch.Tensor): Tensor object which is the weighted sum for loss_dict.
        """
        assert pred_rbox.shape[-1] == tgt_rbox.shape[-1] == 5, (
            "Please check your input rbox data since the shape of pred_rbox and tgt_rbox"
            "are: {}, {}".format(pred_rbox.shape, tgt_rbox.shape)
        )
        loss_angle = self.loss_theta(pred_rbox[:, -1:], tgt_rbox[:, -1:], reduction_override)
        loss_bbox = self.loss_bbox(pred_rbox[:, :-1], tgt_rbox[:, :-1], reduction_override)
        loss = self.weight_theta * loss_angle + self.weight_bbox * loss_bbox
        if reduction_override is None:
            assert loss.shape[0] > 0, f"loss does not exist, please check it, {loss}"
        
        return loss
    
    @staticmethod
    def generalized_iou_loss(gt_bboxes, pr_bboxes, reduction='none'):
        """
        This function is adapted from:
           https://github.com/CoinCheung/pytorch-loss/blob/master/generalized_iou_loss.py
        
        gt_bboxes: tensor (-1, 4) xyxy
        pr_bboxes: tensor (-1, 4) xyxy
        loss proposed in the paper of giou
        
        Note:
            - we should keep the dimension of result complied with `gt_bboxes`.
        """
        gt_area = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        pr_area = (pr_bboxes[:, 2] - pr_bboxes[:, 0]) * (pr_bboxes[:, 3] - pr_bboxes[:, 1])
        
        # iou
        lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
        rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
        TO_REMOVE = 1
        wh = (rb - lt + TO_REMOVE).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        union = gt_area + pr_area - inter
        iou = inter / union
        # enclosure
        lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
        rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
        wh = (rb - lt + TO_REMOVE).clamp(min=0)
        enclosure = wh[:, 0] * wh[:, 1]
    
        giou = iou - (enclosure - union) / enclosure
        loss = 1. - giou.unsqueeze(dim=-1)  # recover the dimension
        
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'none':
            pass
        return loss
