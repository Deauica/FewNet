"""
Utilities for data preprocessing.
1. gaussian 2d distribution. modified from mmdet;
2.
"""
import torch
import numpy as np
import cv2

import concern.config
from torch.utils.data._utils.collate import default_collate, default_convert
import torch.nn.functional as F

def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.
    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.
    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.
    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.
    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.

    Notes:
        When two gaussian kernel meet, max value will be preserved.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2D(
        radius, sigma=diameter, dtype=heatmap.dtype, device=heatmap.device)
    
    x, y = center
    
    height, width = heatmap.shape[:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius - top:radius + bottom,
                      radius - left:radius + right]
    out_heatmap = heatmap
    out_heatmap[y - top:y + bottom, x - left:x + right] = (
        torch.max(masked_heatmap, masked_gaussian * k)
    )
    return out_heatmap


def poly2obb_np_oc(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
    if w < 2 or h < 2:
        return
    while not 0 < a <= 90:
        if a == -90:
            a += 180
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 < a <= np.pi / 2
    return x, y, w, h, a


def poly2obb_np_le135(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)
    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))
    if edge1 < 2 or edge2 < 2:
        return
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))
    angle = norm_angle(angle, 'le135')
    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return x_ctr, y_ctr, width, height, angle


def poly2obb_np(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_np_oc(polys)
    elif version == 'le135':
        results = poly2obb_np_le135(polys)
    elif version == 'le90':
        results = poly2obb_np_le90(polys)
    else:
        raise NotImplementedError
    return results


def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a


def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')
    

class FewNetCollate(concern.config.Configurable):
    def __int__(self, *args, **kwargs):
        pass
    
    def __call__(self, batch):
        """
        对默认的 collate_fn 按照 fewnet 的条件 进行稍微的更改， 使之符合 fewnetloss 的要求，
        且，在此基础上，要满足 DataParallel 对数据的 分割要求.
        
        Args:
            batch (List[Dict]): a list of batch size with each element being a dict of these entries:
              "image" (Union[Tensor, ndarray]): .shape == [3, Hi, Wi], different element may have different shape;
              "score_map" (List[ndarray]): a list with different size for different feature leve;
              "boxes" (ndarray): .shape == [num_tgt_boxes, 4], each element should be (cx, cy, w, h)
              "angle" (ndarray): .shape == [num_tgt_boxes, 1], each element should be (angle) with respect
                                  to specified angle version.
                                  
        Returns:
            results (Dict[str, Union(Tensor, List[Tensor])]): a dict that can meet the need of data parallel
               and also add a new key "num_tgt_boxes" with shape of [B, ], that could reflect the
               number of target boxes for each item in batch.
            
        Notes:
            1. boxes should be update since the shape of corresponding image may change;
            2. results[k] should be List[Tensor] or Dict[str, Tensor] and the first dimension of Tensor
               should be Batch_size so that it can be scattered properly by DataParallel.
            3. currently, pad is only performed on bottom-right in "image" and bottom in
               "boxes" and "angle".
        """
        from collections import OrderedDict
        
        result = OrderedDict()
        elem = batch[0]
        keys = elem.keys()
        
        # for image and score_map
        result["image"] = self.transform_image(  # [B, 3, max_h, max_w]
            [item["image"] for item in batch]
        )
        result["score_map"] = []
        feat_level_num = len(elem["score_map"])  # number of feature level
        for i in range(feat_level_num):
            result["score_map"].append(  # [B, max_hi, max_wi]
                self.transform_image(
                    [item["score_map"][i] for item in batch]
                )
            )
            
        # for boxes and angles
        result["num_tgt_boxes"] = default_convert([len(item["boxes"]) for item in batch])
        result["boxes"] = self.transform_boxes(
            [item["boxes"] for item in batch]
        )
        result["angle"] = self.transform_boxes(
            [item["angle"] for item in batch]
        )
        try:
            if torch.max(result["boxes"]) > 1:  # no normalization performed before
                max_H, max_W = result["image"].shape[-2:]
                result["boxes"][:, 0:-1:2] = result["boxes"][:, 0:-1:2] / max_W
                result["boxes"][:, 1::2] = result["boxes"][:, 1::2] / max_H
        except RuntimeError as e:
            if result["boxes"].shape[1] == 0:  # no boxes, nothing to do
                # print("no annotation for current box")
                pass  # result["boxes"].shape should be [B, -1, 4], -1 can be 0
            else:
                raise RuntimeError(e)
        
        # for others
        for k in keys:
            if k in ["image", "score_map", "boxes", "angle"]:
                continue
            result[k] = default_collate([item[k] for item in batch])
        
        return result
    
    @staticmethod
    def transform_boxes(boxes):
        """ boxes related """
        box_shapes = np.array([box.shape for box in boxes])
        max_num_tgt_boxes, _ = np.max(box_shapes, axis=0)  # _ is 4 or 1
        if not isinstance(boxes[0], torch.Tensor):
            boxes = default_convert(boxes)
        
        padded_boxes = torch.stack([
            F.pad(box, pad=(0, 0, 0, max_num_tgt_boxes - box.shape[0]))
            for box in boxes
        ], dim=0)  # [B, max_num_tgt_boxes, _]
        return padded_boxes
    
    @staticmethod
    def transform_image(images):
        img_shapes = np.array([image.shape for image in images])
        max_H, max_W = np.max(img_shapes, axis=0)[-2:]  # [max_H, max_W]
        if not isinstance(images[0], torch.Tensor):
            images = default_convert(images)  # transform to torch.Tensor
            
        padded_images = torch.stack([  # pad in bottom-right
            F.pad(img, pad=(0, max_W - img.shape[-1], 0, max_H - img.shape[1]),
                  mode="constant", value=0)
            for img in images
        ], dim=0)  # [B, 3, max_H, max_W]
        return padded_images
    
if __name__ == "__main__":
    pass
