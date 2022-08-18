"""
This file generates target for "Few Could be Better than All".

这里牵扯到了 几个相关的 (x, y) 的坐标对:
1. data["polygons"] 的 (x, y): x 为 从左到右, 对应 width， y 为 从上到下, 对应 height;
2. resizer 中的 .may_aug_polys 中的 (x, y): x 为从左到右, ...width， y 为从上到下, ...height;
3. poly2obb 的 (x, y): x 为从左到右, ...width， y 为从上到下, ...height;
4. normalize 过程中的 (x, y): x 为从左到右, ...width， y 为从上到下, ...height;

此外， H, W = image.shape[0], image.shape[1]

在明确 np.meshgrid 采用的 indexing 是 "xy" 之后，统一规整到 (x, y) 而不是 (r, c) 的表示.

显然，当前这个 距离计算，有些问题。 -- 没有获取到 真实的距离，很小。

目前的效果，不是太理想。 和 作者 所设想的 有一定的出入。
"""
import cv2
import os

import torch

from concern.config import State, Configurable

from .utils import gen_gaussian_target  # only support torch version
from .augmenter import AugmenterBuilder
from .utils import poly2obb_np

import inspect
from collections import abc, OrderedDict
import numpy as np


class MakeFewNetTarget(Configurable):
    # angle related
    angle_version = State(default="le135")  # version for the angle
    need_norm = State(default=True)  # normalize for (x, y, w, h)
    bg_value = State(default=0)
    fg_value = State(default=0.7)  # fill background/foreground with {bg, fg}_value, respectively
    
    # gaussian related
    min_radius_limit = State(default=1)
    coef_gaussian = State(default=1)  # parameter for gaussian heatmap's generation
    max_num_gau_center = State(default=100)
    
    # resizer related
    strides = State(default=(8, 16, 32))
    resizer_args = State(
        default=[
            [['Resize', [1./8,  1./8]]],
            [["Resize", [1./16, 1./16]]],
            [["Resize", [1./32, 1./32]]]
        ]
    )  # resizer args
    resizer_builder = State(
        default=AugmenterBuilder().build  # callable
    )  # resizer builder
    
    def __init__(self,
                 angle_version="le135", need_norm=True,  # angle related
                 bg_value=0, fg_value=0.78,
                 min_radius_limit=0, coef_gaussian=1, max_num_gau_center=50,  # gaussian related
                 strides=(8, 16, 32), resizer=None, resizer_builder=None, resizer_args=None,  # resizer
                 debug=False, cmd=None, **kwargs  # dbnet config related
                 ):
        if cmd is None:
            cmd = {}
        
        super(MakeFewNetTarget, self).__init__(cmd=cmd, **kwargs)  # perform dbnet config
        
        argvalues = inspect.getfullargspec(self.__init__)
        for param, val in zip(argvalues.args[1:], argvalues.defaults):
            if not hasattr(self, param):
                setattr(self, param, val)
            elif val is not None:
                setattr(self, param, val)
            else:
                pass
        self.debug = cmd.get("debug", debug)
        
        # resizer initialization
        if self.resizer is None:
            assert callable(self.resizer_builder) or inspect.isclass(self.resizer_builder), (
                "resizer_builder should be a **builder** function or a **class definition**,"
                "However, your resizer_builder is: {}".format(self.resizer_builder)
            )
            self.resizer = [
                self.resizer_builder(aug_args) for aug_args in self.resizer_args
            ]
            self.resizer = [
                augmenter.to_deterministic() if hasattr(augmenter, "to_deterministic") else augmenter
                for augmenter in self.resizer
            ]
        
        # check the validation of `self.resizer`
        if not isinstance(self.resizer, abc.Iterable):
            self.resizer = [self.resizer]
        assert len(self.resizer) == len(self.strides), (
            "Usually, each stride should correspond to one resizer, "
            "However, "
            "the length of your resizer is: {}, while your strides being {}".format(
                len(self.resizer), len(self.strides)
            )
        )
    
    def __call__(self, data, *args, **kwargs):
        """ Generate targets for Few Could be Better than All.
        
        Args:
            data (Dict[str, Any]): a dict containing at least these entries:
              "polygons":
              "ignore_tags":
              "shape": original shape for this image file.
            
        Returns:
            data (Dict[str, Any]): input `data` with these newly added entries:
              "boxes": [num_tgt_boxes_sample, 4], (x, y, w, h)
              "angle": [num_tgt_boxes_sample, 1], value depends on the `angle_version`
              "score_map": List[ndarray], score_map[0, 1, 2] is for stride (8, 16, 32), respectively.
                 score_map[i].shape == [H/stride, W/stride].
        """
        # step 1. generate rotated box information
        rboxes = list()
        for _, polygon in enumerate(data["polygons"]):
            polygon = np.reshape(polygon, [8])
            rbox = poly2obb_np(polygon, version=self.angle_version)
            rboxes.append(rbox)
        rboxes = np.array(rboxes, dtype=np.float32)
        
        assert rboxes.shape[-1] == 5, (
            "descriptor for each rbox should be: (cx, cy, w, h, theta),"
            "However, Your rboxes.shape: {} and rboxes[0] is: {}".format(
                rboxes.shape, rboxes[0]
            )
        )
        if self.need_norm:  # 这里的归一化，最大值，不是 data["shape"], 而是 data["image"].shape
            max_H, max_W = data["image"].shape[-2:]
            rboxes[:, [0, 2]] = rboxes[:, [0, 2]] / max_W
            rboxes[:, [1, 3]] = rboxes[:, [1, 3]] / max_H
            # theta 不需要做 归一化，直接采用 弧度制就可以了
        
        data["boxes"] = rboxes[:, :4]
        data["angle"] = rboxes[:, 4:]
        
        # step 2. generate score maps
        score_maps = list()
        for aug in self.resizer:
            src_canvas = np.full(data["image"].shape[0:2], self.bg_value, dtype=np.float32)
            aug_canvas = aug.augment_image(src_canvas)  # 这里采用的是 imgaug 的方式
            aug_polys = self.may_aug_polys(aug, data["image"].shape, data["polygons"])
            score_map = self.gen_single_score_map(aug_canvas, aug_polys)
            score_maps.append(score_map)
        data["score_map"] = score_maps
        
        if self.debug:
            # 可视化 score_maps
            for _, stride in enumerate(self.strides):
                score_map = cv2.applyColorMap(
                    (data["score_map"][_] * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                cv2.imwrite(
                    os.path.join("debug", f"score_map_{stride}.jpg"),
                    score_map
                )

        return data

    @staticmethod
    def may_aug_polys(aug, img_shape, polys):
        import imgaug as ia   # import imgaug
        aug_polys = []
        for poly in polys:
            keypoints = [ia.Keypoint(p[0], p[1]) for p in poly]
            keypoints = aug.augment_keypoints(
                [ia.KeypointsOnImage(keypoints, shape=img_shape)]
            )[0].keypoints
            t_poly = [(p.x, p.y) for p in keypoints]
            aug_polys.append(np.array(t_poly))
    
        return aug_polys

    @staticmethod
    def distance_point2line(xs, ys, p1, p2):
        """
        Modified from:
        https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
        """
        px, py = p1[0] - p2[0], p1[1] - p2[1]  # xs, ys is row-col type
        norm = px * px + py * py + 1e-6  # eps = 1e-6
        u = ((xs - p1[0]) * px + (ys - p1[1]) * py) / float(norm)
        np.clip(u, 0, 1)
        x = p1[0] + u * px
        y = p1[1] + u * py
        dx = x - xs
        dy = y - ys
        return np.sqrt(dx * dx + dy * dy)
    
    def gen_single_score_map(self, canvas, polys):
        import cv2
        # step 1. fill foreground with pre-defined fg_value
        polys = [
            poly.astype(np.int32) for poly in polys
        ]
        cv2.fillPoly(canvas, polys, self.fg_value)
        
        # step 2. generate gaussian candidate -- list(((x, y), radius))
        gaussian_candidates = []
        for poly in polys:  # 分别在 每一个 poly 中单独获取 随机点
            (x_min, y_min), (x_max, y_max) = np.min(poly, axis=0), np.max(poly, axis=0)
            poly[:, 0] = poly[:, 0] - x_min
            poly[:, 1] = poly[:, 1] - y_min
            
            # generate positive_mask -- tiny_coordinate
            tiny_H, tiny_W = y_max - y_min + 1, x_max - x_min + 1
            tiny_mask = np.zeros([tiny_H, tiny_W])
            cv2.fillPoly(tiny_mask, [poly], 1)  # 1 for positive, 0 for negative
            t_mask = np.random.binomial(
                n=1, p=0.5, size=tiny_mask.shape
            )
            positive_mask = t_mask * tiny_mask
            
            # generate distance array
            # [num_lines, dist_from_line_to_point(2D)] -> [2D]
            dist_point_lines = np.zeros([len(poly), *tiny_mask.shape])
            xs = np.arange(start=0, stop=tiny_W)  # x, ... width
            ys = np.arange(start=0, stop=tiny_H)  # y, ... height
            xs, ys = np.meshgrid(xs, ys, indexing="xy")  # meshgrid, [height, width] for ret xs and ys
            for i in range(len(poly)):
                j = (i + 1) % len(poly)
                point_1, point_2 = poly[i], poly[j]
                dist_point_line = self.distance_point2line(xs, ys, point_1, point_2)
                dist_point_lines[i] = dist_point_line
            dist_point_lines = dist_point_lines.min(axis=0)  # [tiny_H, tiny_W]
            
            # obtain first-step xs, ys, max_radius
            positive_point_lines = dist_point_lines * positive_mask  # [tiny_H, tiny_W]
            nk = min(np.sum(positive_mask).astype(np.int), self.max_num_gau_center)
            nk_ind, nk_val = self.topk_by_partition(
                positive_point_lines.flatten(), nk, axis=0, ascending=False,
            )
            nk_xs, nk_ys = nk_ind % tiny_W, nk_ind // tiny_W  # xs, ys
            t_mask = nk_val > self.min_radius_limit
            nk_val = nk_val[t_mask]  # now nk_val only contain value greater than min_radius_limit
            nk_xs, nk_ys = nk_xs[t_mask], nk_ys[t_mask]
            
            # generate center and radius
            t_scale = 0.5 + np.random.rand(*nk_val.shape) * 0.5
            poly_radius = self.min_radius_limit + t_scale * (nk_val - self.min_radius_limit)
            poly_gau_candinates = [
                ((xs + x_min, ys + y_min), radius)  # radius should be integer
                for xs, ys, radius in zip(nk_xs, nk_ys, poly_radius.astype(np.int32))
            ]
            gaussian_candidates.extend(poly_gau_candinates)  # extend gaussian_candidates
        
        if self.debug:
            pass
            
        # step 3. generate single score map
        for (x, y), radius in gaussian_candidates:
            canvas = gen_gaussian_target(
                torch.as_tensor(canvas), (x, y), radius).cpu().numpy()
            
        return canvas

    @staticmethod
    def topk_by_partition(input, k, axis=None, ascending=True):
        """
        Inherited from: https://hippocampus-garden.com/numpy_topk/
        """
        if not ascending:
            input *= -1
        ind = np.argpartition(input, k, axis=axis)
        ind = np.take(ind, np.arange(k), axis=axis)  # k non-sorted indices
        input = np.take_along_axis(input, ind, axis=axis)  # k non-sorted values
    
        # sort within k elements
        ind_part = np.argsort(input, axis=axis)
        ind = np.take_along_axis(ind, ind_part, axis=axis)
        if not ascending:
            input *= -1
        val = np.take_along_axis(input, ind_part, axis=axis)
        return ind, val