"""
This file generates target for "Few Could be Better than All".

这里牵扯到了 几个相关的 (x, y) 的坐标对:
1. data["polygons"] 的 (x, y): x 为 从左到右, 对应 width， y 为 从上到下, 对应 height;
2. resizer 中的 .may_aug_polys 中的 (x, y): x 为从左到右, ...width， y 为从上到下, ...height;
3. poly2obb 的 (x, y): x 为从左到右, ...width， y 为从上到下, ...height;
4. normalize 过程中的 (x, y): x 为从左到右, ...width， y 为从上到下, ...height;

此外， H, W = image.shape[0], image.shape[1]

在明确 np.meshgrid 采用的 indexing 是 "xy" 之后，统一规整到 (x, y) 而不是 (r, c) 的表示.

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
    need_norm = State(default=False)  # normalize for (x, y, w, h), this will be performed in collate_fn
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
                 angle_version="le135", need_norm=False,  # angle and coordinates related
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
        
        # for angle
        assert self.angle_version in ["le135", "le90", "oc"], (
            "supported angle version are: {},"
            "However, your angle_version: {}".foramt(
                ["le135", "le90", "oc"], self.angle_version
            )
        )
        self.angle_minmax = dict(
            oc=(0, np.pi / 2), le135=(-np.pi / 4, np.pi * 3 / 4),
            le90=(-np.pi / 2, np.pi / 2)
        )[self.angle_version]
    
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
        rboxes = np.array(rboxes, dtype=np.float32).reshape([-1, 5])  # [cx, cy, w, h, theta]
        
        if self.need_norm:  # 这里的归一化，最大值，不是 data["shape"], 而是 data["image"].shape
            max_H, max_W = data["image"].shape[:-1]  # DB do not convert to (C, H, W) to keep (H, W, C)
            rboxes[:, 0:-1:2] = rboxes[:, 0:-1:2] / max_W  # increase robustness of indexing
            rboxes[:, 1::2] = rboxes[:, 1::2] / max_H
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
    def point2line(xs, ys, point_1, point_2):
        """Compute the distance from point to a line. This function is adapted
        from mmocr -- base_textdet_targets.py
        Args:
            xs (ndarray): The x coordinates of size hxw.  -- width
            ys (ndarray): The y coordinates of size hxw.  -- height
            point_1 (ndarray): The first point with shape 1x2.
            point_2 (ndarray): The second point with shape 1x2.  -- (x, y) coordinates
        Returns:
            result (ndarray): The distance matrix of size hxw.
        """
        # suppose a triangle with three edge abc with c=point_1 point_2
        # a^2
        a_square = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        # b^2
        b_square = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        # c^2
        c_square = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] -
                                                                  point_2[1])
        # -cosC=(c^2-a^2-b^2)/2(ab)
        neg_cos_c = (
            (c_square - a_square - b_square) /
            (np.finfo(np.float32).eps + 2 * np.sqrt(a_square * b_square)))
        # sinC^2=1-cosC^2
        square_sin = 1 - np.square(neg_cos_c)
        square_sin = np.nan_to_num(square_sin)
        # distance=a*b*sinC/c=a*h/c=2*area/c
        result = np.sqrt(a_square * b_square * square_sin /
                         (np.finfo(np.float32).eps + c_square))
        # set result to minimum edge if C<pi/2
        result[neg_cos_c < 0] = np.sqrt(np.fmin(a_square,
                                                b_square))[neg_cos_c < 0]
        return result
    
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
            xs, ys = np.meshgrid(xs, ys, indexing="xy")  # meshgrid, xs - width, ys - height
            for i in range(len(poly)):
                j = (i + 1) % len(poly)
                point_1, point_2 = poly[i], poly[j]
                dist_point_line = self.point2line(xs, ys, point_1, point_2)
                dist_point_lines[i] = dist_point_line
            dist_point_lines = dist_point_lines.min(axis=0)  # [tiny_H, tiny_W]
            # Though the coordinates is different, However,
            # the value in dist_point_lines can also reflect the distance
            # from this point to line.
            
            # obtain first-step xs, ys, max_radius
            positive_point_lines = dist_point_lines * positive_mask  # [tiny_H, tiny_W]
            nk = min(np.sum(positive_mask).astype(np.int32), self.max_num_gau_center)
            nk_ind, nk_val = self.topk_by_partition(
                positive_point_lines.flatten(), nk, axis=0, ascending=False,
            )
            nk_xs, nk_ys = nk_ind % tiny_W, nk_ind // tiny_W  # nk_xs: width, nk_ys: height
            t_mask = nk_val > self.min_radius_limit
            nk_val = nk_val[t_mask]  # now nk_val only contain value greater than min_radius_limit
            nk_xs, nk_ys = nk_xs[t_mask], nk_ys[t_mask]  # check
            
            # generate center and radius
            # t_scale = 0.5 + np.random.rand(*nk_val.shape) * 0.5
            t_scale = np.ones_like(nk_val)
            poly_radius = np.ceil(
                self.min_radius_limit + t_scale * (nk_val - self.min_radius_limit)
            )
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
        ind = np.argpartition(input, k - 1, axis=axis)  # use (k - 1) instead of k for extreme situation
        ind = np.take(ind, np.arange(k), axis=axis)  # k non-sorted indices
        input = np.take_along_axis(input, ind, axis=axis)  # k non-sorted values
    
        # sort within k elements
        ind_part = np.argsort(input, axis=axis)
        ind = np.take_along_axis(ind, ind_part, axis=axis)
        if not ascending:
            input *= -1
        val = np.take_along_axis(input, ind_part, axis=axis)
        return ind, val
