"""
该文件的主要目的是， 生成 用于 patch extraction 训练所需要的 target.

这里 仅仅只是第一种实现方式， 采用的是 直接在 stride 为 4 的基础上，进行 缩放，生成。
但是，实际上，也可以在 original size 的地方生成，随后再 执行缩放
"""


from concern.config import Configurable, State

import cv2
import numpy as np

from shapely.geometry import Polygon, Point
import pyclipper  # 用以 Polygon 的缩放

import imgaug as ia  # resize
from .augmenter import AugmenterBuilder

import os
from scipy.special import entr as _entr
import warnings
from collections import Iterable
from copy import deepcopy
from math import ceil, floor


class MakePatchTarget(Configurable):
    """
    该类 用以 从 标注的文本区域 生成 PatchExtraction 所需要的 target，该 target
    随后参与 Loss 的具体计算
    """
    # target_map hyper-parameter
    shrink_ratio = State(default=0.4)  # follow the default value of DBNet
    # 下面三个 weight_* 是 target 生成中的 权重超参数
    weight_entropy = State(default=0.7)
    weight_positive = State(default=0.4)
    weight_negative = State(default=0.05)
    # 以 stride 来表征 不同的 feat levels
    # 目前 仅仅打算，按照 strides[0] 也就是 4 来进行 shrink text region, border, background
    # 的划分
    strides = State(default=[4, 8, 16])
    # prob_type: "gau" or "dis"
    prob_type = State(default="gau")
    min_valid_distance = State(default=1)
    
    # target_mask hyper-parameter
    # dist_anchor_points 与 max_half_patch_size 联合起来，控制 Patch 之间 IOU 的阈值
    dist_anchor_points = State(default=None)
    max_half_patch_size = State(default=None)
    iou_threshold = State(default=None)
    
    def __init__(self, debug=False, cmd={}, **kwargs):
        super().__init__(cmd, **kwargs)
        
        if not isinstance(self.strides, Iterable):
            self.strides = [self.strides] # 转换成 Iterable
        self.resize_scales = [1. / stride for stride in self.strides]
        
        # 对 self.prob_type 做出一定的判断
        self.prob_type = self.prob_type.lower()
        assert self.prob_type in ["gau", "dis"], (
            'error, prob_type can only be "gau" or "dis"'
        )
        
        # 对 target_mask 所需要的超参数 作出对应的 初始化
        assert len(list(filter(
                lambda x: x is not None,
                [self.dist_anchor_points, self.max_half_patch_size, self.iou_threshold]
            ))
        ) > 1, (
            "At least two should be specified for dist_anchor_points, "
            "max_half_patch_size and iou_threshold"
        )
        
        if self.dist_anchor_points is not None:
            if self.max_half_patch_size is None:
                self.max_half_patch_size = MakePatchTarget.gen_patch_anchor(
                    self.iou_threshold, dist_anchor_points=self.dist_anchor_points
                )
        elif self.max_half_patch_size is not None:
            if self.dist_anchor_points is None:
                self.dist_anchor_points = MakePatchTarget.gen_patch_anchor(
                    self.iou_threshold, max_half_patch_size=self.max_half_patch_size
                )
        else:
            pass  # 不需要做任何的操作
        
        # other
        self.debug = debug
        if "debug" in cmd:
            self.debug = cmd["debug"]
            
        warnings.filterwarnings("ignore")
      
    @staticmethod
    def gen_patch_anchor(iou_threshold, max_half_patch_size=None, dist_anchor_points=None):
        r""" 从 iou_threshold 和 任意的一个 max_half_patch_size,
        或者是 dist_anchor_points 生成另外一个
        """
        assert not (max_half_patch_size is None and dist_anchor_points is None), (
            "max_half_patch_size and dist_anchor_points can not be None simultaneously"
        )
        
        if max_half_patch_size is None:
            ret = ceil(
                (1 + iou_threshold) * dist_anchor_points / (2 - 2 * iou_threshold)
            )
            return ret
        else:
            ret = floor(
                (2 - 2 * iou_threshold) * max_half_patch_size / (1 + iou_threshold)
            )
            return ret
    
    @staticmethod
    def may_aug_polys(aug, img_shape, polys):
        import numpy as np
        
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
    def draw_prob_map_by_gau(canvas, polys, ignore_tags):
        r"""
        从实际效果上来看，这里 没有那么的 平滑，在 positive 与 border 或者 border 与 neg
        之间，存在 少量的 “黑点”， 可以认为是 高斯平滑 的一些 漏掉的点，最终带来的影响
        """
        assert canvas.max() == canvas.min() == 0
        assert len(polys) == len(ignore_tags)
    
        for poly, ignore_tag in zip(polys, ignore_tags):
            if ignore_tag:
                continue
            
            cv2.fillPoly(canvas, [poly.astype(np.int)], 1)
        canvas = cv2.GaussianBlur(canvas, (5, 5), 2)
        return canvas
    
    @staticmethod
    def distance_point2line(xs, ys, point_1, point_2):
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])
        # try:
        cosin = (square_distance - square_distance_1 - square_distance_2) / \
                (2 * np.sqrt((square_distance_1 * square_distance_2) + 1e-6))  # 引入 很小的 eps=1e-6
        # except RuntimeWarning as w:
        #     pass # 用以 debug
        
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        return result
    
    @staticmethod
    def draw_prob_map_by_dis(canvas, shrink_ratio, stride, polys, ignore_tags,
                             debug=False, min_valid_distance=3, eps = 0.4):
        r"""
        该函数 返回一个 包括了 positive, border, negative 的 canvas. 其中，
        canvas["positive"] = 1, canvas["negative"] = 0, 0 < canvas["border"] < 1
        
        positive 的定义，是 shrink(polys),
        border 的定义，是 shrink(polys) 与 dilate(polys) 之间,
        negative 的定义，就是在 dilated(polys) 之外
        
        shrink 和 dilate 的具体 distance 由 `shrink_ratio`, `poly_area` 以及 `stride` 来控制
        
        这里采用的是 distance 的方式 来计算 概率， 需要 shrink_ratio, poly_area, stride
        
        此外，这里会根据 shrink 的情况，来动态的更新 ignore_tags, 并且返回
        """
        assert canvas.max() == canvas.min() == 0
        assert len(polys) == len(ignore_tags)
        polys = deepcopy(polys)
        ignore_tags = deepcopy(ignore_tags)
        
        shrunk_canvas = np.zeros_like(canvas, dtype=np.float32)  # 0,1 的范围内
        border_canvas = np.zeros_like(canvas, dtype=np.float32)
        
        for i, (poly, ignore_tag) in enumerate(zip(polys, ignore_tags)):
            """ 分别对每一个 poly 进行处理 """
            if ignore_tag:
                continue
            
            # 1. 获取 dilated_poly 和 shrunk_poly，分别是 放大 和 缩小的结果
            polygon = Polygon(poly)
            distance = polygon.area * (1 - np.power(shrink_ratio, 2)) / polygon.length
            # distance /= stride  # 针对 stride 做一定的处理
            if distance < min_valid_distance:
                warnings.warn("distance is too small: {}".format(distance))
            
            subject = [tuple(l) for l in poly]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            dilated_poly = np.array(padding.Execute(distance)[0])
            shrunk_poly = np.array(padding.Execute(-distance)[0])
            if not shrunk_poly.size:  #
                ignore_tags[i] = 1
                continue
            
            # 2. 填充 shrunk_canvas
            cv2.fillPoly(shrunk_canvas, [shrunk_poly], 1)
            
            # 3. 填充 border_convas，使用 border 到 {shrunk}_poly 的距离 归一化后的结果
            #    归一化的方式，就是 d / (distance * 2),当然，这个到底赢不应该是 2，
            #    也可以进一步的 控制该参数
            x_max, y_max = np.max(dilated_poly, axis=0)
            x_min, y_min = np.min(dilated_poly, axis=0)
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            shrunk_poly[:, 0] = shrunk_poly[:, 0] - x_min
            shrunk_poly[:, 1] = shrunk_poly[:, 1] - y_min  # 将 shrunk_poly 转换到对应的坐标下
            xs = np.broadcast_to(
                np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
            ys = np.broadcast_to(
                np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))
            
            # 构建 distance_map,
            # distance_map[i,j] = (i, j) 这个坐标 到 poly 的最小距离， 稍微做了一下归一化
            distance_map = np.zeros(
                (shrunk_poly.shape[0], height, width), dtype=np.float32)
            for i in range(shrunk_poly.shape[0]):  # 对每一个边 分别进行处理
                j = (i + 1) % shrunk_poly.shape[0]
                absolute_distance = MakePatchTarget.distance_point2line(
                    xs, ys, shrunk_poly[i], shrunk_poly[j]
                )
                # np.clip 保证，仅仅计算了 border 的 distance
                distance_map[i] = np.clip(absolute_distance / (distance * 2), 0.3, 1) # eps = 0.01
            distance_map = distance_map.min(axis=0)
            cv2.fillPoly(distance_map, [shrunk_poly], 1)  # 保证 内部都是1， 从而在后面 1-x 都是0

            xmin_valid = min(max(0, x_min), border_canvas.shape[1] - 1)
            xmax_valid = min(max(0, x_max), border_canvas.shape[1] - 1)
            ymin_valid = min(max(0, y_min), border_canvas.shape[0] - 1)
            ymax_valid = min(max(0, y_max), border_canvas.shape[0] - 1)
            border_canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
                1 - distance_map[
                    ymin_valid - y_min:ymax_valid - y_max + height,
                    xmin_valid - x_min:xmax_valid - x_max + width],
                border_canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
            
        # ret step: canvas = border_canvas + shrunk_canvas
        canvas = border_canvas + shrunk_canvas
        if debug:
            border_canvas_path = os.path.join("debug", "border_canvas.jpg")
            shrunk_canvas_path = os.path.join("debug", "shrunk_canvas.jpg")
            canvas_path = os.path.join("debug", "canvas.jpg")
            cv2.imwrite(border_canvas_path, (border_canvas * 255).astype(np.int))
            cv2.imwrite(shrunk_canvas_path, (shrunk_canvas * 255).astype(np.int))
            cv2.imwrite(canvas_path, (canvas * 255).astype(np.int))
        
        return np.clip(canvas, 0, 1), ignore_tags

    def __call__(self, data, *args, **kwrags):
        r"""
        从 polygonal annotation 的基础上，生成 所需要的 target，当前仅仅生成 stride 为 4 的 target,
        这里认为， stride 为 8 和 16 的都需要通过 TransposeConv 来完成上采样到 stride 为 4，
        再来 计算 Loss 和 region 的 selection.
        
        data: {
          "filename", "image", "polygons", "ignore_tags", "shape", "is_training"
        }
        增加 "resized_image", "resized_polys", "target_map“， "target_mask"
        """
        # 1. resize image and corresponding poly
        augmenter_args = [
            ['Resize', [self.resize_scales[0], self.resize_scales[0]]]
        ]
        augmenter = AugmenterBuilder().build(augmenter_args)
        resizer = augmenter.to_deterministic()  # 重要
        data["resized_image"] = resizer.augment_image(data["image"])
        data["resized_polys"] = MakePatchTarget.may_aug_polys(
            resizer, data["image"].shape, data["polygons"]
        )  # List[np.ndarray]
        
        # 2. obtain probability map, prob_map.shape == resized_image.shape
        H, W = data["resized_image"].shape[:2]
        prob_map = np.zeros([H, W], dtype=np.float32)
        if self.prob_type == "gau":
            prob_map = MakePatchTarget.draw_prob_map_by_gau(
                prob_map, data["resized_polys"], data["ignore_tags"]
            )
        else:
            prob_map, data["ignore_tags"] = MakePatchTarget.draw_prob_map_by_dis(
                prob_map, self.shrink_ratio, self.strides[0],  # 目前使用的是 strides[0]
                data["resized_polys"], data["ignore_tags"],
                self.debug, self.min_valid_distance, self.weight_positive
            )
        
        # 3. calculate information entropy
        neg_prob_map = 1 - prob_map  # prob_map for negative label
        entropy_map = _entr(prob_map) + _entr(neg_prob_map)
        entropy_map /= np.max(entropy_map)  # linear scaling to [0, 1]
        pos_map = (prob_map == 1).astype(np.float)
        neg_map = (prob_map == 0).astype(np.float)

        target_map = (
                self.weight_entropy * entropy_map + self.weight_positive * pos_map +
                self.weight_negative * neg_map
        )
        data["target_map"] = target_map
        
        # 4. target_mask generation
        target_mask = np.zeros([H, W], dtype=np.int32)
        xs = np.arange(start=self.max_half_patch_size - 1, stop=W - self.max_half_patch_size + 1,
                       step=self.dist_anchor_points, dtype=np.int)
        ys = np.arange(start=self.max_half_patch_size - 1, stop=H - self.max_half_patch_size + 1,
                       step=self.dist_anchor_points, dtype=np.int)
        
        # xs = np.append(xs, W - self.max_half_patch_size + 1)
        # ys = np.append(ys, H - self.max_half_patch_size + 1)
        
        coords = np.meshgrid(xs, ys)  # 这里生成了 所有的 anchor point 的坐标
        target_mask[coords[0], coords[1]] = 1
        data["target_mask"] = target_mask
        data["target_coords"] = coords
        # coods[0] 表示 x, coords[1] 表示 y, coords[i].shape == [H/4, W/4]
        
        if self.debug:
            data["target_mask"] = (data["target_mask"] * 255).astype(np.int32)
            data["target_mask"] = np.tile(data["target_mask"], [3, 1, 1]).transpose([1, 2, 0])
            
            fix_i, fix_j = len(xs) // 2, len(ys) // 2
            offsets = [[0, 0], [1, 0], [1, 1], [0, 1]]
            colors = [ (0, 0, 255), (0, 255, 0), (255, 0, 0), (128, 128, 128)]
            
            for offset, color in zip(offsets, colors):
                ri, rj = fix_i + offset[0], fix_j + offset[1]
                center_coord_x, center_coord_y = coords[0][ri, rj], coords[1][ri, rj]
                top_left, right_bottom = (
                    (center_coord_x - self.max_half_patch_size + 1,
                     center_coord_y - self.max_half_patch_size + 1),
                    (center_coord_x + self.max_half_patch_size - 1,
                     center_coord_y + self.max_half_patch_size - 1)
                )
                data["target_mask"] = cv2.rectangle(data["target_mask"],
                                                    top_left, right_bottom, color, 1)
                if hasattr(data["target_mask"], "get"):
                    data["target_mask"] = data["target_mask"].get()
                data["target_mask"][center_coord_x, center_coord_y] = color
            
        
        if self.debug:
            # write (target_map) and (resized_image with resized_polys)
            debug_imgname = "resized_" + os.path.basename(data["filename"])
            debug_imgpath = os.path.join("debug", debug_imgname)
            debug_target_name = "target_" + os.path.basename(data["filename"])
            debug_target_path = os.path.join("debug", debug_target_name)
            debug_target_mask_name = "target_mask_" + os.path.basename(data["filename"])
            debug_target_mask_path = os.path.join("debug", debug_target_mask_name)
            
            debug_sum_path = os.path.join("debug", "img_sum.jpg")
            
            for i in range(len(data["polygons"])):
                resized_poly = data["resized_polys"][i].astype(np.int)
                ignore = data["ignore_tags"][i]
                if ignore:
                    color = (255, 0, 0)  # depict ignorable polygons in blue
                else:
                    color = (0, 0, 255)  # depict polygons in red
                    
                cv2.polylines(data["resized_image"], [resized_poly], True, color, 2)
            
            cv2.imwrite(debug_imgpath, data["resized_image"])
            cv2.imwrite(debug_target_path, (data["target_map"] * 255).astype(np.int))
            cv2.imwrite(
                debug_sum_path,
                data["resized_image"] + (data["target_map"] * 255).astype(np.int)[:, :, np.newaxis]
            )
            cv2.imwrite(
                debug_target_mask_path, data["target_mask"]
            )

        return data  # data should be returned
