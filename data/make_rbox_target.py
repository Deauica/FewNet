"""
Generate Rotated bbox target

Reference: MMROTATE 的实现
"""

from concern.config import State, Configurable
from .utils import poly2obb_np, norm_angle

import math
import numpy as np
import cv2


class MakeRBoxTarget(Configurable):
    """ 从 QUAD 的标注内容，转换到 (x, y, w, h, theta) 的标注方式 """
    version = State(default="le135")
    need_norm = State(default=True)
    
    def __init__(self, debug=False, cmd=None, **kwargs):
        super(MakeRBoxTarget, self).__init__(cmd, **kwargs)
        
        self.debug = debug
        if "debug" in cmd:
            self.debug = cmd["debug"]
    
    def __call__(self, data, *args, **kwargs):
        """实现 QUAD 的标注方式，转换到 (x, y, w, h, theta) 的具体方法接口,
        该类，应该作为 修改标注内容的最后一个步骤，这主要是因为，当前的 各种
        针对 标注的增强方式，都是针对 QUAD 的方式来实现的。
        
        此外，需要注意的是，这里的 (x, y, w, h, theta) 都是 归一化 后的结果, 归一化后：
        (x / width, y / height, w / width, h / height, theta)
        为了进一步的 修改，这里的 width, height 都改变成之前的一半
        
        Required Keys:
          "polygons": [N, 8] ndarray
          "ignore_tags": [N] ndarray
          
        Modified keys:
          "polygons": [N, 5] ndarray
        """
        rboxes = []
        for _, polygon in enumerate(data["polygons"]):
            polygon = np.reshape(polygon, [8])
            rbox = poly2obb_np(polygon, version=self.version)
            rboxes.append(rbox)
        rboxes = np.array(rboxes, dtype=np.float32)
        
        if self.debug:
            import os
            from decoders.utils import obb2poly_np
            rboxes_prob = []
            for rbox in rboxes:
                rbox_prob = (*rbox, 1)
                rboxes_prob.append(rbox_prob)
            rboxes_prob = np.array(rboxes_prob)
    
            polygons_transformed = obb2poly_np(rboxes_prob, version=self.version)
            polygons_transformed = np.array(polygons_transformed[:, :-1], dtype=np.float32)
            cv2.polylines(data["image"], polygons_transformed.reshape([-1, 4, 2]).astype(np.int32),
                          True, (255, 0, 0), 2)
            
            cv2.imwrite(
                os.path.join("debug", "rotated_{}".format(os.path.basename(data["filename"]))),
                data["image"]
            )
        
        assert rboxes.shape[-1] == 5, (
            "descriptor for each rbox should be: (cx, cy, w, h, theta),"
            "However, Your rboxes.shape: {} and rboxes[0] is: {}".format(
                rboxes.shape, rboxes[0]
            )
        )
        if self.need_norm:  # ? original shape ? 显然，这里 data["shape"] 是 orig-shape, 这里是错误的
            rboxes[:, 0:2:3] = rboxes[:, 0:2:3] / (data["shape"][1] / 2)  # x / width
            rboxes[:, 1:2:4] = rboxes[:, 1:2:4] / (data["shape"][0] / 2)  # y / height
        data["polygons"] = rboxes
        return data
    
    def pred2polys(self):
        pass