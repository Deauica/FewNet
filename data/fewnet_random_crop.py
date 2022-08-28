"""
RandomCrop in mmocr,

This class is adapted from:
  https://github.com/open-mmlab/mmocr/blob/dev-1.x/mmocr/datasets/transforms/ocr_transforms.py
"""

import numpy as np
from typing import Tuple, Dict, List, Union, Optional
from shapely.geometry import Polygon
from concern.config import State, Configurable

import cv2


def poly2bbox(polygon) -> np.array:
    """Converting a polygon to a bounding box.
    Args:
         polygon (ArrayLike): A polygon. In any form can be converted
             to an 1-D numpy array. E.g. list[float], np.ndarray,
             or torch.Tensor. Polygon is written in
             [x1, y1, x2, y2, ...].
     Returns:
         np.array: The converted bounding box [x1, y1, x2, y2]
    """
    assert len(polygon) % 2 == 0
    polygon = np.array(polygon, dtype=np.float32)
    x = polygon[::2]
    y = polygon[1::2]
    return np.array([min(x), min(y), max(x), max(y)])


def is_poly_inside_rect(poly, rect: np.ndarray) -> bool:
    """Check if the polygon is inside the target region.
        Args:
            poly (ArrayLike): Polygon in shape (N, ).
            rect (ndarray): Target region [x1, y1, x2, y2].
        Returns:
            bool: Whether the polygon is inside the cropping region.
        """

    poly = poly2shapely(poly)
    rect = poly2shapely(bbox2poly(rect))
    return rect.contains(poly)


def bbox2poly(bbox) -> np.array:
    """Converting a bounding box to a polygon.
    Args:
        bbox (ArrayLike): A bbox. In any form can be accessed by 1-D indices.
         E.g. list[float], np.ndarray, or torch.Tensor. bbox is written in
            [x1, y1, x2, y2].
    Returns:
        np.array: The converted polygon [x1, y1, x2, y1, x2, y2, x1, y2].
    """
    assert len(bbox) == 4
    return np.array([
        bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]
    ])


def poly2shapely(polygon) -> Polygon:
    """Convert a polygon to shapely.geometry.Polygon.
    Args:
        polygon (ArrayLike): A set of points of 2k shape.
    Returns:
        polygon (Polygon): A polygon object.
    """
    polygon = np.array(polygon, dtype=np.float32)
    assert polygon.size % 2 == 0 and polygon.size >= 6

    polygon = polygon.reshape([-1, 2])
    return Polygon(polygon)


class RandomCropInstance(Configurable):
    """Randomly crop images and make sure to contain at least one intact
    instance.
    
    Args:
        min_crop_side_ratio (float): The ratio of the shortest edge of the cropped
            image to the original image size.
    """
    size = State(default=(512, 512))
    min_crop_side_ratio = State(default=0.1)
    require_original_image = State(default=False)
    
    def __init__(self, min_crop_side_ratio: float = 0.1, cmd={}, **kwargs) -> None:
        super(RandomCropInstance, self).__init__(cmd=cmd, **kwargs)
        
        self.min_crop_side_ratio = min_crop_side_ratio
        self.debug = cmd.get("debug", False)
        
        if not 0. <= self.min_crop_side_ratio <= 1.:
            raise ValueError('`min_crop_side_ratio` should be in range [0, 1],')

    def _sample_valid_start_end(self, valid_array: np.ndarray, min_len: int,
                                max_start_idx: int,
                                min_end_idx: int) -> Tuple[int, int]:
        """Sample a start and end idx on a given axis that contains at least
        one polygon. There should be at least one intact polygon bounded by
        max_start_idx and min_end_idx.
        Args:
            valid_array (ndarray): A 0-1 mask 1D array indicating valid regions
                on the axis. 0 indicates text regions which are not allowed to
                be sampled from.
            min_len (int): Minimum distance between two start and end points.
            max_start_idx (int): The maximum start index.
            min_end_idx (int): The minimum end index.
        Returns:
            tuple(int, int): Start and end index on a given axis, where
            0 <= start < max_start_idx and
            min_end_idx <= end < len(valid_array).
        """
        assert isinstance(min_len, int)
        assert len(valid_array) > min_len

        start_array = valid_array.copy()
        max_start_idx = min(len(start_array) - min_len, max_start_idx)
        start_array[max_start_idx:] = 0
        start_array[0] = 1
        diff_array = np.hstack([0, start_array]) - np.hstack([start_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        start = np.random.randint(region_starts[region_ind],
                                  region_ends[region_ind])

        end_array = valid_array.copy()
        min_end_idx = max(start + min_len, min_end_idx)
        end_array[:min_end_idx] = 0
        end_array[-1] = 1
        diff_array = np.hstack([0, end_array]) - np.hstack([end_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        # Note that end index will never be region_ends[region_ind]
        # and therefore end index is always in range [0, w+1]
        end = np.random.randint(region_starts[region_ind],
                                region_ends[region_ind])
        return start, end

    def _sample_crop_box(self, img_size: Tuple[int, int],
                         results: Dict) -> np.ndarray:
        """Generate crop box which only contains intact polygon instances with
        the number >= 1.
        Args:
            img_size (tuple(int, int)): The image size (h, w).
            results (dict): The results dict.
        Returns:
            ndarray: Crop area in shape (4, ).
        """
        assert isinstance(img_size, tuple)
        h, w = img_size[:2]

        # polygons = results['gt_polygons']
        polygons = [
            np.array(line["points"])
            for line in results["polys"]
        ]
        if not polygons:
            return np.array([0, 0, w, h])  # return 0 directly

        # Crop box can be represented by any integer numbers in
        # range [0, w] and [0, h]
        x_valid_array = np.ones(w + 1, dtype=np.int32)
        y_valid_array = np.ones(h + 1, dtype=np.int32)
        
        # Randomly select a polygon that must be inside
        # the cropped region
        kept_poly_idx = np.random.randint(0, len(polygons))
        for i, polygon in enumerate(polygons):
            polygon = polygon.reshape((-1, 2))

            clip_x = np.clip(polygon[:, 0], 0, w)
            clip_y = np.clip(polygon[:, 1], 0, h)
            min_x = np.floor(np.min(clip_x)).astype(np.int32)
            min_y = np.floor(np.min(clip_y)).astype(np.int32)
            max_x = np.ceil(np.max(clip_x)).astype(np.int32)
            max_y = np.ceil(np.max(clip_y)).astype(np.int32)

            x_valid_array[min_x:max_x] = 0
            y_valid_array[min_y:max_y] = 0

            if i == kept_poly_idx:
                max_x_start = min_x
                min_x_end = max_x
                max_y_start = min_y
                min_y_end = max_y

        min_w = int(w * self.min_crop_side_ratio)
        min_h = int(h * self.min_crop_side_ratio)

        x1, x2 = self._sample_valid_start_end(x_valid_array, min_w,
                                              max_x_start, min_x_end)
        y1, y2 = self._sample_valid_start_end(y_valid_array, min_h,
                                              max_y_start, min_y_end)

        return np.array([x1, y1, x2, y2])

    def _crop_img(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop image given a bbox region.
            Args:
                img (ndarray): Image.
                bbox (ndarray): Cropping region in shape (4, )
            Returns:
                ndarray: Cropped image.
        """
        assert img.ndim == 3
        h, w, _ = img.shape
        assert 0 <= bbox[1] < bbox[3] <= h
        assert 0 <= bbox[0] < bbox[2] <= w
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def transform(self, results: Dict) -> Dict:
        """Applying random crop on results.
        Args:
            results (dict): Result dict contains the data to transform.
        Returns:
            dict: The transformed data.
        """
        # pre-process for negative coordinates
        img = results["image"]
        img_H, img_W = img.shape[:2]  # [H, W, 3]
        polygons = [
            np.array(line["points"]).reshape(-1, 2)
            for line in results["polys"]
        ]
        if polygons:  # polygons should exists
            max_polyX, max_polyY = np.max(np.array([
                np.max(polygon, axis=0) for polygon in polygons
            ]), axis=0)
            min_polyX, min_polyY = np.min(np.array([
                np.min(polygon, axis=0) for polygon in polygons
            ]), axis=0)
            pad_left, pad_right = -min(min_polyX, 0), max(img_W - 1, max_polyX) - img_W + 1
            pad_top, pad_bottom = -min(min_polyY, 0), max(img_H - 1, max_polyY) - img_H + 1
            img = np.pad(img, np.ceil(
                np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            ).astype(np.int32))  # 0 pad is okay
            polygons = [
                polygon + (pad_left, pad_top)
                for polygon in polygons
            ]
            results["image"] = img
            for i in range(len(results["polys"])):  # update poly
                results["polys"][i]["points"] = polygons[i].tolist()
                
        # select cropped region and check whether text should be remained
        # current `img` should be the cropped image
        # and `lines` contain the corresponding text region
        crop_box = self._sample_crop_box(results["image"].shape, results)
        img = self._crop_img(results["image"], crop_box)
        
        crop_x = crop_box[0]
        crop_y = crop_box[1]
        crop_w = crop_box[2] - crop_box[0]
        crop_h = crop_box[3] - crop_box[1]
        ori_lines, lines = results["polys"], []
        for line in results['polys']:
            poly = np.array(line["points"]).reshape(-1, 2)
            poly = (poly - (crop_x, crop_y)).flatten()  # no mulitplying here
            if is_poly_inside_rect(poly, [0, 0, crop_w, crop_h]):
                lines.append({**line, "points": poly.reshape([-1, 2])})
        
        # adapt `img` and `lines` for target size
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        padimg = np.zeros(  # ensure the shape is okay by proper padding
            (self.size[1], self.size[0], img.shape[2]), img.dtype)
        padimg[:h, :w] = cv2.resize(img, (w, h))
        img = padimg
        for i in range(len(lines)):   # modify the coordinates for line
            lines[i]["points"] = (lines[i]["points"] * scale).tolist()
            
        # assignment
        if not self.require_original_image:
            results["image"] = img  # img after cropped
        
        results["scale_w"], results["scale_h"] = scale, scale
        results['polys'] = lines
        results["lines"] = ori_lines
        
        # for debug
        if self.debug:
            import os
            d_polys = [
                np.array(poly["points"]).astype(np.int32).reshape(-1, 2)
                for poly in results["polys"]
            ]
            cv2.imwrite(
                os.path.join("debug", "f_crop_" + os.path.basename(results["filename"])),
                cv2.polylines(results["image"], d_polys, True, (0, 0, 255), 2)
            )

        return results
    
    def __call__(self,
                 results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(min_crop_side_ratio = {self.min_crop_side_ratio})'
        return repr_str
