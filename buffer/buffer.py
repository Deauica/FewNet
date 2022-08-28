""" buffer.py """

stage = 22  # 7

if stage == 1:
    """
    测试 torchvision.ops.roi_pool 的输出结果，保留了维度信息，
    最终结果是 [K, C, output_size[0], output_size[1]]
    """
    from torchvision.ops import nms, roi_align, roi_pool
    import torch
    
    fp = torch.tensor(list(range(5 * 5 * 3 * 2))).float()
    fp = fp.view([2, 3, 5, 5])
    print("fp.shape: {}".format(fp.shape))
    
    boxes = torch.tensor([
        [0, 0, 0, 1, 1],
        [0, 1, 1, 2, 2],
        [1, 2, 2, 3, 3],
        [1, 3, 3, 4, 4,]
    ]).float()
    
    pooled_features = roi_pool(fp, boxes, [4, 4])
    print("pooled_features.shape: {}".format(pooled_features.shape))
    print("pooled_features: {}".format(pooled_features))
    
elif stage == 2:
    """
    验证环境的更替 是否正确，从 1.4.0 的 torch 更换到 1.8.1 的 torch
    """
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())

elif stage == 3:
    from torch import nn
    import torch
    from torchvision.models import resnet50
    import requests
    from PIL import Image
    import torchvision.transforms as T
    
    class DETRdemo(nn.Module):
        """
        Demo DETR implementation.

        Demo implementation of DETR in minimal number of lines, with the
        following differences wrt DETR in the paper:
        * learned positional encoding (instead of sine)
        * positional encoding is passed at input (instead of attention layer)
        * fc bbox predictor (instead of MLP)
        The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
        Only batch size 1 supported.
        """
        
        def __init__(self, num_classes, hidden_dim=256, nheads=8,
                     num_encoder_layers=6, num_decoder_layers=6):
            super().__init__()
            
            # create ResNet-50 backbone
            self.backbone = resnet50()
            del self.backbone.fc
            
            # create conversion layer
            self.conv = nn.Conv2d(2048, hidden_dim, 1)
            
            # create a default PyTorch transformer
            self.transformer = nn.Transformer(
                hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
            
            # prediction heads, one extra class for predicting non-empty slots
            # note that in baseline DETR linear_bbox layer is 3-layer MLP
            self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
            self.linear_bbox = nn.Linear(hidden_dim, 4)
            
            # output positional encodings (object queries)
            self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
            
            # spatial positional encodings
            # note that in baseline DETR we use sine positional encodings
            self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
            self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        
        def forward(self, inputs):
            # propagate inputs through ResNet-50 up to avg-pool layer
            x = self.backbone.conv1(inputs)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            # convert from 2048 to 256 feature planes for the transformer
            h = self.conv(x)
            
            # construct positional encodings
            H, W = h.shape[-2:]
            pos = torch.cat([
                self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ], dim=-1).flatten(0, 1).unsqueeze(1)
            
            # propagate through the transformer
            h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                                 self.query_pos.unsqueeze(1)).transpose(0, 1)
            
            # finally project transformer outputs to class labels and bounding boxes
            return {'pred_logits': self.linear_class(h),
                    'pred_boxes': self.linear_bbox(h).sigmoid()}
        
    # test code for DETRdemo
    detr = DETRdemo(num_classes=91)
    state_dict = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
        map_location='cpu', check_hash=True)
    detr.load_state_dict(state_dict)
    detr.eval()

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # propagate through the model
    outputs = detr(transform(im).unsqueeze(dim=0))
    
    pass

elif stage == 4:
    from shapely.geometry import Polygon, MultiPoint, Point
    poly = [
        [1, 1], [3, 1],
        [3, 3], [1, 3]
    ]
    pts = [
        [2, 2],
        [4, 4]
    ]
    
    polygon = Polygon(poly)
    points = MultiPoint(pts)
    p1, p2 = Point(pts[0]), Point(pts[1])
    dists = polygon.distance(points)
    d1, d2 = p1.distance(polygon), p2.distance(polygon)
    print(dists)
    print(d1, d2)

elif stage == 5:
    import torch
    A_xs = torch.arange(10)
    A_ys = torch.arange(10)
    grid_xs, grid_ys = torch.meshgrid((A_xs, A_ys))
    print(grid_xs.shape)
    grid = torch.stack(
        (grid_xs, grid_ys), dim = 0
    )
    print(grid.shape)
    
    grid = grid.flatten(start_dim=-2).permute((1, 0))
    print(grid.shape)  # [100, 2]
    
    B, nk = 8, 4
    top_indices = torch.randint(high=99, size=[B, nk])
    print(top_indices.shape)  # [8, 4]
    
    top_indices = grid[top_indices]
    print(top_indices.shape)  # [8, 4, 2]
    
    top_indices = top_indices[:, :, 0] * 10 + top_indices[:, :, 1]  # [8, 4]
    top_indices_tile = torch.tile(top_indices.unsqueeze(dim=-1), (3,))
    print(top_indices.shape, "\n", top_indices_tile.shape)

elif stage == 6:
    """
    测试 opencv 的 minarearect 接口
    """
    import cv2
    import numpy as np
    import math

    np.set_printoptions(suppress=True)
    
    # step 1, read quad data
    gt_filename = "datasets/TD_TR/TD500/train_gts/IMG_0064.gt"
    polygons = []
    with open(gt_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            coords = line.strip().split(",")[:-1]
            coords = [ float(coord) for coord in coords]
            polygons.append(coords)
    
    polygons = np.array(polygons, dtype=np.float32).reshape([-1, 4, 2])
    print(polygons, "\n", polygons.shape)
    
    min_area_rects = []
    for polygon in polygons:
        rotated_min_area_rects = cv2.minAreaRect(polygon)
        # ((rotated_center_x, rotated_center_y), (w, h), angle)
        horizontal_min_area_rects = (
            *list(rotated_min_area_rects)[:-1], 0
        )
        box = np.array(cv2.boxPoints(horizontal_min_area_rects))
        m = np.min(box, axis=0)
        x, y = list(m)
        
        min_area_rects.append(
            [
                x, y, *rotated_min_area_rects[1],
                rotated_min_area_rects[-1] * math.pi / 180
            ]
        )
    
    min_area_rects = np.array(min_area_rects)
    print(min_area_rects)
    
elif stage == 7:
    """
    测试 rbox_target 的 mmrot 所提供的接口
    """
    from decoders.utils import obb2poly_np
    from data.utils import poly2obb_np
    import numpy as np
    np.set_printoptions(suppress=True)

    # step 1, read quad data
    gt_filename = "datasets/TD_TR/TD500/train_gts/IMG_1986.gt"  # 1986.gt
    polygons = []
    with open(gt_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            coords = line.strip().split(",")[:-1]
            coords = [float(coord) for coord in coords]
            polygons.append(coords)
    
    polygons = np.array(polygons, dtype=np.float32)
    print(polygons, polygons.shape)
    
    # step 2. generate rbox annotation
    rboxes = []
    for _, polygon in enumerate(polygons):
        rbox = poly2obb_np(polygon, version="le135")
        rboxes.append(rbox)
    
    # step 3. rbox -> points
    rboxes_prob = []
    for rbox in rboxes:
        rbox_prob = ( *rbox, 1)
        rboxes_prob.append(rbox_prob)
    rboxes_prob = np.array(rboxes_prob)
    
    polygons_transformed = obb2poly_np(rboxes_prob, version="le135")
    
    polygons_transformed = np.array(polygons_transformed, dtype=np.float32)
    print(polygons_transformed)

elif stage == 8:
    """ 测试 patch_coords 的生成是否正确 """
    from decoders.seg_detector_conv_trans import PatchTrans
    
    import torch
    
    N, P = 8, 4
    
    patch_xs = torch.tile(
        torch.arange(start=0, end=P).unsqueeze(dim=1),
        dims=(N, P)
    )  # [N, P^2]
    patch_ys = torch.tile(
        torch.arange(start=0, end=P).unsqueeze(dim=0),
        dims=(N, P)
    )  # [N, P^2]
    
    print(patch_xs.shape, patch_ys.shape)
    
elif stage == 9:
    """
    1. 测试 torch.BoolTensor 的乘积效果 -- 无法执行 matmul
    2. Tensor 不同类型之间的转换，这里使用的是 target_tensor = src_tensor.type(Target_Type)
    """
    import torch
    a = torch.tensor([1, 0, 1, 1])
    b = torch.matmul(a.reshape([-1, 1]), a.reshape([1, -1]))
    a = a.type(torch.BoolTensor)
    print(a, b)
    
elif stage == 10:
    """
    测试 positional embedding 的生成, 这里的 pe 选择的是 sin/cos 形式的 positional embedding.
    最后的成果 可以参考 class PositionalEmbedding.
    """
    pass

elif stage == 11:
    import torch
    a = torch.arange(100).reshape([10, 10])
    a_xs = torch.arange(10)
    a_ys = torch.randint(0, 9, size=(8,))
    print(a[(a_xs, a_ys)])  # error
    
elif stage == 12:
    import torch
    a = torch.arange(100).reshape([10, 10])
    print(len(a))

elif stage == 13:
    class A:
        def __init__(self):
            pass
        
    a = A
    b = A()
    import inspect
    
    print(inspect.isclass(a))
    
elif stage == 14:
    """ 测试 cv2.fillPoly(img, pts, color) """
    import cv2
    import numpy as np
    import os
    polygons = [np.array([[682.80408545, 313.03512994],
                          [19.3862761, 181.69786431],
                          [2.24114768, 264.77402744],
                          [665.65895703, 396.11129307]])]
    polygons = [
        polygon.astype(np.int32) for polygon in polygons
    ]
    canvas = np.full([800, 800], 0.4)
    cv2.fillPoly(canvas, polygons, 0.7)
    cv2.imwrite(
        os.path.join("debug", "buffer.jpg"),
        (canvas * 255).astype(np.int32)
    )

elif stage == 15:
    import numpy as np
    a = np.random.random([4, 2])
    print(a)
    
    (x_min, y_min), (x_max, y_max) = np.min(a, axis=0), np.max(a, axis=0)
    print(x_min, y_min, x_max, y_max)

elif stage == 16:
    """ 测试 topk_by_partition """
    from data.make_fewnet_target import MakeFewNetTarget
    import numpy as np
    
    H, W = 8, 10
    nk = 10
    positive_point_lines = np.random.randn(H, W)
    nk_ind, nk_val = MakeFewNetTarget.topk_by_partition(
        input=positive_point_lines.flatten(), k=nk, axis=0, ascending=False
    )
    
    nk_xs, nk_ys = nk_ind // W, nk_ind % W
    for v, x, y in zip(nk_val, nk_xs, nk_ys):
        print(v, x, y, positive_point_lines[x, y])

elif stage == 17:
    """ test for binary mask of numpy array"""
    import numpy as np
    a = np.random.randint(low=0, high=100, size=(10, 10))
    mask = a > 40
    print(a, "\n", mask)
    print(a[mask])

elif stage == 18:
    from PIL import Image
    import cv2
    
    imgpath = "datasets/TD_TR/TD500/train_images/IMG_1986.JPG"
    pil_img = Image.open(imgpath)
    cv_img = cv2.imread(imgpath)
    print(pil_img.size)  # [1600, 1200]
    print(cv_img.shape)  # [1200, 1600, 3], [height, width, channel], [row, col, c]

    import imageio
    import imgaug as ia

    image = imageio.imread(
        "https://upload.wikimedia.org/wikipedia/commons/e/e6/Macropus_rufogriseus_rufogriseus_Bruny.jpg")
    image = ia.imresize_single_image(image, (389, 259))
    print(image.shape)
    
elif stage == 19:
    """ test the mask generation and indexing """
    import torch
    import numpy as np
    
    bs, num_selected_features = 2, 5
    angle_min, angle_max = -np.pi / 4, np.pi * 3 / 4
    logits_threshold = 0.5
    
    outputs = dict(
        logits=torch.rand([bs, num_selected_features, 1]),
        angle=torch.rand([bs, num_selected_features, 1]) * (angle_max - angle_min) + angle_min,
        boxes=torch.rand(bs, num_selected_features, 4)
    )
    
    if len(outputs["logits"].shape) > 2:
        outputs["logits"] = outputs["logits"].squeeze(dim=-1)
    
    logits_mask = outputs["logits"] > logits_threshold
    print(logits_mask.shape, "\n", logits_mask)
    
    out_boxes = outputs["boxes"][logits_mask]
    print(out_boxes.shape)
    
elif stage == 20:
    """ test for collate fn """
    pass

elif stage == 21:
    """
    summarize the data in SynthText downloaded from xunlei
    """
    import zipfile
    import cv2
    import os
    import numpy as np
    from tqdm import tqdm
    
    zip_filepath = r"E:\pcb_tyre_dataset.zip"
    valid_img_list, invalid_img_list = [], []
    
    assert os.path.exists(zip_filepath), (
        "check your zip_filepath: {}".format(zip_filepath)
    )
    
    with zipfile.ZipFile(zip_filepath, mode="r") as f:
        for fpath in tqdm(f.namelist()):
            if not fpath.endswith(".jpg"):
                continue
            
            raw_data = f.read(fpath)
            img = cv2.imdecode(np.frombuffer(raw_data, np.uint8), 1)
            if img is None:
                invalid_img_list.append(fpath)
            else:
                valid_img_list.append((fpath, img.shape))
    
    print(
        "len of valid img list: {} and len of invalid img list: {}".format(
            len(valid_img_list), len(invalid_img_list)
        )
    )

elif stage == 22:
    """
    debug for MakeFewNetTarget due to error raised from MakeFewNetTarget.__call__
    
    This bug may come from RandomCropData.
    """
    config_file = "experiments/fewnet/td500_resnet18.yaml"
    cmd = {
        "batch_size": 1,
        "num_workers": 0,
        "debug": True,
        "name": "make_target_debug"
    }
    
    from concern.config import Config, Configurable
    from train import setup_seed
    import os
    
    
    if not os.path.join(".", "debug"):
        os.mkdir(os.path.join(".", "debug"))
    jpg_paths = [os.path.join("debug", jpg_file)
                 for jpg_file in os.listdir(os.path.join(".", "debug"))
                 if jpg_file.endswith(".jpg") or jpg_file.endswith(".JPG")]
    for jpg_path in jpg_paths:
        os.remove(jpg_path)
    
    setup_seed(seed=20)  # setup seed for reproducibility
    conf = Config()
    experiment_args = conf.compile(conf.load(config_file))['Experiment']
    experiment_args.update(cmd=cmd)
    experiment = Configurable.construct_class_from_config(experiment_args)
    
    train_data_loader = experiment.train.data_loader
    for batch in train_data_loader:
        pass

elif stage == 23:
    """
    verify the correctness of MakeFewNetTarget.distance_point2line
    """
    import numpy as np
    
    xs = np.arange(0, 4, 1)  # width == 4
    ys = np.arange(0, 5, 1)  # height == 5
    print(xs, ys)
    xs, ys = np.meshgrid(xs, ys, indexing="xy")  # xs: col, ys: row
    print(xs, ys)

elif stage == 24:
    """ create a toy dataset to check our model's convergence """
    pass

elif stage == 25:
    """ check the optimizer in pytorch """
    pass

elif stage == 26:
    pass