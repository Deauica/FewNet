""" buffer.py """

stage = 29  # 7

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
    
    change from RandomCropData to RandomCropInstance.
    """
    config_file = "experiments/fewnet/toy_dataset_toy_resnet18.yaml"
    cmd = {
        "batch_size": 2,
        "num_workers": 0,
        "debug": True,
        "name": "random_crop_instance_test"
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
    """
    test the angle version for decoders.gwd_loss.xy_wh_r_2_xy_sigma.
    
    当前的临时结果是,xy_wh_r_2_xy_sigma 和 angle version 没有关系.
    """
    import sys
    sys.path.append(r"E:\Idea\ConvTransformer\code\conv_trans")
    
    from decoders.gwd_loss import xy_wh_r_2_xy_sigma
    from data.utils import poly2obb_np
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    import torch
    import cv2
    
    def gen_gaussian_distribution(mu, cov, M):
        data = np.random.multivariate_normal(mu, cov, M)
        gaussian = multivariate_normal(mean=mu, cov=cov)
        
        return data, gaussian
    
    angle_version = "le90"
    canvas_H, canvas_W = 800, 800
    polygon = [
        588, 829, 1545, 32, 1600, 98, 643, 895
    ]
    rbox = torch.tensor(
        poly2obb_np(polygon, angle_version)  # [cx, cy, w, h, angle]
    ).reshape([-1, 5])
    # rbox[:, 0:-1:2] = rbox[:, 0:-1:2] / canvas_W  # increase robustness of indexing
    # rbox[:, 1::2] = rbox[:, 1::2] / canvas_H
    
    xy, sigma = xy_wh_r_2_xy_sigma(rbox)  # xy is mu and sigma is cov
    if sigma.shape[0] == 1:
        sigma = sigma.reshape([2, 2])
        xy = xy.int().reshape([2])
    
    # plot
    canvas = np.zeros([canvas_H, canvas_W], dtype=np.float64)
    
    poly_np = np.array(polygon).reshape([-1, 2])  # 4, 2
    poly_width = poly_np[:, 0].max() - poly_np[:, 0].min()
    poly_height = poly_np[:, 1].max() - poly_np[:, 1].min()
    M = max(poly_width, poly_height) * 10
    
    _, gau_distribution = gen_gaussian_distribution(
        xy, sigma, M
    )
    X, Y = np.meshgrid(
        np.linspace(xy[0]-M//2, xy[0]+M//2, M), np.linspace(xy[1]-M//2, xy[1]+M//2, M)
    )
    d = np.dstack([X, Y])
    Z = gau_distribution.pdf(d).reshape([M, M])
    plt.imshow(Z)
    plt.title("{}: {}".format(angle_version, rbox.flatten()[-1]))
    plt.show()
    
    cv2.polylines(
        canvas, [np.array(polygon).reshape(-1, 2).astype(np.int32)],
        True, 1, 2
    )
    cv2.imwrite(
        "debug/a_test_angle_version.jpg",
        cv2.applyColorMap((canvas * 255).astype(np.uint8), cv2.COLORMAP_JET)
    )

elif stage == 27:
    """ test for the scipy.stats.multivariate """
    from scipy.stats import multivariate_normal
    import torch
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    mu = (0, 0)
    cov = np.eye(2)
    gaussian = multivariate_normal(mean=mu, cov=cov)
    
    M = 200
    xs, ys = np.meshgrid(
        np.linspace(-10, 10, M), np.linspace(-10, 10, M)
    )
    d = np.dstack([xs, ys])  # [M, M, 2]
    canvas = gaussian.pdf(d).reshape([M, M])
    plt.imshow(canvas)
    plt.show()
    
    cv2.imwrite(
        "debug/a_test_gaussian.jpg",
        cv2.applyColorMap((canvas * 255).astype(np.uint8), cv2.COLORMAP_JET)
    )

elif stage == 28:
    """ another version of gaussian heatmap """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    import cv2
    
    # create 2 kernels
    m1 = (-1, -1)
    s1 = np.eye(2)
    k1 = multivariate_normal(mean=m1, cov=s1)

    m2 = (1, 1)
    s2 = np.eye(2)
    k2 = multivariate_normal(mean=m2, cov=s2)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    xlim = (-3, 3)
    ylim = (-3, 3)
    xres = 100
    yres = 100

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x, y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = k1.pdf(xxyy) + k2.pdf(xxyy)

    # reshape and plot image
    img = zz.reshape((xres, yres))
    plt.imshow(img)
    plt.show()
    
    cv2.imwrite(
        "debug/a_test_gaussian_another.jpg",
        cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET)
    )

elif stage == 29:
    """
    visualize the results, Green is gt while Red is prediction.
    """
    import os
    import cv2
    import numpy as np
    
    result_dir = "results/"
    img_dir = "datasets/toy_dataset/test_images"
    gt_dir = "datasets/toy_dataset/test_gts"
    
    img_res_visdir = os.path.join(
        os.path.dirname(img_dir),
        os.path.basename(img_dir) + "_vis"
    )
    if os.path.exists(img_res_visdir):
        import shutil
        shutil.rmtree(img_res_visdir)
    
    def extract_polygons(anno_path, contain_prob=True):
        scores = []
        polygons = []
        
        with open(anno_path, "r") as f:
            lines = f.readlines()
            lines = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in lines]
            for line in lines:
                line = [
                    float(item) for item in line.split(",")[:-1]
                ]
                polygons.append(
                    np.array(line[:None]).reshape([-1, 2])
                        .astype(np.int32)
                )
                scores.append(
                    float(line[-1]) if contain_prob else 1
                )
        return polygons, scores
    
    for anno_file in os.listdir(result_dir):
        anno_path = os.path.join(result_dir, anno_file)
        
        polygons, scores = extract_polygons(anno_path)
        
        img_file = anno_file.split(".")[0][4:] + ".jpg"
        img_path = os.path.join(img_dir, img_file)
        gt_path = os.path.join(gt_dir, img_file + ".txt")
        if not os.path.exists(img_path):
            img_file = img_file.replace(".jpg", ".JPG")
            img_path = os.path.join(img_dir, img_file)
            gt_path = os.path.join(gt_dir, img_file + ".txt")
        
        assert os.path.exists(img_path) and os.path.exists(gt_path), (
            "img_path  or gt_path do not exist: {}, {}".format(img_path, gt_path)
        )
        
        img = cv2.imread(img_path)
        gt_polygons, _ = extract_polygons(gt_path)
        
        cv2.polylines(img, polygons, True, (0, 0, 255), 2)
        cv2.polylines(img, gt_polygons, True, (0, 255, 0), 4)
        if not os.path.exists(img_res_visdir):
            os.mkdir(img_res_visdir)
            
        cv2.imwrite(
            os.path.join(img_res_visdir, img_file),
            img
        )

elif stage == 30:
    """
    Only utilize during debugging of pycharm.
    targets is generated by pycharm.
    """
    targets = None

    import numpy as np
    import torch
    from typing import List
    import cv2
    import os
    
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    
    def norm_tensor_2_ndarray(t) -> List[np.ndarray]:
        t = t * 255
        
        results = []
        for img in torch.unbind(t, dim=0):
            img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))  # [H, W, C]
            img += RGB_MEAN.reshape([1, 1, -1])
            img = img.clip(0, 255)
            results.append(img.astype(np.int32))
        return results
    
    for jpg_name in os.listdir("debug"):
        if jpg_name.startswith("results") or jpg_name.startswith("t_sm") \
                or jpg_name.startswith("a_poly"):
            os.remove(os.path.join("debug", jpg_name))
    
    if targets is not None:
        # step 1: check for image
        results = norm_tensor_2_ndarray(targets["image"])
        for i, img in enumerate(results):
            cv2.imwrite(
                "debug/results_{}.jpg".format(i),
                img
            )

        # step 2: check for score_amp
        strides = (8, 16, 32)
        if "score_map" in targets:
            for i, score_maps in enumerate(targets["score_map"]):
                stride = strides[i]
                score_maps = score_maps.cpu().numpy()
                for imgnum, score_map in enumerate(score_maps):
                    cv2.imwrite(
                        "debug/t_sm_{}_{}.jpg".format(stride, imgnum),
                        cv2.applyColorMap(
                            (score_map * 255).astype(np.uint8), cv2.COLORMAP_JET
                        )
                    )

        # step 3: check for rbox based on targets["angle"] and targets["boxes"]
        t_rboxes = [
            torch.cat([t_boxes, t_angle], dim=-1)
            for t_boxes, t_angle in zip(targets["boxes"], targets["angle"])
        ]
        
        from decoders.utils import obb2poly
        t_polys = [
            obb2poly(t_rbox, "le135").cpu().numpy()
            for t_rbox in t_rboxes
        ]
        for i, (t_poly, img) in enumerate(zip(t_polys, results)):
            t_poly = t_poly.reshape([-1, 4, 2])
            cv2.polylines(img, t_poly, True, (0, 255, 0), 3)
            cv2.imwrite(
                "debug/a_polyed_{}.jpg".format(i),
                img
            )
        pass

elif stage == 31:
    try:
        from decoders.utils import obb2poly
        from data.utils import poly2obb_np
    except Exception as e:
        import sys
        sys.path.append("..")
        from decoders.utils import obb2poly
        from data.utils import poly2obb_np
    
    from shapely.geometry import Polygon
    import numpy as np
    
    angle_version = "le135"
    box_quad = [712, -24, 712, -24, 697, 24, 697, 24, 0]
    box_rotated = poly2obb_np(
        np.array(box_quad), angle_version
    )
    print(box_rotated)
    
elif stage == 32:
    """ generate new weight_path with pretrained FPN """
    pass

elif stage == 33:
    """ simple code snippet to visualize score_map during inference """
    from collections import OrderedDict
    import torch
    
    def draw_score_maps(pred, batch, *args, **kwargs):
        import cv2
        import numpy as np
        import os
        
        score_maps = pred["score_map"]
        strides = (8, 16, 32)
        filenames = (
            batch["filename"] if "filename" in batch else
            [f"pseudo_{i}.jpg" for i in range(score_maps[0].shape[0])]
        )
        
        for i, f_level_score_maps in enumerate(score_maps):
            stride = strides[i]
            for j, (score_map, filename) in enumerate(zip(f_level_score_maps, filenames)):
                pseudo_img = cv2.applyColorMap(
                    (score_map.cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                cv2.imwrite(
                    os.path.join("debug", f"b_{filename}_score_map_{stride}.jpg"), pseudo_img
                )
        
    
    config_file = "experiments/fewnet/toy_dataset_resnet18.yaml"
    batch_size, num_workers = 1, 0
    weight_path = "workspace/toy_dataset_resnet18_1/model/final"
    result_dir = "results/"
    name = "eval_buf_post_{}".format(str(draw_score_maps.__name__))
    
    cmd = OrderedDict(
        exp=config_file,
        batch_size=batch_size, num_workers=num_workers, resume=weight_path,
        name=name,
        result_dir=result_dir,
        visualize=False
    )
    post_process = draw_score_maps
    
    try:
        from concern.config import Config, Configurable
        from eval import Eval
    except Exception as e:
        import sys
        import os
        if os.path.exists(os.path.join(".", "concern")):
            sys.path.append(".")
        else:
            sys.path.append("..")
        from concern.config import Config, Configurable
        from eval import Eval
        
    # compile config file
    conf = Config()
    experiment_args = conf.compile(conf.load(cmd['exp']))['Experiment']
    experiment_args.update(cmd=cmd)
    experiment = Configurable.construct_class_from_config(experiment_args)
    
    # initialize model
    e = Eval(experiment, experiment_args, cmd=cmd)
    e.init_torch_tensor()
    model = e.init_model()
    e.resume(model, e.model_path)
    model.eval()  # set the eval mode
    
    # obtain pred
    with torch.no_grad():
        for _, data_loader in e.data_loaders.items():
            for _, batch in enumerate(data_loader):
                pred = model.forward(batch, training=False)
                post_process(pred, batch)

elif stage == 34:
    """
    regenerate toy_dataset due to the extreme difficulty of the previous toy dataset,
    currently, toy_dataset is subset of msra td500
    """
    train_list_filepath = "datasets/toy_dataset/train_list.txt"
    src_img_root, src_anno_root = (
        "datasets/TD_TR/TD500/train_images", "datasets/TD_TR/TD500/train_gts"
    )
    dst_img_root, dst_anno_root = (
        "datasets/toy_dataset/train_images", "datasets/toy_dataset/train_gts"
    )
    
    import os
    import shutil
    if os.path.exists(dst_img_root):
        shutil.rmtree(dst_img_root)
        os.mkdir(dst_img_root)
    if os.path.exists(dst_anno_root):
        shutil.rmtree(dst_anno_root)
        os.mkdir(dst_anno_root)
    
        
    with open(train_list_filepath, "r") as f:
        lines = f.readlines()
        train_list_imgpaths = [line.strip() for line in lines]
    
    for imgpath in train_list_imgpaths:
        src_imgpath = os.path.join(src_img_root, imgpath)
        src_anno_path = os.path.join(src_anno_root, imgpath + ".txt")
        
        dst_imgpath = os.path.join(dst_img_root, imgpath)
        dst_anno_path = os.path.join(dst_anno_root, imgpath + ".txt")
        
        assert os.path.exists(src_imgpath) and os.path.exists(src_anno_path), (
            "{} or {} may no exist".format(src_imgpath, src_anno_path)
        )
        shutil.copy(src_imgpath, dst_imgpath)
        shutil.copy(src_anno_path, dst_anno_path)
