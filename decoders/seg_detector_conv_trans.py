"""
该文件，定义了 和 patch transformer 相关的一系列的 网络结构 所需要的 Module 组件
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict
from math import ceil


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 max_pos_len=3000, pos_dim=None, keys=("0", "1"),
                 embed_type="sin_cos", **kwargs):
        super(PositionalEmbedding, self).__init__()
        
        self.max_pos_len, self.pos_dim = max_pos_len, pos_dim
        self.keys = keys
        self.embed_type = embed_type
        self.kwargs = kwargs
        self.embed_tables = self.build_embed_table()
        self.static = None
    
    def forward(self, coord_limit):
        """传入 坐标 的矩阵，并且 返回对应的 positional embedding.
        
        Args:
            coord_limit (Sequence): value should be[B, len_0, len_1, ..., len_n]
            
        Returns:
            pe (torch.Tensor): size is [B, len_0, len_1, ..., len_n, self.pos_dim]
            
        Note:
            coord_grid 的长度 - 1 应该和 self.keys 是相同的
        """
        assert len(coord_limit) == len(self.keys) + 1, (
            "self.keys: {} \nHowever,coord_grid: {}\n, "
            "In fact, You only need specified the limit for batch and each dim".format(
                self.keys, coord_limit))
        
        pe = torch.zeros([*coord_limit[1:]])
        coord_per_dim_wo_batch = []
        for limit_dim in coord_limit[1:]:
            coord_per_dim_wo_batch.append(torch.arange(limit_dim))
        coords_per_batch = torch.meshgrid(coord_per_dim_wo_batch)  # [n, len_0, len_1, ..., len_n]
        for i, embed_table in enumerate(self.embed_tables):
            pe += embed_table(coords_per_batch[i])
        return pe.expand([*coord_limit, self.pos_dim])  # 最后返回的是 [B, len_0, len_1, ..., len_n, pos_dim]
    
    def build_embed_table(self):
        embed_table = OrderedDict()
        d_model, position_bias = self.kwargs["d_model"], 0
        if self.embed_type == "sin_cos":
            self.static = True  # 使用静态的 embedding table
            for k in self.keys:
                i_mat = torch.arange(start=0, end=self.pos_dim, step=2) + position_bias
                i_mat = torch.pow(10000., i_mat / d_model).reshape([1, -1])
                pos_mat = torch.arange(start=0, end=self.max_pos_len, step=1).reshape([-1, 1])
    
                table_weight = torch.zeros([self.max_pos_len, self.pos_dim])
                table_weight[:, 0::2] = torch.sin(pos_mat / i_mat)  # 自动 broadcast
                table_weight[:, 1::2] = torch.cos(pos_mat / i_mat)
                self.embed_tables.update({
                    k: nn.Embedding.from_pretrained(table_weight)
                })
                position_bias += self.max_pos_len  # 对 position_bias 加以更新
        else:
            raise ValueError("Please check your embed type: {}".foramt(self.embed_type))
        
        return embed_table


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        # TODO 针对 m.bias
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)


def gen_patch_size(iou_threshold, dist_anchor_points):
    ret = ceil(
        (1 + iou_threshold) * dist_anchor_points / (2 - 2 * iou_threshold)
    )
    return ret


class ConvFPN(nn.Module):
    r"""
    定义的是 conv network 所实现的 FPN，
    相对于 之前 DBNet 的 FPN，在这里添加了一个 BatchNorm2D
    """
    def __init__(self,
                 need_conv_fpn=False,
                 in_channels=(64, 128, 256, 512),
                 inner_channels=256,
                 bias=False,
                 ):
        super(ConvFPN, self).__init__()
        self.need_conv_fpn = need_conv_fpn
        if self.need_conv_fpn:
            self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
            self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
            self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
    
        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.in5.apply(weight_init)
        self.in4.apply(weight_init)
        self.in3.apply(weight_init)
        self.in2.apply(weight_init)
            
        self.out5 = nn.Sequential(
            nn.Conv2d(
                inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'),
            nn.BatchNorm2d(num_features=inner_channels // 4)
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(
                inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.BatchNorm2d(num_features=inner_channels // 4)
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(
                inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(num_features=inner_channels // 4)
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(num_features=inner_channels // 4)
        )
        
        self.out5.apply(weight_init)
        self.out4.apply(weight_init)
        self.out3.apply(weight_init)
        self.out2.apply(weight_init)  # 执行基本的初始化

    def forward(self, features):
        r"""
        显然，输入的 features 是一个 Tuple[feature], 最后的输出，也是一个 Tuple[Feature],
        
        输入：
        c2, c3, c4, c5 分别是 stride 为 4, 8, 16, 32 的 feature map,
        dimension number 分别是 in_channels{[0], [1], [2], [3]}
        
        输出：
        p2, p3, p4, p5 都是 stride 为 4 的 feature map.
        dimension number 分别是 {inner_channels} // 4
        """
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)  # self.in{2, 3, 4, 5} 主要起到了 降低维度的 作用
        
        if self.need_conv_fpn:
            out4 = self.up5(in5) + in4  # 1/16
            out3 = self.up4(out4) + in3  # 1/8
            out2 = self.up3(out3) + in2  # 1/4, self.up{3, 4, 5} 就是 Upsample 2倍 的作用
        else:
            out2, out3, out4, in5 = in2, in3, in4, in5

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        # self.out{2, 3, 4, 5} 包括了一个 conv + Upsample 的作用，
        # 其中， conv 包括了 3x3 的 卷积，以及 降低维度的作用，而 Upsample 则是 横向层面的尺度同步
        return p2, p3, p4, p5
    

class PatchModule(nn.Module):
    r"""
    定义了 生成 Patch 所需要的 组件
    """
    def __init__(self,
                 patch_size=None, dist_anchor_point=None, iou_threshold=None,
                 bias=False, inner_channels=256, num_levels=3,
                 n_k=(512, 256, 128),  # 注意顺序 与 features 保持一致
                 share_tower_depth=1, entropy_tower_depth=1, patch_tower_depth=1
                 ):
        r"""
        PatchModule 在生成 patch 的过程中，有如下的几个 准则：
        1. 为了方便起见，这里采用了 “分离” 的原则，也就是说：
          1.1 anchor point 的选择；
          1.2 candidate patch 的明确是分开的.
          
        2. Loss 的计算，包括了两个：
          2.1 anchor point 位置的 Loss 计算，采用了 smooth l1 loss 完成计算；
          2.2 patch size 的 Loss 计算，以 较小的 size， 则最终的 Loss 也较小。
          最终的目的是，鼓励 high entropy 的 anchor point 以及 small patch size 两个原则。
        """
        
        super(PatchModule, self).__init__()
        assert num_levels == len(n_k), (
            "len of N_k should be same as num_levels, but during running, "
            "N_k: {}, num_levels: {}".format( n_k, num_levels)
        )
        
        self.patch_size, self.dist_anchor_point, self.iou_threshold = (
            patch_size, dist_anchor_point, iou_threshold
        )
        self.bias = bias
        self.inner_channels = inner_channels
        self.num_levels = num_levels  # hierarchical feature map 的层次数
        self.N_k = n_k
        self.share_tower_depth, self.entropy_tower_depth, self.patch_tower_depth = (
            share_tower_depth, entropy_tower_depth, patch_tower_depth
        )
        
        if self.patch_size is None:
            assert self.iou_threshold and self.dist_anchor_point, (
                "When patch_size is None, iou_threshold and dist_anchor_point"
                "can not be None"
            )
            self.patch_size = gen_patch_size(self.iou_threshold, self.dist_anchor_point)
        
        # share_head, entropy_head, patch_head generalization
        tower_config = dict(
            share_tower=dict(
                num_convs=self.share_tower_depth,
                conv_func=nn.Conv2d,
                norm_type="BN"
            ),
            entropy_tower=dict(
                num_convs=self.entropy_tower_depth,
                conv_func=nn.Conv2d,
                norm_type="BN"
            ),
            patch_tower=dict(
                num_convs=self.patch_tower_depth,
                conv_func=nn.Conv2d,
                norm_type="BN"
            )
        )
        for module_name, config in tower_config:
            conv_func = config["conv_funcs"]
            num_convs = config["num_convs"]
            norm_type = config["norm_type"]
            tower = []
            
            for _ in range(num_convs):
                tower.append(
                    conv_func(inner_channels, inner_channels,
                              kernel_size=3, stride=1, padding=1, bias=True)
                )
                if norm_type == "BN":
                    tower.append(
                        ModuleListDial(
                            [nn.BatchNorm2d(inner_channels) for _ in range(self.num_levels)])
                    )  # ModuleListDial 其实 隐含的规定了 在提取特征的过程中，针对固定的 feature level，提取次序当固定下来
                tower.append(nn.ReLU())
            
            self.add_module(f"{module_name}", nn.Sequential(
                *tower
            ))
        
        self.entropy_head = nn.Conv2d(
            inner_channels, 1, kernel_size=1, stride=1  # MLP，预测该点的 entropy value
        )
        self.patch_head = nn.Linear(
            in_features=inner_channels, out_features=4, bias=True
        )
        # Linear Layer，之所以采用 Linear Layer 而不是上面 kernel 为 1 的 conv layer， 是因为
        # 此时已经经过了 anchor point 的筛选，且 后面的 loss 的计算，也是在 筛选后的 anchor point
        # 基础上来做的
        
            
    def forward(self, features, data):
        r"""
        最后返回的是 patch, patch_loss 两个部分
        
        Args:
            features (Suquence[bs, inner_channels // 4, h/4, w/4]): 经过 FPN 特征提取后的 Feature,
              每一个 索引 对应一个 feature level;
            data (Sequence[bs]): 每一个元素 代表一个 sample,
              data["target_coords"] 是这里的核心, 存储了 anchor point 的坐标
            
        Returns:
            1. predictions: List[Dict] 类型. 每一个 索引代表一个 feature level, 其中，
               1. predictions["entropy_map"]: 用以索引 预测的 entropy, .shape == [bs, 1, h, w]
               2. predictions["patch_coord"]: 用以索引 预测的 patch coordination, .shape == [bs, nk_i, 4],
                  [0], [1], [2], [3] 分别表示 top-left's {x, y}, bottom-right's {x, y}
            之所以 设置 predictions[i] 表示 i-th feature level 而不是 i-th data sample 的原因，就是 Patch
            本身仍然只是 生成 中间步骤，用以后续的 特征提取
        """
        predictions = []
        if not torch.is_tensor(data["target_coords"]):
            data["target_coords"] = torch.as_tensor(data["target_coords"])  # [2, anchor_H, anchor_W]
        if len(data["target_coords"].shape) == 4:
            data["target_coords"] = data["target_coords"][0]  # 目前认为 anchor point 的配置，应该是固定的
            
        for i, x in enumerate(features):
            """ 针对每一个 feature level 单独计算 """
            x = self.share_tower(x)
            entropy_x = self.entropy_tower(x)  # [B, inner_channels, H, W]
            entropy_map = F.sigmoid(self.entropy_head(entropy_x))  # sigmoid, [B, 1, H, W]
            B, C, H, W = entropy_x.shape
            nk = self.N_k[i]  # top_nk
            
            # anchor point selection
            anchor_points_entropy = entropy_map[
                                    :, 0, data["target_coords"][0], data["target_coords"][1]
                                    ]  # [B, anchorH, anchorW]
            anchor_points_entropy = anchor_points_entropy.flatten(
                start_dim=-2, end_dim=-1
            )  # [B, anchorH * anchor_W]
            _, top_nk_indices = torch.topk(anchor_points_entropy, dim=-1, k=nk)  # [B, nk]
            top_nk_indices_xy = (
                data["target_coords"].flatten(start_dim=-2).permute((1, 0))[top_nk_indices]
            )  # [B, nk, 2]
            top_nk_indices = top_nk_indices_xy[:, :, 0] * W + top_nk_indices_xy[:, :, 1]  # [B, nk]
            
            # extract feature from `patch_x` based on `top_nk_indices`
            patch_x = self.patch_tower(x)  # [B, C, H, W]
            patch_x = patch_x.flatten(start_dim=-2).permute((0, -1, -2))  # [B, H*W, C]
            anchor_points_x = torch.gather(
                input=patch_x, dim=1, index=torch.tile(top_nk_indices.unsqeeze(dim=-1), (C,))
            )  # [B, nk, C]
            anchor_patch_coords = F.sigmoid(self.patch_head(anchor_points_x))  # [B, nk, 4]
            
            prediction = dict(
                entropy_map=entropy_map,  # [B, 1, H, W]
                patch_coords=anchor_patch_coords,  # [B, nk, 4]
                patch_coords_norm=anchor_patch_coords,  # [B, nk, 4], 用于计算 patch_loss_dict
                anchor_coords=top_nk_indices_xy  # [B, nk, 2]，这里是 绝对值
            )
            predictions.append(prediction)
        
        # 对 patch_coords 进行修正，将之 转换为 绝对值
        # 当然，这里的 绝对值，仍然是在 stride 为 4 的范围，下面的代码 会不会更新 pred 里面的内容
        # WARN 这个可以在 debug 的时候对照一下
        for pred in predictions:
            patch_coords = pred["patch_coords"]  # [B, nk, 4]
            anchor_coords = pred["anchor_coords"]  # [B, nk, 2]
            patch_coords[:, :, 0] = anchor_coords[:, :, 0] - self.patch_size * patch_coords[:, :, 0]
            patch_coords[:, :, 2] = anchor_coords[:, :, 2] + self.patch_size * patch_coords[:, :, 2]
            
            patch_coords[:, :, 1] = anchor_coords[:, :, 1] - self.patch_size * patch_coords[:, :, 1]
            patch_coords[:, :, 3] = anchor_coords[:, :, 3] + self.patch_size * patch_coords[:, :, 3]
        
        return predictions
        

class PatchTrans(nn.Module):
    r"""
    Patch Transformer 的入口类，定义了 相关的 网络架构 和 数据流通的途径
    """
    def __init__(self,
                 need_conv_fpn=True, inner_channels=256,
                 in_channels=(64, 128, 256, 512),
                 bias=True,
                 patch_size=None, dist_anchor_point=None, iou_threshold=None,
                 num_levels=3, n_k=(512, 256, 128),
                 share_tower_depth=2, entropy_tower_depth=1, patch_tower_depth=1,
                 
                 d_model=None, nhead=8,
                 pixel_pixel_depth=2, region_region_depth=2, scale_scale_depth=0,
                 result_type="bbox",
                 # 可以添加一个参数 表明 选择 positional embedding 的类型，默认是 不可训练的 pe
                 max_pos_len=3000,  # max for height and width will be 3000
                 pe_type="sin_cos", pos_dim=None
                 ):
        super(PatchTrans, self).__init__()
        assert scale_scale_depth == 0, (
            "current **only** region-region relationship will be utilized"
        )
        
        self.need_conv_fpn = need_conv_fpn
        self.inner_channels, self.in_channels = inner_channels, in_channels
        self.bias = bias
        self.conv_fpn = None if not self.need_conv_fpn else ConvFPN(
            need_conv_fpn=self.need_conv_fpn, in_channels = in_channels,
            inner_channels = self.inner_channels, bias=self.bias
        )
        
        # patch module initialization
        self.patch_size, self.dist_anchor_point, self.iou_threshold = (
            patch_size, dist_anchor_point, iou_threshold
        )
        self.num_levels, self.n_k = num_levels, n_k
        self.share_tower_depth, self.entropy_tower_depth, self.patch_tower_depth = (
            share_tower_depth, entropy_tower_depth, patch_tower_depth
        )
        self.patch_module = PatchModule(
            self.patch_size, self.dist_anchor_point, self.iou_threshold,
            self.bias, self.inner_channels, self.num_levels, self.n_k,
            self.share_tower_depth, self.entropy_tower_depth, self.patch_tower_depth
        )
        
        # multiple encoder -- pixel-pixel, region-region, scale-scale
        self.d_model = d_model if d_model else self.inner_channels
        self.nhead = nhead
        self.pixel_pixel_depth, self.region_region_depth, self.scale_scale_depth = (
            pixel_pixel_depth, region_region_depth, scale_scale_depth
        )
        self.pp_encoderlayer = nn.TransformerEncoderLayer(self.d_model, self.nhead)
        self.rr_encoderlayer = nn.TransformerEncoderLayer(self.d_model, self.nhead)
        self.pp_encoder = nn.TransformerEncoder(
            encoder_layer=self.pp_encoderlayer, num_layers=self.pixel_pixel_depth,
            norm=nn.LayerNorm(self.d_model)
        )
        self.rr_encoder = nn.TransformerEncoder(
            encoder_layer=self.rr_encoderlayer, num_layers=self.region_region_depth,
            norm=nn.LayerNorm(self.d_model)
        )
        if self.scale_scale_depth:
            self.ss_encoderlayer = nn.TransformerEncoderLayer(self.d_model, self.nhead)
            self.ss_encoder = nn.TransformerEncoder(
                encoder_layer=self.ss_encoderlayer, num_layers=self.scale_scale_depth,
                norm=nn.LayerNorm(self.d_model)
            )
        else:
            self.ss_encoderlayer, self.ss_encoder = None, None
            
        # detection head -- cls_head, box_head
        self.result_type = result_type.lower()
        if self.result_type == "bbox":
            self.box_head = nn.Linear(
                in_features=self.d_model, out_features=5 + 1  # (x, y, w, h, theta, score)
            )  # theta 是否需要归一化？
        elif self.result_type == "bezier":
            self.box_head = nn.Linear(in_features=self.d_model, out_features=16 + 1)
        else:
            raise ValueError(
                "self.result_type should be 'bbox' or 'bezier'. "
                "However, your result_type is: {}".format(
                    self.result_type
                )
            )
        
        # positional embedding table generation
        self.max_pos_len, self.pe_type = max_pos_len, pe_type
        self.pos_dim = pos_dim if pos_dim else self.d_model
        self.pe_table = PositionalEmbedding(
            max_pos_len=self.max_pos_len, pos_dim=self.pos_dim,
            keys=(0, 1), embed_type=self.pe_type
        )

        
    def forward(self, features, data):
        r"""
        模型的 具体前向过程，这里 传入的是 backbone 输出的 features，
        此外，与之前 DBNet 的实现保持一致，这里不进行 loss 的计算，只是执行 forward 的前向过程
        """
        # step 1. conv fpn
        p2, p3, p4, p5 = self.conv_fpn(features)  # p{2, 3, 4, 5} .shape == (B, inner_channels, H/4, W/4)
        features = (p3, p4, p5)  # 这里去掉了一个
        
        # step 2, obtain patch and patch_loss
        assert len(features) == self.num_levels
        
        patch_preds = self.patch_module(features, data)
        patch_coords = []   # num_levels -- [B, nk_i, 4], x_tl, y_tl, x_br, y_br
        anchor_coords = []  # num_levels -- [B, nk_i, 2], anchor_x, anchor_y
        for p_pred in patch_preds:
            patch_coords.append(p_pred["patch_coords"])
            anchor_coords.append(p_pred["anchor_coords"])
        
        # step 3: extract patch features and patch_masks based on patch_coords
        # 这个 步骤，必须保证 patch_coords[i] 与 features[i] 的对齐
        patch_full_features = []  # List[FloatTensor], (num_levels, [B, nk, P*P, C]),
        patch_valid_pos = []  # List[BoolTensor], (num_levels, [B, nk, P*P])
        
        for _, anchor_coord, real_patch_limit, feature, nk in enumerate(zip(
                anchor_coords, patch_coords, features, self.n_k)):
            """
            本次训练，仅仅用于 patch 特征的提取 和 mask 的生成，实际的 TransformerEncoder
            在这里 并不会起到作用.
            
            anchor_coord.shape == [B, nk, 2], real_patch_limit.shape == [B, nk, 4],
            feature.shape == [B, C, H, W], nk is scalar
            
            这里的 limit 指的是 [top-left_x, top_left_y, bottom_right_x, bottom_right_y]
            目前的情况下，应该保证 不同level下，feature.shape 仍然是固定的 [H, W].
            """
            full_patch_limit = torch.empty_like(real_patch_limit)  # [B, nk, 4]
            full_patch_limit[:, :, :2] = anchor_coord[:, :, 0:2] - self.patch_size  # x_tl, y_tl
            full_patch_limit[:, :, 2:] = anchor_coord[:, :, 0:2] + self.patch_size  # x_br, y_br
            
            # step 1: extract full_patch_features based on full_patch_coord and feature
            # full_patch_features.shape == [B, nk, p, p, C]
            full_patch_coords_x, full_patch_coords_y = self.gen_patch_coords_from_limit(
                patch_limit=full_patch_limit, patch_size=self.patch_size,
            )  # Tuple[torch.Tensor[B, nk, P, P], torch.Tensor[B, nk, P, P]]
            _, canvas_W = feature.shape[-2:]
            full_patch_coords = (
                    canvas_W * full_patch_coords_x + full_patch_coords_y
            ).flatten(start=-2, end=-1)  # [B, nk, P^2]
            full_patch_features = torch.gather(  # 根据 patch 进行特征提取，这一部分 挺麻烦的
                input=feature.permute([0, 2, 3, 1]).flatten(start=1, end=2),  # [B, H*W, C]
                dim=1,
                index=torch.tile(  # [B, nk * P^2, C]
                    full_patch_coords.flatten(start=1, end=2).unsqueeze(-1),  # [B, nk * P^2, 1]
                    dims=(self.inner_channels, )  # self.inner_channels == C
                )
            ).reshape([*full_patch_coords.shape, -1])  # [B, nk, P^2, C]
            patch_full_features.append(full_patch_features)  #
            
            # step 2: obtain mask for patch, full_patch_mask: [B, nk, P^2, P^2]
            real_patch_border = []
            for k in range(4):
                real_patch_border.append(
                    torch.tile(  # [B, nk, P, P]
                        real_patch_limit[..., k].unsqueeze(-1), dims=(self.patch_size, self.patch_size, )
                    ).unflatten(dim=-1, sizes=(self.patch_size, self.patch_size))
                )
                
            full_patch_valid_pos_x = real_patch_border[0] <= full_patch_coords_x <= real_patch_border[2]
            full_patch_valid_pos_y = real_patch_border[1] <= full_patch_coords_y <= real_patch_border[3]
            full_patch_valid_pos = (full_patch_valid_pos_x & full_patch_valid_pos_y).flatten(start=-2)
            patch_valid_pos.append(full_patch_valid_pos)
        
        # step 4: add positional embedding and CLS token to ``patch_full_features``
        # and then pass the feature through self.pp_encoder
        region_all_features = []
        for _, patch_feature, valid_pos in enumerate(zip(patch_full_features, patch_valid_pos)):
            B, nk, P_2, C = patch_feature.shape  # P_2 is P^2
            patch_feature = (  # [B*nk, P^2, C]
                    patch_feature.flatten(start=0, end=1) +
                    self.pe_table([B*nk, math.sqrt(P_2), math.sqrt(P_2)]).flatten(start=-3, end=-2)
            )
            patch_feature = patch_feature.permuate(2, 0, 1)  # [P^2, B*nk, C]
            valid_pos = valid_pos.flatten(start=0, end=1)  # [B*nk, P^2] target: [B*nk*nhead, 1+P^2, 1+P^2+1]
            cls_pos = torch.ones([B * nk, 1])
            cls_token = torch.randn([B*nk, C]).unsqueeze(dim=0)  # [1, B*nk, C]
            
            patch_feature = torch.cat((cls_token, patch_feature), dim=0)  # [1 + P^2, B * nk, C]
            valid_pos = torch.cat((cls_pos, valid_pos), dim=1)  # [B*nk, 1 + P^2]
            valid_mask = torch.matmul(  # [B*nk, 1+P^2, 1+P^2]
                valid_pos.unsqueeze(dim=-1), valid_pos.unsqueeze(dim=1)
            )
            patch_feature = self.pp_encoder(patch_feature, valid_mask)  # [1 + P^2, B * nk, C]
            region_all_features.append(patch_feature[0].reshape([B, nk, C]))  # [B, nk, C]
            
        region_all_features = torch.cat(region_all_features, dim=1)  # [B, \sum_i{nk_i}, C]
        
        # step 5. pass ``region_all_features`` through self.rr_encoder
        region_all_features = region_all_features.permute((1, 0, 2))  # [\sum_i{nk_i}, B, C]
        region_all_features = self.rr_encoder(region_all_features)  # similar to feature grouping network
        
        # step 6: prediction and obtain ultimate loss
        region_all_features = region_all_features.permute([1, 0, 2])  # [B, \sum_i{nk_i}, C]
        pred = self.box_head(region_all_features)  # [B, \sum_i{nk_i}, 5]，在这里 没有 sigmoid
        pred = dict(
            boxes=F.sigmoid(pred[..., 0:4]),  # 0:2, center, 2:3 width, 3:4 height
            angle=pred[..., 4:5],  # no sigmoid
            logits=pred[..., -1:],  # no sigmoid in train
            patch_coords=patch_preds["patch_coords"]
        )
        return pred
        
    @staticmethod
    def gen_patch_coords_from_limit(patch_limit: torch.Tensor, patch_size: int):
        """ generate `patch_coords` from `patch_limit`
        
        Args:
            patch_limit (torch.Tensor): [..., 4], [0, 1, 2, 3] is for [x_tl, y_tl, x_br, y_br]
            patch_size (int): 表明当前坐标下的 patch 的 size 情况
            
        Returns:
            Tuple[patch_xs, patch_ys] (Tuple[torch.Tensor, torch.Tensor]):
              patch_xs: [..., patch_size, patch_size]: x 坐标，
              patch_ys: [..., patch_size, patch_size]: y 坐标
              
            这里的 x, y 坐标，值得分别是 dim=0 和 dim=1 的索引坐标.
              
        Note:
            虽然这里的 patch_coords 的最后一个维度是 patch_size * patch_size，但是里面真实的坐标
            上下界 受限制于 patch_limit[:, 2] 和 patch_limit[:, 3] 中所限制的上下界.
        """
        patch_limit_shape = patch_limit.shape
        assert patch_limit_shape[-1] == 4, (
            "patch_limit's shape should be: (..., 4), "
            "However, your patch_limit.shape: {}".format(patch_limit_shape)
        )
        
        patch_limit = patch_limit.reshape([-1, 4])  # [N, 4]
        N, P = patch_limit.shape[0], patch_size

        patch_xs = torch.tile(
            torch.arange(start=0, end=P).unsqueeze(dim=1),
            dims=(N, P)
        )  # [N, P^2]
        patch_ys = torch.tile(
            torch.arange(start=0, end=P).unsqueeze(dim=0),
            dims=(N, P)
        )  # [N, P^2]
        
        # offset 的增加, 此时 patch_xs, patch_ys 是 以 x_tl, y_tl 为偏置下的坐标
        offset_xs = torch.tile(patch_limit[:, 0:1], dims=(P * P, ))
        offset_ys = torch.tile(patch_limit[:, 1:2], dims=(P * P, ))
        patch_xs = patch_xs + offset_xs
        patch_ys = patch_ys + offset_ys  # [N, P^2]
        # 使用 x_br, y_br 作为限制，保证 所有的坐标 都是在 patch_limit 下,
        # 需要注意的是，这里 不可以使用 x_br, y_br 整合后的 一维下标 作为限制，
        # 而是应该直接使用 x_br, y_br 作为限制
        limit_x_br = torch.tile(patch_limit[:, 2:3], dims=(P * P,))
        limit_y_br = torch.tile(patch_limit[:, 3:4], dims=(P * P,))
        patch_xs = torch.clamp_max(patch_xs, limit_x_br)
        patch_ys = torch.clamp_max(patch_ys, limit_y_br)  # [N, P^2]
        
        return (
            patch_xs.reshape([*patch_limit_shape[:-1], patch_size, patch_size]),   # [..., P, P]
            patch_ys.reshape([*patch_limit_shape[:-1], patch_size, patch_size]),   # [..., P, P]
        )
