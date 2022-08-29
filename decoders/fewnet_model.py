"""
Model definition for "Few Could be Better than All".

需要注意的是，当前网络定义模块，不涉及 loss 的计算，FewNet 整体 loss 的计算，定义在了
decoders/fewnet_loss.py.
此外，在真实的执行过程中，这里仅仅只是 decoder 的部分，整体的网络架构参考：
structure/model.py 中的 ``SegDetectorModel``.

"""

import torch
from torch import nn
from collections import OrderedDict

from .fpn import VisionFPN
from .positional_embedding import PositionalEmbedding

FPN = VisionFPN  # feature pyramid network


def _reset_parameters(module):
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class CoordConv(nn.Module):  # CoordConv definition
    pass


class FeatureSampling(nn.Module):
    """
    1. coord_conv,
    2. (constrained) deformable conv, pooling,
    3. topk
    ->
    List[N_k_i, 256],
    smooth l1 loss. need sigmoid in forward.
    
    整体的网络，就是
    一个 coord conv + 一个 contrained deformable pooling + 1 mlp.
    
    当前，暂时没有实现： coord conv, constrained deformable pooling.
    分别使用 conv2d 和 maxpool2d 来代替。
    """
    
    def __init__(self,
                 c=256,  # channel number for the output of FPN
                 coord_conv_module=None, nk=(256, 128, 64),
                 constrained_deform_pool_module=None,
                 *args, **kwargs):
        super(FeatureSampling, self).__init__()
        
        self.C = c
        self.coord_conv_module = (
            coord_conv_module if coord_conv_module else
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1)
        )  # self.C to self.C
        if constrained_deform_pool_module:
            self.constrained_deform_pool_module = constrained_deform_pool_module
        else:
            self.constrained_deform_pool_module = nn.MaxPool2d(
                kernel_size=2, stride=2
            )  # (H, W) --> (H/2, W/2)
        
        self.mlp_module = nn.Conv2d(
            in_channels=self.C, out_channels=1, kernel_size=1, stride=1)
        self.Nk = nk
        self.args, self.kwargs = args, kwargs
        
        _reset_parameters(self)
        
    def forward(self, features, *args, **kwargs):
        r""" Generate Significance map for the input features.
        
        Args:
             features (List[torch.Tensor]): a list with each element corresponding to the feature map
               of a specified level. Each feature map's shape should be [batch_size, C, Hi, Wi].
               
        Returns:
            score_maps (List[torch.Tensor]): A list with each element corresponding to the feature map
              of a specified level. Each feature map's shape should be [batch_size, 1, Hi/2, Wi/2]
            descriptors (torch.Tensor): a tensor with shape of [B, \sum_i Nk_i, C], which will be the input
              of `Feature Grouping Network` after permutation.
            coordinates (torch.Tensor): a tensor with shape of [B, \sum_i Nk_i, 3] representing the coordinates
              for the corresponding feature vector. Form of coordinate should be (feature_level, r, c).
        """
        outputs = []
        descriptors, coordinates = None, None
        for i, feature in enumerate(features):
            B, C, Hi, Wi = feature.shape
            if self.coord_conv_module:
                feature = self.coord_conv_module(feature)  # [B, C, Hi, Wi]
            if self.constrained_deform_pool_module:
                feature = self.constrained_deform_pool_module(feature)  # (B, C, Hi/2, Wi/2)
                
            # generate significance map
            significance_map = self.mlp_module(feature).sigmoid().squeeze()  # (B, Hi/2, Wi/2)
            outputs.append(significance_map)
            
            # feature sampling
            nk = self.Nk[i]
            _, topk_indices = torch.topk(significance_map.flatten(-2, -1), nk, dim=-1)  # [B, nk]
            # topk_feats = feature.flatten(-2, -1).permute(0, 2, 1)[topk_indices]  # [B, nk, C]
            topk_feats = torch.gather(
                input=feature.flatten(-2, -1).permute(0, 2, 1),
                dim=1, index=torch.tile(topk_indices.unsqueeze(dim=-1), (C, ))
            )
            topk_indices_r, topk_indices_c = (
                torch.div(topk_indices, significance_map.shape[-1], rounding_mode="floor"),
                torch.remainder(topk_indices, significance_map.shape[-1])
            )
            topk_coords = torch.cat([
                torch.full([B, nk, 1], i, device=topk_indices_r.device),  # self.device
                topk_indices_r.unsqueeze(dim=-1), topk_indices_c.unsqueeze(dim=-1)],
                dim=-1
            )  # [B, nk, 3], (feature_level, r, c)
            if descriptors is None:
                descriptors = topk_feats
                coordinates = topk_coords
            else:
                descriptors = torch.cat(  # [B, \sum_i{nk_i}, C]
                    [descriptors, topk_feats], dim=1)
                coordinates = torch.cat(  # [B, \sum_i{nk_i} 3]
                    [coordinates, topk_coords], dim=1)
        return outputs, descriptors, coordinates


class FeatureGrouping(nn.Module):
    """
    4 transformer encoder layers.
    """
    def __init__(self,
                 c=256,
                 num_encoder_layer=4, model_dim=512, nhead=8,
                 pe_type="sin_cos", num_dims=3,
                 *args, **kwargs):
        super(FeatureGrouping, self).__init__()
        self.num_encoder_layer, self.model_dim, self.nhead = (
            num_encoder_layer, model_dim, nhead
        )
        self.C = c
        self.args, self.kwargs = args, kwargs
        
        self.input_proj = nn.Linear(in_features=self.C, out_features=self.model_dim, bias=True)
        
        self.pe_type, self.num_dims = pe_type, num_dims
        self.pe_table = PositionalEmbedding(
            pos_dim=self.model_dim, num_dims=self.num_dims, embed_type=self.pe_type)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.nhead)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=self.num_encoder_layer,
            norm=nn.LayerNorm(self.model_dim)
        )
        
        _reset_parameters(self)
    
    def forward(self, descriptors, coordinates, *args, **kwargs):
        """ Perform the Feature Grouping.
        
        Args:
             descriptors (torch.Tensor): [B, \\sum_i {Nk_i}, C]
             coordinates (torch.Tensor): [B, \\sum_i {Nk_i}, 3]
             
        Returns:
            descriptors:
        """
        # generate positional embedding
        coords_pe = self.pe_table(coordinates.permute(2, 0, 1).contiguous())  # [B, \sum_i nk, model_dim]
        descriptors = self.input_proj(descriptors)  # [B, \sum_i nk, model_dim]
        descriptors = (descriptors + coords_pe).permute(1, 0, 2)  # [Nk, B, model_dim]
        
        # pass descriptors through transformer encoder
        descriptors = self.encoder(descriptors)  # [Nk, B, model_dim], Nk = \sum_i {nk_i}
        return descriptors.permute(1, 0, 2).contiguous()  # [B, Nk, model_dim]


class FewNet(nn.Module):
    """
    1. feature sampling network;
    2. feature grouping netowrk;
    3. cls, detection head.
    
    current implementation contain no backbone.
    """
    def __init__(self,
                 fpn=None,
                 feature_sampling=None, feature_grouping=None,
                 target_mode="rbox", is_coord_norm=True, inner_channels=256,
                 *args, **kwargs):
        assert target_mode.lower() in ["rbox"], (
            "Current FewNet only support these target mode: {}"
            "However, your target_model is: {}".format(
                ["rbox", ],target_mode
            )
        )
        super(FewNet, self).__init__()
        self.args, self.kwargs = args, kwargs
        
        self.fpn, self.feature_sampling, self.feature_grouping = (
            fpn, feature_sampling, feature_grouping
        )
        self.target_mode = target_mode
        self.is_coord_norm = is_coord_norm  # 对于 detection 来说，坐标是否是 normalized ?
        self.C = inner_channels  # out_channels for feature pyramid network
        if self.target_mode.lower() == "rbox":
            self.cls_head = nn.Linear(self.C, 1)  # cls_logits
            self.xywh_head = nn.Sequential(
                nn.Linear(self.C, 4)
            ) if not self.is_coord_norm else nn.Sequential(
                nn.Linear(self.C, 4),
                nn.Sigmoid()
            )
            self.angle_head = nn.Linear(self.C, 1)
        else:
            pass
        
        _reset_parameters(self)  # Parameter initialization
        
    def forward(self, features, *args, **kwargs):
        out = OrderedDict()
        # pass features through fpn
        p2, p3, p4, p5 = self.fpn(features)  # p_i for i-th feature level, stride 2^i
        features = (p2, p3, p4)  # 注意这里的 顺序 和 feature_sampling 中定义的 N_k 的顺序
        
        # pas features through feature sampling network
        score_maps, descriptors, coordinates = self.feature_sampling(features)  # [B, C, H, W], [B, Nk, C]
        
        # pass features through feature grouping network
        descriptors = self.feature_grouping(descriptors, coordinates)  # [B, Nk, C]
        
        # pass descriptors through head
        logits = self.cls_head(descriptors)
        boxes = self.xywh_head(descriptors)
        angle = self.angle_head(descriptors)
        
        out.update(  # 注意这里 和 FewNetLoss 的对应
            score_map=score_maps,  # score_maps --> score_map
            logits=logits, boxes=boxes, angle=angle
        )
        return out


def build_fewnet(
        # param for fpn
        need_conv_fpn=True, in_channels=(64, 128, 256, 512), inner_channels=256, bias=False,
        # param for feature sampling network
        coord_conv_module=None, nk=(256, 128, 64), constrained_deform_pool_module=None,
        # param for feature grouping network
        num_encoder_layer=4, model_dim=256, nhead=8,  # model_dim is 256 instead of 512
        pe_type="sin_cos", num_dims=3,
        # param for fewnet
        target_mode="rbox", is_coord_norm=True,
        device=None
):
    conv_fpn = FPN(
        need_conv_fpn=need_conv_fpn,
        in_channels=in_channels, inner_channels=inner_channels, bias=bias)
    feature_sampling = FeatureSampling(
        inner_channels, coord_conv_module=coord_conv_module, nk=nk,
        constrained_deform_pool_module=constrained_deform_pool_module
    )
    feature_grouping = FeatureGrouping(
        inner_channels, num_encoder_layer=num_encoder_layer, model_dim=model_dim,
        nhead=nhead, pe_type=pe_type, num_dims=num_dims
    )
    fewnet = FewNet(
        fpn=conv_fpn, feature_sampling=feature_sampling, feature_grouping=feature_grouping,
        target_mode=target_mode, is_coord_norm=is_coord_norm,
        inner_channels=inner_channels
    )
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    assert "cuda" in device or "cpu" in device
    
    return fewnet.to(device)
