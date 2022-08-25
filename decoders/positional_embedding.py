"""
Utilities for positional embedding generation.
"""

import torch
from torch import nn
from collections import OrderedDict


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 max_pos_len=3000, pos_dim=None, num_dims=2,
                 embed_type="sin_cos", **kwargs):
        super(PositionalEmbedding, self).__init__()
        
        self.max_pos_len, self.pos_dim = max_pos_len, pos_dim
        self.num_dims = num_dims
        self.embed_type = embed_type
        self.kwargs = kwargs
        self.model_dim = self.kwargs.get("model_dim", self.pos_dim)
        self.embed_tables = self.build_embed_table()
        self.static = None  # whether pe_table will be updated by bp
        
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
    
    def forward(self, coords):
        """传入 坐标 的矩阵，并且 返回对应的 positional embedding.
        
        Args:
            coords (torch.Tensor): [num_dims, ...]. coords[i] corresponds to coordinates from the view of
              i-th dims.
              
        Returns:
            out (torch.Tensor): [..., pos_dim], positional embedding tensor.
        """
        assert coords.shape[0] == self.num_dims, (
            "Please check your coords, since the shape of coords: {}, "
            "However, the num_dims: {}".format(coords.shape, self.num_dims)
        )
        
        out = torch.zeros([*coords.shape[1:], self.pos_dim]).to(self.device)
        for i in range(coords.shape[0]):
            out += self.embed_tables[i](coords[i])
        return out
        
    def build_embed_table(self):
        embed_tables = OrderedDict()
        d_model, position_bias = self.model_dim, 0
        if self.embed_type == "sin_cos":
            self.static = True  # 使用静态的 embedding table
            for k in range(self.num_dims):
                i_mat = torch.arange(start=0, end=self.pos_dim, step=2) + position_bias
                i_mat = torch.pow(10000., i_mat / d_model).reshape([1, -1])
                pos_mat = torch.arange(start=0, end=self.max_pos_len, step=1).reshape([-1, 1])
                
                table_weight = torch.zeros(
                    [self.max_pos_len, self.pos_dim], requires_grad=False,
                    device=self.device
                )
                table_weight[:, 0::2] = torch.sin(pos_mat / i_mat)  # 自动 broadcast
                table_weight[:, 1::2] = torch.cos(pos_mat / i_mat)
                embed_tables.update({
                    k: nn.Embedding.from_pretrained(table_weight)
                })
                position_bias += self.max_pos_len  # 对 position_bias 加以更新
        else:
            raise ValueError("Please check your embed type: {}".foramt(self.embed_type))
        
        return embed_tables

