"""
实现 DBNet 中的 Feature Pyramid Network 的基本功能
"""
import torch
from torch import nn
import torch.nn.functional as F


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        # TODO 针对 m.bias
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)
        
        
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
