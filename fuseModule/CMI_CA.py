import torch
import torch.nn as nn

# from fuseModule.CoordAttention import CoordAtt
from thop import clever_format
from thop import profile

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1, padding=0, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class CMI_CA(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CMI_CA, self).__init__()

        self.transitionLayer = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        '''
        self.transitionLayer2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))

        self.ca_rgb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // ratio),
            nn.Conv2d(in_channels//ratio, in_channels, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid())

        self.ca_t = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // ratio),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid())
            
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        '''

        self.map_rgb = ChannelAttentionModule(in_channels, in_channels, ratio)
        self.map_t   = ChannelAttentionModule(in_channels, in_channels, ratio)

    def forward(self, rgb, t):

        rgbt = torch.cat([rgb, t], dim=1)
        rgbt_TL = self.transitionLayer(rgbt)

        map_rgb     = self.map_rgb(rgbt_TL)
        rgb_enhance = rgb * map_rgb + rgbt_TL

        map_t     = self.map_t(rgbt_TL)
        t_enhance = t * map_t + rgbt_TL

        return rgb_enhance, t_enhance

if __name__ == '__main__':

    c = 2048

    IS = CMI_CA(in_channels=c)

    a = torch.randn(2, c, 8, 8)
    b = torch.randn(2, c, 8, 8)

    fuse = IS(a, b)

    FLOPs, params = profile(IS, inputs=(a, b))
    FLOPs, params = clever_format([FLOPs, params], "%.3f")
    print('params: ', params)
    print('FLOPs: ', FLOPs)