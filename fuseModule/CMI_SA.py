import torch
import torch.nn as nn

# from fuseModule.CoordAttention import CoordAtt
from thop import clever_format
from thop import profile


class CMI_SA(nn.Module):
    def __init__(self, in_channels):
        super(CMI_SA, self).__init__()

        self.transitionLayer = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))

        self.conv1rgb = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False))

        self.conv2rgb = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False))

        self.conv1t = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False))

        self.conv2t = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False))

        # self.coordattSI1 = CoordAtt(inp=in_channels, oup=in_channels)
        # self.coordattSI2 = CoordAtt(inp=in_channels, oup=in_channels)

    def forward(self, rgb, t):
        rgbt = torch.cat([rgb, t], dim=1)
        rgbt_TL = self.transitionLayer(rgbt)

        rgb_max, _ = torch.max(rgb, dim=1, keepdim=True)
        rgb_max = self.conv2rgb(self.conv1rgb(rgb_max))
        SA_rgb = torch.sigmoid(rgb_max)
        rgb_enhance = rgbt_TL * SA_rgb + rgb
        # rgb_enhance = self.coordattSI1(rgb_enhance)

        t_max, _ = torch.max(t, dim=1, keepdim=True)
        t_max = self.conv2t(self.conv1t(t_max))
        SA_t = torch.sigmoid(t_max)
        t_enhance = rgbt_TL * SA_t + t
        # t_enhance = self.coordattSI2(t_enhance)

        return rgb_enhance, t_enhance

if __name__ == '__main__':

    c = 2048

    IS = CMI_SA(in_channels=c)

    a = torch.randn(2, c, 8, 8)
    b = torch.randn(2, c, 8, 8)

    # fuse = IS(a, b)

    FLOPs, params = profile(IS, inputs=(a, b))
    FLOPs, params = clever_format([FLOPs, params], "%.3f")
    print('params: ', params)
    print('FLOPs: ', FLOPs)