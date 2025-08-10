import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules import C3k2_T

__all__ = ["C3k2_DWRSeg"]

class DWR(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv_3x3 = Conv(dim, dim // 2, 3)

        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)

        self.conv_1x1 = Conv(dim * 2, dim, k=1)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out

class DWRSeg_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, k=1)

        self.dcnv3 = DWR(out_channels)

        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)

        x = self.dcnv3(x)

        x = self.gelu(self.bn(x))
        return x

class C3k2_DWRSeg(nn.Module):
    style = "nij"
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.m = C3k2_T(DWRSeg_Conv, c1, c2, n=n, c3k=c3k, e=e, g=g, shortcut=shortcut)
    def forward(self, x):
        return self.m(x)


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v1 = C3k2_DWRSeg(64, 64)

    out = mobilenet_v1(image)
    print(out.size())