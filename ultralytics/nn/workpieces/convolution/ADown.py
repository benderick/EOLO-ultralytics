import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

__all__ = ['ADown']


class ADown(nn.Module):
    style = "ij" #接受输入和输出
    def __init__(self, c1, c2, *args):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16)
    model = ADown(32, 32)
    print(model(x).shape)