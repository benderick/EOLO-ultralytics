import torch
import torch.nn as nn
import torch.nn.functional as F

class wConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, bias=False):
        super(wConv2d, self).__init__()       
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.groups = groups         
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')  
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device),torch.tensor([1.0], device=device),torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

        if self.Phi.shape != (kernel_size, kernel_size):
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size ({kernel_size}, {kernel_size})")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
    

if __name__ == "__main__":

    # 创建输入张量：形状 [B, C, H, W]
    x = torch.randn(1, 3, 256, 256)

    # 实例化模型
    model = wConv2d(in_channels=3, out_channels=32, kernel_size=3, den=[0.7], stride=1, padding=1, groups=1, bias=False)

    # # 设备配置：使用GPU（如果可用）
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = x.to(device)
    # model = model.to(device)

    # 前向传播
    output = model(x)

    # 打印模型结构
    print("例: 输入1*3*256*256, \n使用3*3卷积核(通道数in:3, out:32)\n")
    print("微信公众号:AI缝合术")

    # 打印输入输出形状
    print("输入形状:", x.shape)     # [B, C, H, W]
    print("输出形状:", output.shape)  # [B, C_out, H_out, W_out]
