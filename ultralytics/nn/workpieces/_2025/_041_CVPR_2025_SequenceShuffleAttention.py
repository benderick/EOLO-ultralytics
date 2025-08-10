import torch
import torch.nn as nn


__all__ = ["SequenceShuffleAttention"]
class SequenceShuffleAttention(nn.Module):
    style = "i"
    def __init__(self, dim, hidden_features=None, group=4, act_layer=nn.GELU):
        super().__init__()
        self.group = group  # 分组数，用于通道重排
        
        # 定义 gating 部分，使用平均池化后经过卷积层和 Sigmoid 激活
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化，输出大小为 1x1
            nn.Conv2d(dim, dim, groups=self.group, kernel_size=1, stride=1, padding=0),  # 卷积层，使用分组卷积
            nn.Sigmoid()  # Sigmoid 激活函数
        )

    # 通道重排操作，打乱输入张量的通道
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()  # 获取输入张量的维度
        assert num_channels % self.group == 0  # 确保通道数可以被分组数整除
        group_channels = num_channels // self.group  # 每个组的通道数

        # 将输入张量 reshape 成 (batch_size, group_channels, group, height, width)
        x = x.reshape(batchsize, group_channels, self.group, height, width)
        # 调整维度顺序，使得每个组的通道打乱
        x = x.permute(0, 2, 1, 3, 4)
        # 将张量恢复成 (batch_size, num_channels, height, width)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    # 通道重新排列操作，和通道重排类似，但这里不进行打乱，只是重新排列
    def channel_rearrange(self, x):
        batchsize, num_channels, height, width = x.data.size()  # 获取输入张量的维度
        assert num_channels % self.group == 0  # 确保通道数可以被分组数整除
        group_channels = num_channels // self.group  # 每个组的通道数

        # 将输入张量 reshape 成 (batch_size, group, group_channels, height, width)
        x = x.reshape(batchsize, self.group, group_channels, height, width)
        # 调整维度顺序，使得每个组的通道重新排列
        x = x.permute(0, 2, 1, 3, 4)
        # 将张量恢复成 (batch_size, num_channels, height, width)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    # 前向传播函数
    def forward(self, x):
        y = x  # 保存输入张量，用于残差连接
        x = self.channel_shuffle(x)  # 对输入进行通道重排
        x = self.gating(x)  # 使用 gating 对输入进行处理
        x = self.channel_rearrange(x)  # 对处理后的张量进行通道重新排列

        return y * x  # 将原始输入与处理后的输出相乘

from torchinfo import summary  # 需要安装 torchinfo：pip install torchinfo

if __name__ == '__main__':
    # 设置输入参数
    batch_size = 1      # 批次大小
    in_channels = 64    # 输入通道数
    out_channels = 64   # 输出通道数
    input_resolution = (256, 256)  # 输入分辨率

    # 创建随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, in_channels, input_resolution[0], input_resolution[1]).cuda()  # 输入张量

    # 创建 SequenceShuffleAttention 模块
    model = SequenceShuffleAttention(dim=in_channels).cuda()

    # 使用 torchinfo 进行模型分析
    summary(model, input_size=(batch_size, in_channels, input_resolution[0], input_resolution[1]))
    print("\n微信公众号: AI缝合术!\n")

    # 前向传播
    output = model(x)

    # 打印输入和输出张量的形状
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")