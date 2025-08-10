import  torch
from    torch import nn, einsum
from    einops import rearrange
import  torch.nn.functional as F


__all__ = ["Channel_Attention", "C2fAttn_Channel_Attention"]
class Channel_Attention(nn.Module):
    style = "i"
    def __init__(
        self, 
        dim, 
        heads=8, 
        bias=False, 
        dropout = 0.,
        window_size = 4
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
       
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        # -----------------------------------------
        b,c,h,w = x.shape
        ph, pw = self.ps, self.ps
        # 计算需要pad的大小
        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        # 如果需要，进行 padding（右边、下边补0）
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        origin_h, origin_w = h, w
        h, w = x.shape[2], x.shape[3]  # 记录 pad 后的尺寸
		# -----------------------------------------
        
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1) 

        q,k,v = map(lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)', ph=self.ps, pw=self.ps, head=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(out, 'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)', h=h//self.ps, w=w//self.ps, ph=self.ps, pw=self.ps, head=self.heads)
        out = self.project_out(out)
        
        # -----------------------------------------
        # 去掉 padding
        if pad_h != 0 or pad_w != 0:
            out = out[:, :, :origin_h, :origin_w]
        # -----------------------------------------
        
        return out




# ----
from ultralytics.nn.modules import C2fAttn_T
class C2fAttn_Channel_Attention(nn.Module):
    style = "nij"
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.m = C2fAttn_T(Channel_Attention, c1, c2, n=n, e=e)

    def forward(self, x):
        return self.m(x)
# ----
if __name__ == "__main__":
    # 设定测试参数
    batch_size = 1  # 批大小
    channels = 32   # 通道数
    height = 13    # 特征图高度
    width = 11     # 特征图宽度
    heads = 8       # 注意力头数
    window_size = 4 # 窗口大小
    
    # 创建测试输入张量
    x = torch.randn(batch_size, channels, height, width)
    
    # 初始化通道注意力模块
    channel_attn = Channel_Attention(dim=channels, heads=heads, window_size=window_size)
    print(channel_attn)
    print("微信公众号: AI缝合术!")
    
    # 前向传播
    output = channel_attn(x)
    
    # 打印输出形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)
    m = C2fAttn_Channel_Attention(32, 64)
    o = m(x)
    print(o.shape)
    