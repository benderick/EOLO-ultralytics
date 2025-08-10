import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f, C3, Conv, GhostConv, C2

"""
CSP Block : c1, c2, n=1, shortcut=False, g=1, e=0.5, *args
    CSP-T Block: T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args
    CSP-B-T Block: B, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args
    
Bottleneck Block : c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5
    Bottleneck-T Block: T, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5

Attn Block : c
    Attn-T Block : T, c

CSPAttn Block : c1, c2, n=1, e=0.5
    CSPAttn-T Block : T: nn.Module, c1, c2, n=1, e=0.5
"""

# ---------------------------------
class Bottleneck_T(nn.Module):
    """Standard bottleneck."""

    def __init__(self, T, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = T(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class GhostBottleneck_T(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, T, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            T(c_, c_),
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + x if self.add else self.conv(x)
# --------------------------------- 
class C1_D(nn.Module):
    """CSP Bottleneck with 1 convolution."""
    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(T(c2, c2) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y

class C1_T(nn.Module):
    """CSP Bottleneck with 1 convolution."""
    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck_T(T, c2, c2, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y

class C1_B_T(nn.Module):
    """CSP Bottleneck with 1 convolution."""
    def __init__(self, B, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(B(T, c2, c2, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y

# ---------------------------------
class C2_D(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  #optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(T(self.c, self.c) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))
    
class C2_T(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  #optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck_T(T, self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))
    
class C2_B_T(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, B, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  #optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(B(T, self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))
    
# ---------------------------------  
class C2f_D(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(T(self.c, self.c) for _ in range(n))
        
class C2f_T(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_T(T, self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class C2f_B_T(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, B, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(B(T, self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# ---------------------------------
class C3_D(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(T(c_, c_) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3_T(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_T(T, c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    
class C3_B_T(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, B, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(B(T, c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# ---------------------------------
class C3f_D(nn.Module):
    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(T(c_, c_) for _ in range(n))

    def forward(self, x):
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))

class C3f_T(nn.Module):
    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_T(T, c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))
    
class C3f_B_T(nn.Module):
    def __init__(self, B, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(B(T, c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))

# ---------------------------------
class C3k_D(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(T(c_, c_) for _ in range(n)))

class C3k_T(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_T(T, c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))
        
class C3k_B_T(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, B, T, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(B(T, c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))
        
# --------------------------------- 
class C3k2_D(C2f):
    def __init__(self, T, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_D(T, self.c, self.c, 2, shortcut, g) if c3k else T(self.c, self.c) for _ in range(n)
        )

class C3k2_T(C2f):
    def __init__(self, T, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_T(T, self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_T(T, self.c, self.c, shortcut, g) for _ in range(n)
        )

class C3k2_B_T(C2f):
    def __init__(self, B, T, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_B_T(B, T, self.c, self.c, 2, shortcut, g) if c3k else B(T, self.c, self.c, shortcut, g) for _ in range(n)
        )  
   
# ---------------------------------
class AttnBlock_T(nn.Module):
    def __init__(self, T, c) -> None:
        super().__init__()
        self.attn = T(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = True

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
class C2Attn_T(C2):
    def __init__(self, T, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.Sequential(*(AttnBlock_T(T, self.c) for _ in range(n)))

class C2fAttn_T(C2f):
    def __init__(self, T, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(AttnBlock_T(T, self.c) for _ in range(n))

# -----------------------------------------
 
class CSP_T(C2f):
    def __init__(self, T, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(T(self.c) for _ in range(n))

class PSABlock_T(nn.Module):
    def __init__(self, T,  c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = T(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
 
class C2PSA_T(nn.Module):
    style = "nij"
    def __init__(self, T: nn.Module, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock_T(T, self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))