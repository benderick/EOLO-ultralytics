import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

__all__ = ['SBA']

def Upsample(x, size, align_corners = False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

class SBA(nn.Module):
    style = "l"
    def __init__(self, inc, input_dim=64, *args):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = Conv(input_dim//2, input_dim//2, 1)
        self.d_in2 = Conv(input_dim//2, input_dim//2, 1)       
                
        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim//2, kernel_size=1, bias=False)
        
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        H_feature, L_feature = x

        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)
        
        g_L_feature =  self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)
        
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature, size= L_feature.size()[2:], align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature, size= H_feature.size()[2:], align_corners=False) 
        
        H_feature = Upsample(H_feature, size = L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out

if __name__ == "__main__":
    # Example usage
    sba = SBA(inc=[256, 128], input_dim=64)
    x = (torch.randn(1, 256, 16, 16), torch.randn(1, 128, 32, 32))
    output = sba(x)
    print(output.shape)  # Should be (1, 64, 32, 32) if input_dim is set to 64