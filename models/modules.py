import torch
import torch.nn as nn

class CrackAM(nn.Module):
    """Crack Attention Module for feature refinement"""
    def __init__(self, channels, rate=1, add_maxpool=False, **_):
        super(CrackAM, self).__init__()
        self.fc = nn.Conv2d(int(channels), channels, kernel_size=1, padding=0)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        max_pool_h = torch.max(x, dim=3)[0]  # (N, C, H, 1)
        max_pool_v = torch.max(x, dim=2)[0]  # (N, C, 1, W)
        xtmp = torch.concat((max_pool_h, max_pool_v), dim=2)  # Shape: [batch_size, channels, width+height]
        x_se = xtmp.mean((2), keepdim=True).unsqueeze(-1)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)

def Conv2dBN(in_channels, out_channels, kernel_size=3, stride=1, rate=1, name=None):
    """Basic convolution block with batch normalization"""
    if stride > 1:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=rate, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=rate, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

def Conv2dBNv2(in_channels, out_channels, kernel_size=3, rate=1, name=None):
    """Advanced convolution block with bottleneck structure"""
    mid_channels = int(out_channels/2)
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding="same"),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, dilation=rate, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def EConv2dBN(in_channels, out_channels, kernel_size=3, rate=1, name=None):
    """Efficient convolution block with separable convolutions"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=1, dilation=rate, padding="same"),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), stride=1, dilation=rate, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def upsampling_branch_block(in_channels, out_channels, scale_factor=(2,2), name=None):
    """Upsampling block for feature map expansion"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=scale_factor)
    )

class hybird_aspa(nn.Module):
    """Hybrid Attention-based Spatial Pyramid Attention module"""
    def __init__(self, in_channels, out_channels, kernel_size=3, rates=[1,1,1,1], hs_att=True):
        super(hybird_aspa, self).__init__()
        self.Conv2dBN1 = Conv2dBNv2(in_channels, out_channels, kernel_size, rate=rates[0])
        self.Conv2dBN2 = Conv2dBNv2(in_channels, out_channels, kernel_size, rate=rates[1])
        self.Conv2dBN3 = Conv2dBNv2(in_channels, out_channels, kernel_size, rate=rates[2])
        self.Conv2dBN4 = Conv2dBNv2(in_channels, out_channels, kernel_size, rate=rates[3])
        self.att = CrackAM(out_channels)
        self.conv11 = Conv2dBN(out_channels, out_channels, kernel_size=1)
        self.conv_out = EConv2dBN(out_channels, out_channels, rate=1)

    def forward(self, x):
        d1 = self.Conv2dBN1(x)
        d2 = self.Conv2dBN2(d1)
        d3 = self.Conv2dBN3(d2)
        d4 = self.Conv2dBN4(d3)
        o1 = d1 + d2 + d3 + d4
        o1 = self.att(o1)
        o2 = self.conv11(o1)
        o2 = x + o2
        o3 = self.conv_out(o2)
        return o3

class effective_stem_block(nn.Module):
    """Efficient stem block for initial feature extraction"""
    def __init__(self, in_channels, out_channels, stride):
        super(effective_stem_block, self).__init__()
        self.conv1 = Conv2dBN(in_channels, int(out_channels/2), kernel_size=(1,5), stride=stride)
        self.conv2 = Conv2dBN(in_channels, int(out_channels/2), kernel_size=(5,1), stride=stride)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        o1 = torch.cat((d1, d2), dim=1)
        return o1 