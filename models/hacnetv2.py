import torch
import torch.nn as nn
from .modules import (
    effective_stem_block,
    hybird_aspa,
    upsampling_branch_block
)

class HACNetV2(nn.Module):
    """
    HACNetV2: Hybrid Attention-based Crack Detection Network
    """
    def __init__(self, channels=32):
        super(HACNetV2, self).__init__()
        # Initial feature extraction
        self.stemf = effective_stem_block(3, channels, stride=1)

        # Fine-grained feature processing
        self.hybird_aspa_f1 = hybird_aspa(channels, channels, 3, [1,3,9,27])
        self.hybird_aspa_f2 = hybird_aspa(channels, channels, 3, [1,3,9,27])
        self.hybird_aspa_f3 = hybird_aspa(channels, channels, 3, [1,3,9,27])
        
        # Coarse-grained feature processing
        mid_channels = channels * 2
        self.stemh = effective_stem_block(channels, mid_channels, stride=2)
        
        # Upsampling branches
        self.up_p1 = upsampling_branch_block(mid_channels, channels)
        self.up_p2 = upsampling_branch_block(mid_channels, channels)
        self.up_p3 = upsampling_branch_block(mid_channels, channels)

        # Coarse-grained ASPA modules
        self.hybird_aspa_h1 = hybird_aspa(mid_channels, mid_channels, 3, [1,3,9,27])
        self.hybird_aspa_h2 = hybird_aspa(mid_channels, mid_channels, 3, [1,3,9,27])
        self.hybird_aspa_h3 = hybird_aspa(mid_channels, mid_channels, 3, [1,3,9,27])

        # Output layers
        self.conv_out = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding="same")
        self.branch_o1 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding="same")
        self.branch_o2 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding="same")
        self.branch_o3 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding="same")

    def forward(self, x):
        # Initial feature extraction
        f0 = self.stemf(x)
        h0 = self.stemh(f0)

        # Coarse-grained processing
        h1 = self.hybird_aspa_h1(h0)
        hf1 = self.up_p1(h1)
        h2 = self.hybird_aspa_h2(h1)
        hf2 = self.up_p2(h2)
        h3 = self.hybird_aspa_h3(h2)
        hf3 = self.up_p3(h3)

        # Fine-grained processing with skip connections
        f1 = self.hybird_aspa_f1(f0)
        f1 = f1 + hf1
        f2 = self.hybird_aspa_f2(f1)
        f2 = f2 + hf2
        f3 = self.hybird_aspa_f3(f2)
        f3 = f3 + hf3

        # Multi-scale outputs
        o1 = self.branch_o1(f1)
        o2 = self.branch_o2(f2)
        o3 = self.branch_o3(f3)
        o4 = self.conv_out(torch.cat([o1, o2, o3], dim=1))

        outputs = [o1, o2, o3, o4]
        return outputs 