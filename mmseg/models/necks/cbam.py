import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmseg.registry import MODELS

@MODELS.register_module()
class CBAMNeck(BaseModule):
    
    def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7):
        super(CBAMNeck, self).__init__()
        assert len(in_channels) == len(out_channels), "in_channels and out_channels must have the same length"
        
        self.cbam_modules = nn.ModuleList()
        for in_ch, out_ch in zip(in_channels, out_channels):
            cbam = self._build_cbam(in_ch, out_ch, reduction, kernel_size)
            self.cbam_modules.append(cbam)
    
    def _build_cbam(self, in_channels, out_channels, reduction, kernel_size):
        class ChannelAttention(nn.Module):
            def __init__(self, in_channels, reduction=16):
                super(ChannelAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.max_pool = nn.AdaptiveMaxPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(in_channels, in_channels // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channels // reduction, in_channels, bias=False)
                )
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                B, C, _, _ = x.size()
                avg_out = self.fc(self.avg_pool(x).view(B, C))
                max_out = self.fc(self.max_pool(x).view(B, C))
                out = avg_out + max_out
                out = self.sigmoid(out).view(B, C, 1, 1)
                return x * out.expand_as(x)
        
        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=7):
                super(SpatialAttention, self).__init__()
                assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
                padding = (kernel_size - 1) // 2
                self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                concat = torch.cat([avg_out, max_out], dim=1)
                out = self.conv(concat)
                return x * self.sigmoid(out)
        
        class CBAM(nn.Module):
            def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7):
                super(CBAM, self).__init__()
                self.channel_attention = ChannelAttention(in_channels, reduction)
                self.spatial_attention = SpatialAttention(kernel_size)
                self.conv = ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=dict(type='SyncBN', requires_grad=True),
                    act_cfg=dict(type='ReLU')
                )
                self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
            
            def forward(self, x):
                out = self.channel_attention(x)
                out = self.spatial_attention(out)
                out = self.conv(out)
                residual = self.residual_conv(x)
                return out + residual
        
        return CBAM(in_channels, out_channels, reduction, kernel_size)
    
    def forward(self, inputs):
        assert len(inputs) == len(self.cbam_modules), "Number of input feature layers does not match number of CBAM modules"
        
        enhanced_features = []
        for i, (x, cbam) in enumerate(zip(inputs, self.cbam_modules)):
            enhanced_x = cbam(x)
            enhanced_features.append(enhanced_x)
        
        return enhanced_features