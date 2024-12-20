import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


class PPM(nn.Module):

    def __init__(self,
                 in_channels,
                 pool_scales=(1, 2, 3, 6),
                 out_channels=256,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            ) for scale in pool_scales
        ])
        self.conv = ConvModule(
            in_channels + len(pool_scales) * out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        ppm_outs = [x]
        for i, stage in enumerate(self.stages):
            ppm = stage(x)
            ppm = F.interpolate(ppm, size=x.size()[2:], mode='bilinear', align_corners=True)
            ppm_outs.append(ppm)
        x = torch.cat(ppm_outs, dim=1) 
        x = self.conv(x)                
        return x


@MODELS.register_module()
class EnhancedSegformerHead(BaseDecodeHead):

    def __init__(self, ppm_out_channels=256, pool_scales=(1, 2, 3, 6), interpolate_mode='bilinear', **kwargs):
        super(EnhancedSegformerHead, self).__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index), "Number of in_channels must match number of in_index"

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.ppm = PPM(
            in_channels=self.channels * num_inputs, 
            pool_scales=pool_scales,
            out_channels=ppm_out_channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        self.cls_seg = nn.Conv2d(
            ppm_out_channels,
            self.num_classes,
            kernel_size=1
        )

    def forward(self, inputs):

        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            x = conv(x)
            x = resize(
                input=x,
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )
            outs.append(x)

        x = torch.cat(outs, dim=1)

        x = self.ppm(x)

        out = self.cls_seg(x)

        return out