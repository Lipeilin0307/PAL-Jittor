# -*- coding: utf-8 -*-
"""ALC_No_Sigmoid 的 Jittor 迁移版。
源: PAL/model/ALC/ALC_no_sigmoid.py (torch, 230 行)。

迁移要点:
1. torchvision.models.resnet.BasicBlock / AsymBiChaFuse / _FCNHead 与 ACM 完全同源,
   直接复用 model/acm.py 与 model/fusion.py 的 Jittor 版
   (_FCNHead 自带末层 [1] bias 的 stop_fuse 规避, ALC 末层同样 classes=1 会命中
   同一融合 codegen bug)。
2. forward 内 7 处 torchvision.transforms.Resize 作用于 4D 张量:
   torchvision 0.14 transforms.Resize 默认 InterpolationMode.BILINEAR,
   对张量走 F.interpolate(mode='bilinear', align_corners=False, antialias=False)
   -> jt.nn.interpolate(x, size=[h,w], mode='bilinear')(探针已核对与 torch
   align_corners=False 逐点一致)。
   注意默认配置(layer_num=3, tinyFlag=False)下 L174/L180/L187 三处 Resize 的
   目标尺寸与输入尺寸相同(恒等), L199 为真正的 x2 上采样(H/2 -> H)。
3. BN momentum 语义已探明与 torch 一致: stem/downsample/head 0.9, block 内 0.1。
4. GroupNorm / ConvTranspose2d / MaxPool2d / Dropout 直换。
5. 类名 ALC_No_Sigmoid, 无参构造, 单输出, 属性名与 torch 版一致保证
   state_dict 键名 1:1 (jittor BN 无 num_batches_tracked, 转换时跳过)。
"""
import jittor as jt
from jittor import nn

from .acm import BasicBlock, _FCNHead, conv1x1
from .fusion import AsymBiChaFuse


class ALC_No_Sigmoid(nn.Module):
    def __init__(self, in_channels=3, layers=[4, 4, 4], channels=[8, 16, 32, 64],
                 fuse_mode='AsymBi', act_dilation=16, classes=1, tinyFlag=False,
                 norm_layer=nn.BatchNorm2d, groups=1, norm_kwargs=None, **kwargs):
        super(ALC_No_Sigmoid, self).__init__()

        self.layer_num = len(layers)
        self.tinyFlag = tinyFlag
        self.groups = groups
        self._norm_layer = norm_layer
        stem_width = int(channels[0])
        self.momentum = 0.9
        if tinyFlag:
            self.stem = nn.Sequential(
                norm_layer(in_channels, momentum=self.momentum),
                nn.Conv2d(in_channels, stem_width * 2, kernel_size=3, stride=1,
                          padding=1, bias=False),
                norm_layer(stem_width * 2, momentum=self.momentum),
                nn.ReLU()
            )
        else:
            self.stem = nn.Sequential(
                norm_layer(in_channels, momentum=self.momentum),          # 0
                nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2,
                          padding=1, bias=False),                         # 1
                norm_layer(stem_width, momentum=self.momentum),           # 2
                nn.ReLU(),                                                # 3
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1, bias=False),                         # 4
                norm_layer(stem_width, momentum=self.momentum),           # 5
                nn.ReLU(),                                                # 6
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1, bias=False),                         # 7
                norm_layer(stem_width * 2, momentum=self.momentum),       # 8
                nn.ReLU(),                                                # 9
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)          # 10
            )

            self.head = _FCNHead(in_channels=channels[0], channels=classes,
                                 momentum=self.momentum)

            self.layer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                           out_channels=channels[1],
                                           in_channels=channels[1], stride=1)

            self.layer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                           out_channels=channels[2], stride=2,
                                           in_channels=channels[1])

            self.layer3 = self._make_layer(block=BasicBlock, blocks=layers[2],
                                           out_channels=channels[3], stride=2,
                                           in_channels=channels[2])
            self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2],
                                              kernel_size=4, stride=2, padding=1)
            self.uplayer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                             out_channels=channels[2], stride=1,
                                             in_channels=channels[2])

            self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1],
                                              kernel_size=4, stride=2, padding=1)

            self.deconv0 = nn.ConvTranspose2d(channels[1], channels[0],
                                              kernel_size=4, stride=2, padding=1)

            self.uplayer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                             out_channels=channels[1], stride=1,
                                             in_channels=channels[1])

            if self.layer_num == 4:
                self.layer4 = self._make_layer(block=BasicBlock, blocks=layers[3],
                                               out_channels=channels[3], stride=2,
                                               in_channels=channels[3])

            if self.layer_num == 4:
                self.fuse34 = self._fuse_layer(fuse_mode, channels=channels[3])

            self.fuse23 = self._fuse_layer(fuse_mode, channels=channels[2])
            self.fuse12 = self._fuse_layer(fuse_mode, channels=channels[1])

    def _make_layer(self, block, out_channels, in_channels, blocks, stride):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or out_channels != in_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels * block.expansion, momentum=self.momentum),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample,
                            self.groups, norm_layer=norm_layer))
        self.inplanes = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, self.groups,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _fuse_layer(self, fuse_mode, channels):
        if fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuse(channels=channels)
        else:
            raise ValueError('Unknown fuse_mode')
        return fuse_layer

    def execute(self, x):
        _, _, hei, wid = x.shape

        x = self.stem(x)      # (N,16,H/4,W/4)
        c1 = self.layer1(x)   # (N,16,H/4,W/4)
        c2 = self.layer2(c1)  # (N,32,H/8,W/8)

        out = self.layer3(c2)  # (N,64,H/16,W/16)

        if self.layer_num == 4:
            c4 = self.layer4(out)  # (N,64,H/32,W/32)
            if self.tinyFlag:
                c4 = nn.interpolate(c4, size=[hei // 4, wid // 4], mode='bilinear')
            else:
                c4 = nn.interpolate(c4, size=[hei // 16, wid // 16], mode='bilinear')
            out = self.fuse34(c4, out)

        if self.tinyFlag:
            out = nn.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear')
        else:
            out = nn.interpolate(out, size=[hei // 16, wid // 16], mode='bilinear')  # 恒等

        out = self.deconv2(out)      # (N,32,H/8,W/8)
        out = self.fuse23(out, c2)   # (N,32,H/8,W/8)
        if self.tinyFlag:
            out = nn.interpolate(out, size=[hei, wid], mode='bilinear')
        else:
            out = nn.interpolate(out, size=[hei // 8, wid // 8], mode='bilinear')  # 恒等

        out = self.deconv1(out)      # (N,16,H/4,W/4)
        out = self.fuse12(out, c1)   # (N,16,H/4,W/4)

        out = self.deconv0(out)      # (N,8,H/2,W/2)
        pred = self.head(out)        # (N,1,H/2,W/2)

        if self.tinyFlag:
            out = pred
        else:
            out = nn.interpolate(pred, size=[hei, wid], mode='bilinear')  # x2 上采样

        return out

    def evaluate(self, x):
        return self.execute(x)
