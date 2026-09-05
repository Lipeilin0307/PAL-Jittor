# -*- coding: utf-8 -*-
"""ACM_No_Sigmoid 的 Jittor 迁移版。
源: PAL/model/ACM/ACM_no_sigmoid.py (torch, 170 行)。

迁移要点:
1. torchvision.models.resnet.BasicBlock 仅在 __init__ 借类定义(无预训练权重),
   此处 vendor 一个 Jittor 版 BasicBlock(conv3x3->BN->ReLU->conv3x3->BN + shortcut)。
2. BatchNorm momentum: 探针实测 Jittor nn.BatchNorm2d 的 momentum 与 torch 同语义
   (running = (1-m)*running + m*batch_stat), 直接照搬数值:
   stem/downsample/head 的 BN momentum=0.9, BasicBlock 内部 BN 用 torch 默认 0.1。
3. GroupNorm / ConvTranspose2d / MaxPool2d / AdaptiveAvgPool2d / Dropout 直换。
4. forward 末尾 F.interpolate(scale_factor=4, mode='bilinear') (torch 默认
   align_corners=False) -> jt.nn.interpolate 同参数, 探针已逐点核对一致。
5. 类名保持 ACM_No_Sigmoid, 无参构造; 属性名与 torch 版完全一致,
   保证 state_dict 键名 1:1 对应(jittor BN 无 num_batches_tracked, 转换时跳过)。
"""
import jittor as jt
from jittor import nn

from .fusion import AsymBiChaFuse


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding, no bias"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution, no bias"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    """vendor 自 torchvision.models.resnet.BasicBlock (expansion=1 分支)。

    torch 原版中 BasicBlock 内部 bn1/bn2 由 norm_layer(planes) 构造,
    使用 torch 默认 momentum=0.1 (与 ACM 外层 stem/downsample 的 0.9 不同),
    此处显式保留该行为。
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, momentum=0.1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=momentum)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=momentum)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ACM_No_Sigmoid(nn.Module):
    def __init__(self, in_channels=3, layers=[3, 3, 3], channels=[8, 16, 32, 64],
                 fuse_mode='AsymBi', tiny=False, classes=1,
                 norm_layer=nn.BatchNorm2d, groups=1, norm_kwargs=None, **kwargs):
        super(ACM_No_Sigmoid, self).__init__()
        self.layer_num = len(layers)
        self.tiny = tiny
        self._norm_layer = norm_layer
        self.groups = groups
        self.momentum = 0.9
        stem_width = int(channels[0])  # channels: 8 16 32 64
        if tiny:  # 默认 False
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
        self.fuse2 = self._fuse_layer(fuse_mode, channels=channels[2])

        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1],
                                          kernel_size=4, stride=2, padding=1)
        self.uplayer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                         out_channels=channels[1], stride=1,
                                         in_channels=channels[1])
        self.fuse1 = self._fuse_layer(fuse_mode, channels=channels[1])

        self.head = _FCNHead(in_channels=channels[1], channels=classes,
                             momentum=self.momentum)

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

        x = self.stem(x)      # (N,16,128,128)
        c1 = self.layer1(x)   # (N,16,128,128)
        c2 = self.layer2(c1)  # (N,32, 64, 64)
        c3 = self.layer3(c2)  # (N,64, 32, 32)

        deconvc2 = self.deconv2(c3)        # (N,32, 64, 64)
        fusec2 = self.fuse2(deconvc2, c2)  # (N,32, 64, 64)
        upc2 = self.uplayer2(fusec2)       # (N,32, 64, 64)

        deconvc1 = self.deconv1(upc2)      # (N,16,128,128)
        fusec1 = self.fuse1(deconvc1, c1)  # (N,16,128,128)
        upc1 = self.uplayer1(fusec1)       # (N,16,128,128)

        pred = self.head(upc1)             # (N,1,128,128)

        if self.tiny:
            out = pred
        else:
            # jittor 1.3.8.5 融合 codegen bug 规避: C=1 的 bias 广播加在与下游
            # interpolate 融合时生成未定义标识符(op0_outputstrideN), 编译失败。
            # stop_fuse() 打断融合边界, 数值完全不变。
            out = nn.interpolate(pred.stop_fuse(), scale_factor=4, mode='bilinear')  # (N,1,512,512)

        return out

    def evaluate(self, x):
        return self.execute(x)


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, momentum,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1,
                      bias=False),                                        # 0
            norm_layer(inter_channels, momentum=momentum),                # 1
            nn.ReLU(),                                                    # 2
            nn.Dropout(0.1),                                              # 3
            nn.Conv2d(inter_channels, channels, kernel_size=1)            # 4
        )

        # jittor 1.3.8.5 融合 codegen bug 规避: 末层 conv 的 bias 形状为 [1](classes=1),
        # 其 array->broadcast_to->add 融合内核会引用未定义标识符 op0_outputstrideN
        # 导致编译失败。对 bias 标记 stop_fuse 使该参数不进入融合内核, 数值不变。
        self.block[4].bias.stop_fuse()

    def execute(self, x):
        return self.block(x)
