# -*- coding: utf-8 -*-
"""AsymBiChaFuse 的 Jittor 迁移版。
源: PAL/model/ACM/fusion.py (torch, 51 行)。
逐行对照: AdaptiveAvgPool2d/Conv2d/GroupNorm/ReLU/Sigmoid 结构、索引、参数完全一致，
保证 state_dict 键名与 torch 版一一对应。
"""
import jittor as jt
from jittor import nn


class AsymBiChaFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                                    # 0
            nn.Conv2d(self.channels, self.bottleneck_channels, 1, 1, 0),  # 1
            nn.GroupNorm(1, self.bottleneck_channels),                  # 2
            nn.ReLU(),                                                  # 3
            nn.Conv2d(self.bottleneck_channels, self.channels, 1, 1, 0),  # 4
            nn.GroupNorm(1, self.channels),                             # 5
            nn.Sigmoid()                                                # 6
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.channels, self.bottleneck_channels, 1, 1, 0),  # 0
            nn.GroupNorm(1, self.bottleneck_channels),                  # 1
            nn.ReLU(),                                                  # 2
            nn.Conv2d(self.bottleneck_channels, self.channels, 1, 1, 0),  # 3
            nn.GroupNorm(1, self.channels),                             # 4
            nn.Sigmoid()                                                # 5
        )

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, dilation=1),         # 0
            nn.GroupNorm(1, self.channels),                             # 1
            nn.ReLU()                                                   # 2
        )

    def execute(self, xh, xl):
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * (xl * topdown_wei) + 2 * (xh * bottomup_wei)
        xs = self.post(xs)
        return xs
