# -*- coding: utf-8 -*-
"""探针5: conv(4->1)+bias 后接 interpolate 的融合复现"""
import numpy as np
import jittor as jt
jt.flags.use_cuda = 0
from jittor import nn

def trial(name, fn):
    try:
        y = fn()
        print(name, 'OK, shape=', y.shape, 'mean=', float(y.mean()))
    except Exception as e:
        print(name, 'FAIL:', str(e)[:60])

def head_interp():
    blk = nn.Sequential(
        nn.Conv2d(16, 4, 3, padding=1, bias=False),
        nn.BatchNorm2d(4, momentum=0.9),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(4, 1, 1)
    )
    blk.eval()
    out = blk(jt.randn(2, 16, 64, 64))
    return nn.interpolate(out, scale_factor=4, mode='bilinear')
trial('head+interpolate eval      ', head_interp)

def head_interp_ng():
    blk = nn.Sequential(
        nn.Conv2d(16, 4, 3, padding=1, bias=False),
        nn.BatchNorm2d(4, momentum=0.9),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(4, 1, 1)
    )
    blk.eval()
    with jt.no_grad():
        out = blk(jt.randn(2, 16, 64, 64))
        return nn.interpolate(out, scale_factor=4, mode='bilinear')
trial('head+interpolate no_grad   ', head_interp_ng)
print('PROBE5_DONE')
