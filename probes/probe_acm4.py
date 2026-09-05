# -*- coding: utf-8 -*-
"""探针4: 精确复现 head.block.4.bias 融合失败"""
import numpy as np
import jittor as jt
jt.flags.use_cuda = 0
from jittor import nn

def trial(name, fn):
    try:
        y = fn()
        print(name, 'OK, mean=', float(y.mean()))
    except Exception as e:
        print(name, 'FAIL:', str(e)[:60])

# 1) 更大尺寸的 C=1 bias
trial('conv 4->1 k1 @64x64       ', lambda: nn.Conv2d(4, 1, 1)(jt.randn(2, 4, 64, 64)))

# 2) 完整 head block 链
def head_chain():
    blk = nn.Sequential(
        nn.Conv2d(16, 4, 3, padding=1, bias=False),
        nn.BatchNorm2d(4, momentum=0.9),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(4, 1, 1)
    )
    blk.eval()
    return blk(jt.randn(2, 16, 64, 64))
trial('完整 head 链 @64x64 eval   ', head_chain)

# 3) no_grad 上下文
def head_chain_ng():
    blk = nn.Sequential(
        nn.Conv2d(16, 4, 3, padding=1, bias=False),
        nn.BatchNorm2d(4, momentum=0.9),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(4, 1, 1)
    )
    blk.eval()
    with jt.no_grad():
        return blk(jt.randn(2, 16, 64, 64))
trial('完整 head 链 no_grad       ', head_chain_ng)

# 4) Sequential 内单独 conv(4->1) 接在 dropout 后
def after_dropout():
    dp = nn.Dropout(0.1); dp.eval()
    cv = nn.Conv2d(4, 1, 1)
    return cv(dp(jt.randn(2, 4, 64, 64)))
trial('dropout->conv(4->1)       ', after_dropout)
print('PROBE4_DONE')
