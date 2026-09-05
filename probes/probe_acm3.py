# -*- coding: utf-8 -*-
"""探针3: 绕过 broadcast_to([1]->4D)+add 融合 codegen bug"""
import numpy as np
import jittor as jt
jt.flags.use_cuda = 0

x = jt.randn(2, 1, 8, 8)
b = jt.array([0.5])

def trial(name, fn):
    try:
        y = fn()
        print(name, 'OK, mean=', float(y.mean()))
        return True
    except Exception as e:
        print(name, 'FAIL:', str(e)[:80])
        return False

trial('plain add [1]      ', lambda: x + b)
trial('bias.stop_fuse     ', lambda: x + b.stop_fuse())
trial('x.stop_fuse        ', lambda: x.stop_fuse() + b)
trial('bcast.stop_fuse    ', lambda: x + jt.broadcast_to(b, x.shape).stop_fuse())
trial('conv1x1 bias C=1   ', lambda: jt.nn.Conv2d(4, 1, 1)(jt.randn(2, 4, 8, 8)))
def conv_stopfuse():
    cv = jt.nn.Conv2d(4, 1, 1, bias=False)
    xx = jt.randn(2, 4, 8, 8)
    return cv(xx) + jt.broadcast_to(jt.array([0.3]), (2, 1, 8, 8)).stop_fuse()
trial('conv no-bias + bcast.sf', conv_stopfuse)
print('PROBE3_DONE')
