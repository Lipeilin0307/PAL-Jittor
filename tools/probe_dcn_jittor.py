# -*- coding: utf-8 -*-
"""jittor 手写 deform_conv2d 对拍 (jittor 环境运行)。

对拍 probe_dcn.npz (torchvision fp64 参考, 含梯度)。
实现要点:
- 逐 tap 计算采样坐标 (base grid + offset, 布局 A: 通道 2t=y, 2t+1=x)
- floor 取四角, 双线性加权, 角点出界贡献 0 (与 torchvision 一致, 不做权重归一化)
- gather 用 x.reindex([b,c,y,x]) (索引 clamp 到界内 + 有效性掩码)
- 权重累加用 reshape 后的 matmul; mask 逐 tap 乘
"""
import os
import sys
import numpy as np
import jittor as jt

jt.flags.use_cuda = 0  # CPU + fp64 对拍

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.isnet import deform_conv2d as deform_conv2d_jt  # 与被测实现同源

D = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'tests', 'data', 'probe_dcn.npz'))




fails = []
for tag, (kh, kw), (ph, pw) in [('row', (1, 3), (0, 1)), ('col', (3, 1), (1, 0))]:
    # 注意: jt.array 默认把 float64 numpy 降级为 float32, 必须显式 dtype='float64'
    x = jt.array(D[f'{tag}_x'], dtype='float64')
    offset = jt.array(D[f'{tag}_offset'], dtype='float64')
    mask = jt.array(D[f'{tag}_mask'], dtype='float64')
    weight = jt.array(D[f'{tag}_weight'], dtype='float64')
    bias = jt.array(D[f'{tag}_bias'], dtype='float64')
    out = deform_conv2d_jt(x, offset, weight, bias, 1, (ph, pw), 1, mask)
    d_out = float(np.abs(out.numpy() - D[f'{tag}_out']).max())
    print(f'[{tag}] forward fp64 max diff = {d_out:.3e}', 'PASS' if d_out < 1e-9 else 'FAIL')
    if d_out >= 1e-9:
        fails.append(f'{tag}_fwd')

    # 反向对拍
    g_up = jt.array(D[f'{tag}_gup'], dtype='float64')
    loss = (out * g_up).sum()
    gx, go, gm, gw, gb = jt.grad(loss, [x, offset, mask, weight, bias])
    for name, gj, key in [('gx', gx, 'gx'), ('goffset', go, 'goffset'),
                          ('gmask', gm, 'gmask'), ('gweight', gw, 'gweight'),
                          ('gbias', gb, 'gbias')]:
        d = float(np.abs(gj.numpy() - D[f'{tag}_{key}']).max())
        ok = d < 1e-9
        print(f'[{tag}] grad {name} max diff = {d:.3e}', 'PASS' if ok else 'FAIL')
        if not ok:
            fails.append(f'{tag}_{name}')

print('=' * 50)
if fails:
    print('DCN 探针未过:', fails)
    sys.exit(1)
print('DCN 手写实现前向+反向全部对拍通过 ✓')
