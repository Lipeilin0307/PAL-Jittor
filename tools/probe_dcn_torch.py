# -*- coding: utf-8 -*-
"""DCN (torchvision deform_conv2d) 参考导出 (pal_torch 环境运行)。

目的:
1) 用 numpy 双假设实现对拍 torchvision, 确定 offset 通道布局真值
   (A: 交错 (y,x,y,x,...); B: 前 kk 通道全 y, 后 kk 通道全 x —— DCN wrapper cat(o1,o2) 的自然语义)
2) 导出 fp64 前向输出 + 对 input/offset/mask/weight/bias 的梯度, 供 jittor 手写实现对拍
   (反向路径正确性的直接证据)

配置取 TTOA 实际用法: k=(1,3) pad=(0,1) 与 k=(3,1) pad=(1,0), stride=1, dilation=1, groups=1。
offset 用 ~N(0,1.5^2) 非平凡随机值, 覆盖越界采样路径。
"""
import os
import numpy as np
import torch
from torchvision.ops import deform_conv2d

OUT = r'C:/Users/Alienware/Documents/kimi/workspace/PAL_jittor/tests/data/probe_dcn.npz'


def deform_np(x, offset, weight, bias, stride, padding, dilation, mask, layout):
    """numpy 参考: groups=1。layout='A' 交错(y,x) / 'B' 前kk全y后kk全x。"""
    b, cin, h, w = x.shape
    cout, _, kh, kw = weight.shape
    sh, sw = stride, stride
    ph, pw = padding
    dh, dw = dilation, dilation
    oh = (h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    ow = (w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    kk = kh * kw
    out = np.zeros((b, cout, oh, ow), dtype=np.float64)

    def sample(bi, ci, y, xx):
        if y <= -1 or y >= h or xx <= -1 or xx >= w:
            return 0.0
        y0, x0 = int(np.floor(y)), int(np.floor(xx))
        y1, x1 = y0 + 1, x0 + 1
        v = 0.0
        for yi, xi, wt in ((y0, x0, (y1 - y) * (x1 - xx)), (y0, x1, (y1 - y) * (xx - x0)),
                           (y1, x0, (y - y0) * (x1 - xx)), (y1, x1, (y - y0) * (xx - x0))):
            if 0 <= yi < h and 0 <= xi < w:
                v += wt * x[bi, ci, yi, xi]
        return v

    for bi in range(b):
        for i in range(oh):
            for j in range(ow):
                for o in range(cout):
                    acc = 0.0
                    for ci in range(cin):
                        for t in range(kk):
                            ky, kx = t // kw, t % kw
                            if layout == 'A':
                                oy = offset[bi, 2 * t, i, j]
                                ox = offset[bi, 2 * t + 1, i, j]
                            else:
                                oy = offset[bi, t, i, j]
                                ox = offset[bi, kk + t, i, j]
                            y = i * sh - ph + ky * dh + oy
                            xx = j * sw - pw + kx * dw + ox
                            acc += weight[o, ci, ky, kx] * mask[bi, t, i, j] * sample(bi, ci, y, xx)
                    out[bi, o, i, j] = acc + bias[o]
    return out


rng = np.random.default_rng(99)
refs = {}
for tag, (kh, kw), (ph, pw) in [('row', (1, 3), (0, 1)), ('col', (3, 1), (1, 0))]:
    b, cin, cout, h, w = 2, 8, 6, 7, 9
    kk = kh * kw
    x = rng.standard_normal((b, cin, h, w))
    offset = rng.standard_normal((b, 2 * kk, h, w)) * 1.5   # 非平凡 offset, 覆盖越界
    mask = rng.uniform(0.05, 0.95, (b, kk, h, w))           # 已 sigmoid 后的 mask 语义
    weight = rng.standard_normal((cout, cin, kh, kw))
    bias = rng.standard_normal((cout,))
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    ot = torch.tensor(offset, dtype=torch.float64, requires_grad=True)
    mt = torch.tensor(mask, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
    out = deform_conv2d(xt, ot, wt, bt, stride=1, padding=(ph, pw), dilation=1, mask=mt)
    out_np = out.detach().numpy()

    # 布局判定
    for layout in ('A', 'B'):
        ref = deform_np(x, offset, weight, bias, 1, (ph, pw), 1, mask, layout)
        print(f'[{tag}] layout {layout}: max diff = {np.abs(ref - out_np).max():.3e}')

    # 梯度
    g_up = rng.standard_normal(out_np.shape)
    out.backward(torch.tensor(g_up, dtype=torch.float64))
    refs.update({f'{tag}_x': x, f'{tag}_offset': offset, f'{tag}_mask': mask,
                 f'{tag}_weight': weight, f'{tag}_bias': bias, f'{tag}_out': out_np,
                 f'{tag}_gup': g_up, f'{tag}_gx': xt.grad.numpy(), f'{tag}_goffset': ot.grad.numpy(),
                 f'{tag}_gmask': mt.grad.numpy(), f'{tag}_gweight': wt.grad.numpy(),
                 f'{tag}_gbias': bt.grad.numpy()})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **refs)
print('PROBE_DCN_DONE ->', OUT)
