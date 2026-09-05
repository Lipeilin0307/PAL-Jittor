# -*- coding: utf-8 -*-
"""SCTransNet 迁移前置探针 (在 jittor 环境运行): 逐点核对 jittor 算子与 torch 参考。"""
import os
import sys
import numpy as np
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 1

D = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'tests', 'data', 'probe_sct.npz'))
fails = []

def check(name, yj, yt, tol=2e-6):
    d = float(np.abs(np.asarray(yj) - yt).max())
    ok = d < tol
    print(f'  {name}: max abs diff = {d:.3e} {"PASS" if ok else "FAIL"}')
    if not ok:
        fails.append(name)

# 1) InstanceNorm2d(1, affine=False) eval 模式
m = nn.InstanceNorm2d(1, affine=False)
m.eval()
print('IN state_dict keys:', list(m.state_dict().keys()))
check('InstanceNorm2d', m(jt.array(D['in_x'])).numpy(), D['in_y'])
# 也确认 train 模式下一致 (torch track_running_stats=False 时 train/eval 同行为)
m.train()
check('InstanceNorm2d(train)', m(jt.array(D['in_x'])).numpy(), D['in_y'])

# 2) interpolate bilinear align_corners=True, scale 16
check('interp bilinear ac=True',
      nn.interpolate(jt.array(D['up_x']), scale_factor=16, mode='bilinear',
                     align_corners=True).numpy(), D['up16_bilinear_ac'], tol=1e-5)

# 3) nearest upsample x2 (nn.Upsample 默认)
up = nn.Upsample(scale_factor=2)
yj = up(jt.array(D['near_x'])).numpy()
check('Upsample nearest x2', yj, D['near_y'])

# 4) Conv1d
c = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
c.weight.update(jt.array(D['c1d_w']))
check('Conv1d', c(jt.array(D['c1d_x'])).numpy(), D['c1d_y'], tol=1e-5)

# 5) Softmax(dim=3)
check('Softmax dim=3', nn.Softmax(dim=3)(jt.array(D['sm_x'])).numpy(), D['sm_y'], tol=1e-6)

# 6) normalize dim=-1 手写: x / max(||x||2, eps)
x = jt.array(D['nm_x'])
norm = jt.sqrt((x * x).sum(-1, keepdims=True))
yj = (x / jt.clamp(norm, min_v=1e-12)).numpy()
check('normalize(dim=-1)', yj, D['nm_y'], tol=1e-6)

# 7) 手写 LayerNorm (mean + biased var)
x = jt.array(D['ln_x'])
mu = x.mean(-1, keepdims=True)
xc = x - mu
sigma = (xc * xc).mean(-1, keepdims=True)
yj = (xc / jt.sqrt(sigma + 1e-5) * jt.array(D['ln_w']) + jt.array(D['ln_b'])).numpy()
check('LayerNorm manual', yj, D['ln_y'], tol=1e-5)

# 8) matmul + transpose 最后两维
a, b = jt.array(D['mm_a']), jt.array(D['mm_b'])
check('matmul @ b.mT', jt.matmul(a, b.transpose(-2, -1)).numpy(), D['mm_y'], tol=1e-4)

# 9) chunk
c1, c2 = jt.array(D['ck_x']).chunk(2, dim=1)
check('chunk part1', c1.numpy(), D['ck_1'])
check('chunk part2', c2.numpy(), D['ck_2'])

# 10) AdaptiveAvgPool2d(1) / mean 替代
x = jt.array(D['ap_x'])
check('AdaptiveAvgPool2d(1)', nn.AdaptiveAvgPool2d(1)(x).numpy(), D['ap_adaptive'])
check('mean(dims=(2,3)) 替代全窗 avgpool', x.mean(dims=(2, 3), keepdims=True).numpy(), D['ap_full'])

# 11) Linear
lin = nn.Linear(6, 6)
lin.weight.update(jt.array(D['li_w']))
lin.bias.update(jt.array(D['li_b']))
check('Linear', lin(jt.array(D['li_x'])).numpy(), D['li_y'], tol=1e-5)

print('=' * 50)
if fails:
    print('探针未过:', fails)
    sys.exit(1)
print('全部探针通过 ✓')
