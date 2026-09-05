# -*- coding: utf-8 -*-
"""SCTransNet 迁移前置探针 (在 pal_torch 环境运行): 生成各算子参考输出 npz。
供 jittor 侧 probe_sct_jittor.py 逐点核对。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = r'C:/Users/Alienware/Documents/kimi/workspace/PAL_jittor/tests/data/probe_sct.npz'
rng = np.random.default_rng(7)
refs = {}

# 1) InstanceNorm2d(num_features=1, 默认 affine=False, track_running_stats=False)
#    作用于 4D 注意力图 (b,1,c,hw)
x_in = rng.standard_normal((2, 1, 5, 7)).astype(np.float32)
in_mod = nn.InstanceNorm2d(1).eval()
with torch.no_grad():
    refs['in_x'], refs['in_y'] = x_in, in_mod(torch.from_numpy(x_in)).numpy()

# 2) interpolate bilinear align_corners=True (深监督上采样, 网络内实际形状 (b,1,16,16)->scale16)
x_up = rng.standard_normal((2, 1, 16, 16)).astype(np.float32)
refs['up_x'] = x_up
refs['up16_bilinear_ac'] = F.interpolate(torch.from_numpy(x_up), scale_factor=16,
                                         mode='bilinear', align_corners=True).numpy()
# 3) nn.Upsample(scale_factor=2) 默认 mode='nearest' (UpBlock_attention.up)
x_near = rng.standard_normal((2, 3, 7, 5)).astype(np.float32)
refs['near_x'] = x_near
refs['near_y'] = nn.Upsample(scale_factor=2)(torch.from_numpy(x_near)).numpy()

# 4) Conv1d(1,1,k=3,padding=1,bias=False) (eca_layer_2d)
x_c1 = rng.standard_normal((2, 1, 64)).astype(np.float32)
conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
w1d = rng.standard_normal((1, 1, 3)).astype(np.float32)
with torch.no_grad():
    conv1d.weight.copy_(torch.from_numpy(w1d))
    refs['c1d_x'], refs['c1d_w'], refs['c1d_y'] = x_c1, w1d, conv1d(torch.from_numpy(x_c1)).numpy()

# 5) Softmax(dim=3) 作用于 attn (b,1,c,hw)
x_sm = rng.standard_normal((2, 1, 5, 7)).astype(np.float32)
refs['sm_x'] = x_sm
refs['sm_y'] = nn.Softmax(dim=3)(torch.from_numpy(x_sm)).numpy()

# 6) F.normalize(dim=-1) 沿 hw 维
x_nm = rng.standard_normal((2, 1, 4, 256)).astype(np.float32)
refs['nm_x'] = x_nm
refs['nm_y'] = F.normalize(torch.from_numpy(x_nm), dim=-1).numpy()

# 7) 手写 LayerNorm (WithBias: mu + var(unbiased=False), eps=1e-5) 沿最后一维
x_ln = rng.standard_normal((2, 256, 32)).astype(np.float32)
w_ln = rng.standard_normal((32,)).astype(np.float32)
b_ln = rng.standard_normal((32,)).astype(np.float32)
t = torch.from_numpy(x_ln)
mu = t.mean(-1, keepdim=True)
sigma = t.var(-1, keepdim=True, unbiased=False)
refs['ln_x'], refs['ln_w'], refs['ln_b'] = x_ln, w_ln, b_ln
refs['ln_y'] = ((t - mu) / torch.sqrt(sigma + 1e-5) * torch.from_numpy(w_ln) + torch.from_numpy(b_ln)).numpy()

# 8) matmul: (b,1,c1,hw) @ (b,1,c,hw).transpose(-2,-1)
a = rng.standard_normal((2, 1, 32, 256)).astype(np.float32)
b = rng.standard_normal((2, 1, 480, 256)).astype(np.float32)
refs['mm_a'], refs['mm_b'] = a, b
refs['mm_y'] = (torch.from_numpy(a) @ torch.from_numpy(b).transpose(-2, -1)).numpy()

# 9) chunk(2, dim=1)
x_ck = rng.standard_normal((2, 170, 8, 8)).astype(np.float32)
c1, c2 = torch.from_numpy(x_ck).chunk(2, dim=1)
refs['ck_x'], refs['ck_1'], refs['ck_2'] = x_ck, c1.numpy(), c2.numpy()

# 10) AdaptiveAvgPool2d(1) 与 F.avg_pool2d 全窗 (CCA)
x_ap = rng.standard_normal((2, 6, 7, 5)).astype(np.float32)
refs['ap_x'] = x_ap
refs['ap_adaptive'] = nn.AdaptiveAvgPool2d(1)(torch.from_numpy(x_ap)).numpy()
refs['ap_full'] = F.avg_pool2d(torch.from_numpy(x_ap), (7, 5), stride=(7, 5)).numpy()

# 11) Linear (CCA mlp)
x_li = rng.standard_normal((2, 6)).astype(np.float32)
lin = nn.Linear(6, 6)
refs['li_x'] = x_li
with torch.no_grad():
    refs['li_y'] = lin(torch.from_numpy(x_li)).numpy()
    refs['li_w'], refs['li_b'] = lin.weight.numpy(), lin.bias.numpy()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **refs)
print('PROBE_TORCH_DONE ->', OUT)
