# -*- coding: utf-8 -*-
"""在 PyTorch 环境 (pal_torch) 运行: 导出 SCTransNet 迁移所需算子的参考输出。
覆盖: InstanceNorm2d(4D, affine=False) / interpolate(nearest, bilinear-ac True/False)
     / Conv1d / AdaptiveAvgPool2d(1) / LeakyReLU / chunk / 4D matmul
     / var(unbiased=False) / F.normalize(dim=-1) / softmax(dim=3)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = r'C:/Users/Alienware/Documents/kimi/workspace/PAL_jittor/tests/data/op_probe_ref.npz'
rng = np.random.default_rng(42)

d = {}

# 1) InstanceNorm2d 作用于 4D 注意力图 (b,1,c,hw)
x_in = rng.standard_normal((2, 1, 8, 24)).astype(np.float32)
inst = nn.InstanceNorm2d(1)  # torch 默认 affine=False, track_running_stats=False
print('torch InstanceNorm2d 参数数:', len(list(inst.parameters())))  # 期望 0
inst.train()
y_in_train = inst(torch.from_numpy(x_in)).numpy()
inst.eval()
y_in_eval = inst(torch.from_numpy(x_in)).numpy()
d['in_x'] = x_in
d['in_train'] = y_in_train
d['in_eval'] = y_in_eval

# 2) interpolate: nearest (nn.Upsample(scale_factor=2) 默认 mode)
x_n = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
d['nearest_x'] = x_n
d['nearest_y'] = nn.Upsample(scale_factor=2)(torch.from_numpy(x_n)).numpy()

# 3) interpolate bilinear, align_corners=True (深监督 gt2..5 用)
x_b = rng.standard_normal((2, 3, 5, 7)).astype(np.float32)
d['bil_t_x'] = x_b
d['bil_t_y'] = F.interpolate(torch.from_numpy(x_b), scale_factor=2, mode='bilinear',
                             align_corners=True).numpy()
# 4) bilinear align_corners=False (Reconstruct 的 nn.Upsample(mode='bilinear') 默认)
d['bil_f_y'] = nn.Upsample(scale_factor=2, mode='bilinear')(torch.from_numpy(x_b)).numpy()

# 5) Conv1d (eca_layer_2d): bias=False, k=3, padding=1
x_c1 = rng.standard_normal((2, 1, 10)).astype(np.float32)
w_c1 = rng.standard_normal((1, 1, 3)).astype(np.float32)
c1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
c1.weight.data = torch.from_numpy(w_c1)
d['c1_x'] = x_c1
d['c1_w'] = w_c1
d['c1_y'] = c1(torch.from_numpy(x_c1)).numpy()

# 6) AdaptiveAvgPool2d(1)
x_ap = rng.standard_normal((2, 3, 6, 6)).astype(np.float32)
d['ap_x'] = x_ap
d['ap_y'] = nn.AdaptiveAvgPool2d(1)(torch.from_numpy(x_ap)).numpy()

# 7) LeakyReLU (Res_block, 默认 negative_slope=0.01)
x_lr = rng.standard_normal((2, 4)).astype(np.float32)
d['lr_x'] = x_lr
d['lr_y'] = nn.LeakyReLU()(torch.from_numpy(x_lr)).numpy()

# 8) chunk(2, dim=1), 170ch -> 85+85 (FFN project_in 后是偶数, 均分)
x_ck = rng.standard_normal((2, 10, 3, 3)).astype(np.float32)
a_ck, b_ck = torch.from_numpy(x_ck).chunk(2, dim=1)
d['ck_x'] = x_ck
d['ck_a'] = a_ck.numpy()
d['ck_b'] = b_ck.numpy()

# 9) 4D batched matmul: attn = q @ k.transpose(-2,-1); out = attn @ v
q = rng.standard_normal((2, 1, 8, 16)).astype(np.float32)
k = rng.standard_normal((2, 1, 12, 16)).astype(np.float32)
v = rng.standard_normal((2, 1, 12, 24)).astype(np.float32)
attn = torch.from_numpy(q) @ torch.from_numpy(k).transpose(-2, -1)
d['mm_q'] = q
d['mm_k'] = k
d['mm_v'] = v
d['mm_attn'] = attn.numpy()
d['mm_out'] = (attn @ torch.from_numpy(v)).numpy()

# 10) var(unbiased=False) 手写 LayerNorm 用
x_var = rng.standard_normal((2, 5, 7)).astype(np.float32)
d['var_x'] = x_var
d['var_y'] = torch.from_numpy(x_var).var(-1, keepdim=True, unbiased=False).numpy()

# 11) F.normalize(dim=-1)
x_nm = rng.standard_normal((2, 1, 8, 16)).astype(np.float32)
d['nm_x'] = x_nm
d['nm_y'] = F.normalize(torch.from_numpy(x_nm), dim=-1).numpy()

# 12) softmax(dim=3) 于 (b,1,c,c')
x_sm = rng.standard_normal((2, 1, 8, 12)).astype(np.float32)
d['sm_x'] = x_sm
d['sm_y'] = nn.Softmax(dim=3)(torch.from_numpy(x_sm)).numpy()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **d)
print('PROBE_REF_EXPORT_DONE ->', OUT)
