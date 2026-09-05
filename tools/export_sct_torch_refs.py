# -*- coding: utf-8 -*-
"""在 PyTorch 环境 (pal_torch) 中运行, 导出 SCTransNet 参考:
1) 固定种子随机初始化的 SCTransNet_No_Sigmoid(mode='train') state_dict -> sct_torch_init.npz
   (无官方权重, 采用"同源权重灌两版、同输入比输出"对齐策略)
2) 固定随机输入 [2,3,256,256] eval 模式前向 -> sct_ref_random.npz (6 个深监督输出, fp32)
3) 同上 fp64 双精度 -> sct_ref_random_fp64.npz (结构等价性决定性判据用)
"""
import os
import sys
import numpy as np

PAL_ROOT = r'C:/Users/Alienware/Documents/kimi/workspace/PAL'
OUT_DIR = r'C:/Users/Alienware/Documents/kimi/workspace/PAL_jittor/tests/data'

sys.path.insert(0, PAL_ROOT)
os.makedirs(OUT_DIR, exist_ok=True)

import torch

# thop 仅被源码 __main__ 使用且 pal_torch 未安装, 注入桩模块避免修改 PAL 原仓库
import types
sys.modules.setdefault('thop', types.SimpleNamespace(profile=lambda *a, **k: (0, 0)))

from model.SCTransNet.SCTransNet_no_sigmoid import SCTransNet_No_Sigmoid

# ---------- 1) 固定种子随机初始化权重 ----------
torch.manual_seed(42)
model = SCTransNet_No_Sigmoid(mode='train')  # 与 train_model.py L609-614 构造一致
model.eval()
sd = model.state_dict()
print('[1] 随机初始化 state_dict keys =', len(sd))
np.savez(os.path.join(OUT_DIR, 'sct_torch_init.npz'),
         **{k: v.numpy().astype(np.float32) for k, v in sd.items()})

# ---------- 2) 固定随机输入前向 (fp32, eval) ----------
rng = np.random.default_rng(1234)
x_rand = rng.standard_normal((2, 3, 256, 256), dtype=np.float32)
with torch.no_grad():
    outs = model(torch.from_numpy(x_rand))
assert isinstance(outs, tuple) and len(outs) == 6, f'期望 6 个深监督输出, 实际 {type(outs)}'
names = ['gt5', 'gt4', 'gt3', 'gt2', 'd0', 'out']
print('[2] fp32 输出形状:', [tuple(o.shape) for o in outs])
np.savez(os.path.join(OUT_DIR, 'sct_ref_random.npz'), x=x_rand,
         **{f'y{i}_{n}': o.numpy() for i, (n, o) in enumerate(zip(names, outs))})

# ---------- 3) fp64 结构等价性参考 ----------
md = SCTransNet_No_Sigmoid(mode='train').double()
md.load_state_dict(sd)
md.eval()
with torch.no_grad():
    outs64 = md(torch.from_numpy(x_rand).double())
np.savez(os.path.join(OUT_DIR, 'sct_ref_random_fp64.npz'), x=x_rand,
         **{f'y{i}_{n}': o.numpy() for i, (n, o) in enumerate(zip(names, outs64))})
print('[3] fp64 参考已存; torch fp32 vs fp64 各输出 max diff:')
for i, n in enumerate(names):
    d = float(np.abs(outs[i].numpy().astype(np.float64) - outs64[i].numpy()).max())
    print(f'      {n}: {d:.3e}')
print('EXPORT_DONE')
