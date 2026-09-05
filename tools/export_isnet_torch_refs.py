# -*- coding: utf-8 -*-
"""在 PyTorch 环境 (pal_torch) 中运行, 导出 ISNet 参考:
1) 固定种子随机初始化 ISNet_No_Sigmoid state_dict -> isnet_torch_init.npz
   (无官方权重参与, 采用"同源权重灌两版、同输入比输出"对齐策略)
   注意: 构造后把 4 个 DCN 的 conv_offset_mask 从零初始改为固定种子非平凡随机值 —
   源码 init_offset() 将其置零, 若保持全零则 offset 恒 0、双线性采样路径测不到;
   验收要求 DCN 用非平凡 offset。该改动在导出 state_dict 之前, 两版加载同一套值。
2) 固定随机输入 [2,3,256,256] eval 前向 -> isnet_ref_random.npz (out, edge_out; fp32)
3) 同上 fp64 -> isnet_ref_random_fp64.npz (结构等价性决定性判据)
"""
import os
import sys
import numpy as np

PAL_ROOT = r'C:/Users/Alienware/Documents/kimi/workspace/PAL'
OUT_DIR = r'C:/Users/Alienware/Documents/kimi/workspace/PAL_jittor/tests/data'

sys.path.insert(0, PAL_ROOT)
os.makedirs(OUT_DIR, exist_ok=True)

import torch
from model.ISNet.ISNet_no_sigmoid import ISNet_No_Sigmoid
from model.ISNet.dcn_v2 import DCN

# ---------- 1) 固定种子随机初始化权重 (+ DCN offset 非平凡化) ----------
torch.manual_seed(42)
model = ISNet_No_Sigmoid()
torch.manual_seed(2024)
n_dcn = 0
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, DCN):
            m.conv_offset_mask.weight.normal_(0, 0.5)
            m.conv_offset_mask.bias.normal_(0, 0.5)
            n_dcn += 1
print(f'[1] DCN conv_offset_mask 非平凡化 x {n_dcn}')
model.eval()
sd = model.state_dict()
print('[1] 随机初始化 state_dict keys =', len(sd))
np.savez(os.path.join(OUT_DIR, 'isnet_torch_init.npz'),
         **{k: v.numpy().astype(np.float32) for k, v in sd.items()})

# ---------- 2) 固定随机输入前向 (fp32, eval) ----------
rng = np.random.default_rng(1234)
x_rand = rng.standard_normal((2, 3, 256, 256), dtype=np.float32)
with torch.no_grad():
    out, edge = model(torch.from_numpy(x_rand))
print('[2] fp32 输出:', out.shape, edge.shape,
      'edge range [%.3f, %.3f]' % (edge.min(), edge.max()))
np.savez(os.path.join(OUT_DIR, 'isnet_ref_random.npz'),
         x=x_rand, out=out.numpy(), edge=edge.numpy())

# ---------- 3) fp64 参考 ----------
md = ISNet_No_Sigmoid().double()
with torch.no_grad():
    torch.manual_seed(2024)
    for m in md.modules():
        if isinstance(m, DCN):
            m.conv_offset_mask.weight.normal_(0, 0.5)
            m.conv_offset_mask.bias.normal_(0, 0.5)
md.load_state_dict(sd)   # 直接载入同一 sd 即可 (double() 已转 fp64)
md.eval()
with torch.no_grad():
    out64, edge64 = md(torch.from_numpy(x_rand).double())
np.savez(os.path.join(OUT_DIR, 'isnet_ref_random_fp64.npz'),
         x=x_rand, out=out64.numpy(), edge=edge64.numpy())
print('[3] fp64 参考已存; torch fp32 vs fp64 max diff: out=%.3e edge=%.3e' % (
    float(np.abs(out.numpy().astype(np.float64) - out64.numpy()).max()),
    float(np.abs(edge.numpy().astype(np.float64) - edge64.numpy()).max())))

# ---------- 4) 小输入 fp64 参考 (1,3,64,64): jittor CPU fp64 全尺寸前向过慢,
#    小输入覆盖全部模块与代码路径 (DCN/gate/interpolate 均触发), 结构判定力相同 ----------
x_small = x_rand[:1, :, :64, :64].copy()
with torch.no_grad():
    outs, edges = md(torch.from_numpy(x_small).double())
np.savez(os.path.join(OUT_DIR, 'isnet_ref_small_fp64.npz'),
         x=x_small, out=outs.numpy(), edge=edges.numpy())
print('[4] 小输入 fp64 参考已存:', outs.shape, edges.shape)
print('EXPORT_DONE')
