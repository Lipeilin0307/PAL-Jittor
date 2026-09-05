# -*- coding: utf-8 -*-
"""ACM 迁移专项探针：BN momentum 语义 / interpolate 双线性 / ConvTranspose / sort / AdaptiveAvgPool / GroupNorm"""
import sys
import numpy as np

import jittor as jt
jt.flags.use_cuda = 0
from jittor import nn

print('JT', jt.__version__, 'cuda=', jt.flags.use_cuda)

# ---------- 1) BatchNorm momentum 语义 ----------
# torch: running = (1-m)*running + m*batch_stat  (m=0.9 => 新统计占 0.9)
# jittor 待实测：构造 BN(num_features=1, momentum=?)，置 running=0，喂固定 batch，看更新后 running
bn = nn.BatchNorm2d(1, momentum=0.9)
bn.train()
# jittor BN 参数名
print('BN state keys:', list(bn.state_dict().keys()))
x = jt.zeros((4, 1, 2, 2)) + 10.0  # 均值 10，方差 0
bn(x)
rm = bn.running_mean.numpy().item()
print('BN momentum=0.9 after one batch(mean=10): running_mean =', rm)
# 若 running_mean≈9 -> torch 语义(新统计权重0.9)；若≈1 -> 反向语义(新统计权重0.1)
bn2 = nn.BatchNorm2d(1, momentum=0.1)
bn2.train()
bn2(x)
print('BN momentum=0.1 after one batch(mean=10): running_mean =', bn2.running_mean.numpy().item())

# eval 模式：用 running stats + affine
bn3 = nn.BatchNorm2d(2, momentum=0.9)
bn3.running_mean = jt.array([1.0, -2.0])
bn3.running_var = jt.array([4.0, 9.0])
bn3.weight = jt.array([2.0, 3.0])
bn3.bias = jt.array([0.5, -0.5])
bn3.eval()
xx = jt.array(np.random.randn(2, 2, 3, 3).astype(np.float32))
yy = bn3(xx)
expect = (xx.numpy() - np.array([1.0, -2.0])[None, :, None, None]) / np.sqrt(np.array([4.0, 9.0])[None, :, None, None] + 1e-5)
expect = expect * np.array([2.0, 3.0])[None, :, None, None] + np.array([0.5, -0.5])[None, :, None, None]
print('BN eval max diff vs manual:', float(np.abs(yy.numpy() - expect).max()))

# ---------- 2) interpolate bilinear align_corners=False ----------
a = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)
up = nn.interpolate(jt.array(a), scale_factor=4, mode='bilinear')
print('interp scale4 shape:', up.shape)
print('interp[0,0,:8,:8]:\n', np.round(up.numpy()[0, 0, :8, :8], 4))
# torch 对照值（align_corners=False）手算几个点：
# torch F.interpolate(scale_factor=4, mode='bilinear', align_corners=False) 的 (0,0)=0, (0,1)=0.15625*...? 直接对比 torch 环境更可靠，此处只记录

# ---------- 3) ConvTranspose2d 输出尺寸 ----------
ct = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
inp = jt.randn(2, 64, 30, 30)
print('ConvTranspose 30->', ct(inp).shape)  # 期望 (2,32,60,60)

# ---------- 4) sort ----------
v = jt.array([3.0, 1.0, 2.0, 5.0, 4.0])
sv, idx = jt.sort(v)
print('sort:', sv.numpy(), 'has argsort:', hasattr(jt, 'argsort'))

# ---------- 5) AdaptiveAvgPool2d(1) ----------
ap = nn.AdaptiveAvgPool2d(1)
t = jt.array(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
print('adaptiveavgpool:', ap(t).numpy().ravel(), 'expect', t.numpy().mean())

# ---------- 6) GroupNorm ----------
gn = nn.GroupNorm(1, 8)
t6 = jt.randn(2, 8, 5, 5)
o6 = gn(t6)
m = t6.numpy().mean(); s = t6.numpy().var()
exp6 = (t6.numpy() - m) / np.sqrt(s + 1e-5)
print('GroupNorm(1) max diff:', float(np.abs(o6.numpy() - exp6).max()))

# ---------- 7) BCE with logits ----------
logits = jt.array([0.5, -1.0, 2.0])
target = jt.array([1.0, 0.0, 1.0])
bce = nn.binary_cross_entropy_with_logits(logits, target, reduction='none')
exp = np.maximum(logits.numpy(), 0) - logits.numpy() * target.numpy() + np.log(1 + np.exp(-np.abs(logits.numpy())))
print('bce_with_logits diff:', float(np.abs(bce.numpy() - exp).max()), 'keys ok')

# ---------- 8) MaxPool2d(3,2,1) ----------
mp = nn.MaxPool2d(3, 2, 1)
t8 = jt.randn(1, 1, 64, 64)
print('maxpool 64->', mp(t8).shape)  # 期望 32

# ---------- 9) Dropout eval 恒等 ----------
dp = nn.Dropout(0.1)
dp.eval()
print('dropout eval identity:', bool(np.allclose(dp(jt.ones(4)).numpy(), 1.0)))
print('PROBE_DONE')
