# -*- coding: utf-8 -*-
"""探针2: GroupNorm 逐样本核对 + 手写 BCE-with-logits + CUDA 冒烟"""
import numpy as np
import jittor as jt
from jittor import nn

# ---------- GroupNorm 逐样本核对 ----------
t6 = np.random.randn(2, 8, 5, 5).astype(np.float32)
gn = nn.GroupNorm(1, 8)
o6 = gn(jt.array(t6)).numpy()
m = t6.reshape(2, -1).mean(axis=1, keepdims=True)
v = t6.reshape(2, -1).var(axis=1, keepdims=True)
exp6 = ((t6.reshape(2, -1) - m) / np.sqrt(v + 1e-5)).reshape(2, 8, 5, 5)
print('GroupNorm(1) per-sample max diff:', float(np.abs(o6 - exp6).max()))

gn4 = nn.GroupNorm(4, 8)
o6b = gn4(jt.array(t6)).numpy()
t6g = t6.reshape(2, 4, 2, 5, 5)
mg = t6g.reshape(2, 4, -1).mean(axis=2, keepdims=True)
vg = t6g.reshape(2, 4, -1).var(axis=2, keepdims=True)
exp6b = ((t6g.reshape(2, 4, -1) - mg) / np.sqrt(vg + 1e-5)).reshape(2, 8, 5, 5)
print('GroupNorm(4) per-sample max diff:', float(np.abs(o6b - exp6b).max()))

# ---------- 手写 BCE with logits ----------
logits_np = np.array([0.5, -1.0, 2.0], dtype=np.float32)
target_np = np.array([1.0, 0.0, 1.0], dtype=np.float32)
logits = jt.array(logits_np); target = jt.array(target_np)
bce = jt.maximum(logits, 0) - logits * target + jt.log(1 + jt.exp(-jt.abs(logits)))
exp = np.maximum(logits_np, 0) - logits_np * target_np + np.log(1 + np.exp(-np.abs(logits_np)))
print('manual bce_with_logits diff:', float(np.abs(bce.numpy() - exp).max()))
# jittor 内置签名探测
try:
    b2 = nn.binary_cross_entropy_with_logits(logits, target)
    print('builtin bce (no reduction kw) diff:', float(np.abs(b2.numpy() - exp).max()))
except Exception as e:
    print('builtin bce err:', repr(e))

# ---------- CUDA 冒烟 ----------
jt.flags.use_cuda = 1
a = jt.array([1.0, 2.0, 3.0])
print('cuda add:', (a + 1).numpy())
cv = nn.Conv2d(3, 8, 3, padding=1)
x = jt.randn(2, 3, 32, 32)
print('cuda conv:', cv(x).shape)
ct = nn.ConvTranspose2d(8, 4, 4, 2, 1)
print('cuda convT:', ct(cv(x)).shape)
bn = nn.BatchNorm2d(8); bn.eval()
print('cuda bn:', bn(cv(x)).shape)
up = nn.interpolate(x, scale_factor=4, mode='bilinear')
print('cuda interp:', up.shape)
gn2 = nn.GroupNorm(1, 8)
print('cuda gn diff:', float(np.abs(gn2(cv(x)).numpy().mean() - 0)) < 1)
print('PROBE2_DONE')
