# -*- coding: utf-8 -*-
"""ISNet Jittor 版验收测试 (在 jittor 环境运行):

[1] 形状/契约测试: 随机输入 [2,3,256,256] 前向返回 (out, edge_out), 均 [2,1,256,256];
    out 为无 sigmoid logits, edge_out 内部已过 sigmoid (值域 [0,1])。
[2] 同源权重对齐: torch 随机初始化 state_dict (isnet_torch_init.npz, DCN offset 已非平凡化)
    经 convert_isnet_weights 灌入 jittor 版 (键名 1:1 全覆盖, 含 DCN offset 层 /
    GatedSpatialConv / register_buffer / 死模块), 与 PyTorch 参考输出对比 (eval 模式):
    a) fp64 双精度结构等价性 —— out/edge_out max abs diff < 1e-6 (结构决定性判据)
    b) fp32 logit 级 max abs diff 如实报告 + sigmoid 决策级 diff < 1e-4,
       二值化不一致像素比例要求为 0 (out 与 edge_out 两路都查)
[3] train 模式前向可反向: 1x3x64x64 输入, loss=out.mean()+edge.mean(),
    梯度范数有限非零 (反向覆盖 DCN reindex gather 路径)。
"""
import os
import sys

import numpy as np
import jittor as jt

jt.flags.use_cuda = 0  # CPU 对齐: 规避 CUDA matmul TF32 污染 fp32 对拍

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'tests', 'data')

from model.isnet import ISNet_No_Sigmoid
from convert_isnet_weights import convert

failures = []

# ---------- [1] 形状/契约测试 ----------
model = ISNet_No_Sigmoid()
model.eval()
x = jt.array(np.random.randn(2, 3, 256, 256).astype(np.float32))
out, edge = model(x)
ok = tuple(out.shape) == (2, 1, 256, 256) and tuple(edge.shape) == (2, 1, 256, 256)
emin, emax = float(edge.min()), float(edge.max())
ok = ok and 0.0 <= emin and emax <= 1.0
print(f'[test_isnet/1] (2,3,256,256) -> out {tuple(out.shape)} / edge_out {tuple(edge.shape)}, '
      f'edge 值域 [{emin:.3f},{emax:.3f}] (应 [0,1], 内部已 sigmoid): {"PASS" if ok else "FAIL"}')
if not ok:
    failures.append('shape')

# ---------- [2] 同源权重灌入 + 对齐 ----------
npz = os.path.join(DATA, 'isnet_torch_init.npz')
model_c = convert(npz, out_path=os.path.join(ROOT, 'work_dirs', 'isnet_torch_init_jt.pkl'),
                  verbose=False)
model_c.eval()

sigmoid = lambda a: 1.0 / (1.0 + np.exp(-a))
ref = np.load(os.path.join(DATA, 'isnet_ref_random.npz'))
xr = ref['x']
with jt.no_grad():
    out_j, edge_j = model_c(jt.array(xr))
    out_j, edge_j = out_j.numpy(), edge_j.numpy()

print('-' * 60)
print('[test_isnet/2a] fp32 对齐 (torch CPU 参考, eval):')
# out 分支: logits, 决策经 sigmoid
d_logit = float(np.abs(out_j - ref['out']).max())
d_sig = float(np.abs(sigmoid(out_j) - sigmoid(ref['out'])).max())
mis = float(np.mean((sigmoid(out_j) > 0.5) != (sigmoid(ref['out']) > 0.5)))
ok1 = d_sig < 1e-4 and mis == 0
print(f'  [out]  logit max abs diff = {d_logit:.3e} | sigmoid diff = {d_sig:.3e} (<1e-4) '
      f'| 二值化不一致 = {mis:.2e} (要求0): {"PASS" if ok1 else "FAIL"}')
# edge_out 分支: 已是概率
d_edge = float(np.abs(edge_j - ref['edge']).max())
mis2 = float(np.mean((edge_j > 0.5) != (ref['edge'] > 0.5)))
ok2 = d_edge < 1e-4 and mis2 == 0
print(f'  [edge] prob max abs diff = {d_edge:.3e} (<1e-4) '
      f'| 二值化不一致 = {mis2:.2e} (要求0): {"PASS" if ok2 else "FAIL"}')
if not ok1:
    failures.append('fp32_out')
if not ok2:
    failures.append('fp32_edge')

# ---------- fp64 结构等价性 ----------
# 注: 用 (1,3,64,64) 小输入做 fp64 判定 —— jittor CPU fp64 全尺寸(256)前向超 300s
# 不可用; 64x64 覆盖全部模块与代码路径 (DCN/gate/interpolate 均触发), 结构判定力相同;
# 全尺寸路径由 [2a] fp32 对齐覆盖。且 DCN 权重累加用广播乘+sum (非 matmul),
# 不依赖 cublas (cublasGemmEx 不支持 fp64)。
print('-' * 60)
model64 = convert(npz, out_path=None, verbose=False)
model64.float64()
model64.eval()
ref64 = np.load(os.path.join(DATA, 'isnet_ref_small_fp64.npz'))
with jt.no_grad():
    # 注意: jt.array 默认把 float64 numpy 降级为 float32, 必须显式 dtype='float64'
    out64, edge64 = model64(jt.array(ref64['x'], dtype='float64'))
print('[test_isnet/2b] fp64 结构等价性 (输入 1x3x64x64, 阈值 1e-6):')
for n, yj, yr in [('out', out64, ref64['out']), ('edge_out', edge64, ref64['edge'])]:
    d64 = float(np.abs(yj.numpy() - yr).max())
    ok = d64 < 1e-6
    print(f'  [{n}] max abs diff = {d64:.3e}: {"PASS" if ok else "FAIL"}')
    if not ok:
        failures.append(f'fp64_{n}')

# ---------- [3] train 模式可反向 ----------
print('-' * 60)
model_g = convert(npz, out_path=None, verbose=False)
model_g.train()
xs = jt.array(xr[:1, :, :64, :64].copy())
out_g, edge_g = model_g(xs)
loss = out_g.mean() + edge_g.mean()
grads = jt.grad(loss, model_g.parameters())
# 逐个数梯度会重复执行反向子图; 拼成单向量一次 sync
flat = jt.concat([g.reshape(-1) for g in grads], dim=0)
gnorm = float(jt.sqrt((flat * flat).sum()))
ok = np.isfinite(gnorm) and gnorm > 0
print(f'[test_isnet/3] train 模式前向+反向 (输入 1x3x64x64): loss = {float(loss):.6f}, '
      f'全局梯度范数 = {gnorm:.6e} (有限非零): {"PASS" if ok else "FAIL"}')
if not ok:
    failures.append('backward')

print('=' * 60)
if failures:
    print('test_isnet 未通过:', failures)
    sys.exit(1)
print('test_isnet 全部通过 ✓')
