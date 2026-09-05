# -*- coding: utf-8 -*-
"""SCTransNet Jittor 版验收测试 (在 jittor 环境运行):

[1] 形状/契约测试: 随机输入 [2,3,256,256], mode='train' 前向应返回 6 个输出
    (gt5, gt4, gt3, gt2, d0, out), 形状均 [2,1,256,256]; mode='test' 应仅返回 out。
[2] 同源权重对齐: torch 随机初始化 state_dict (sct_torch_init.npz) 经
    convert_sct_weights 灌入 jittor 版 (键名 1:1, 全覆盖校验, 含 LayerNorm/
    InstanceNorm/Conv1d/死参数), 与 PyTorch 参考输出对比 (eval 模式):
    a) fp64 双精度结构等价性 —— 严格门限 6 个输出逐一 max abs diff < 1e-6
       (结构决定性判据: fp64 下浮点累加噪声消失, 结构不一致此处必暴露)
    b) fp32 logit 级 max abs diff 如实报告 + sigmoid 决策级 diff < 1e-4,
       二值化不一致像素比例要求为 0
[3] train 模式前向可反向: 6 输出各取 mean 求和作 loss, 反向后梯度范数有限非零。
"""
import os
import sys

import numpy as np
import jittor as jt

jt.flags.use_cuda = 0  # CPU 对齐: 规避本机 CUDA 下 jittor matmul 走 TF32 引入的 ~2e-2 误差

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'tests', 'data')

from model.sct import SCTransNet_No_Sigmoid
from model.convert_sct_weights import convert

failures = []
NAMES = ['gt5', 'gt4', 'gt3', 'gt2', 'd0', 'out']

# ---------- [1] 形状/契约测试 ----------
model = SCTransNet_No_Sigmoid(mode='train')
model.eval()
x = jt.array(np.random.randn(2, 3, 256, 256).astype(np.float32))
outs = model(x)
ok = isinstance(outs, tuple) and len(outs) == 6 and all(
    tuple(o.shape) == (2, 1, 256, 256) for o in outs)
print(f'[test_sct/1] mode=train 随机输入 (2,3,256,256) -> {len(outs)} 个输出, '
      f'形状 {[tuple(o.shape) for o in outs]}: {"PASS" if ok else "FAIL"}')
if not ok:
    failures.append('shape_train')
model_t = SCTransNet_No_Sigmoid(mode='test')
model_t.eval()
o_test = model_t(x)
ok = tuple(o_test.shape) == (2, 1, 256, 256)
print(f'[test_sct/1b] mode=test -> 单输出 {tuple(o_test.shape)}: {"PASS" if ok else "FAIL"}')
if not ok:
    failures.append('shape_test')

# ---------- [2] 同源权重灌入 + 对齐 ----------
npz = os.path.join(DATA, 'sct_torch_init.npz')
model_c = convert(npz, out_path=os.path.join(ROOT, 'work_dirs', 'sct_torch_init_jt.pkl'))
model_c.eval()

sigmoid = lambda a: 1.0 / (1.0 + np.exp(-a))
ref = np.load(os.path.join(DATA, 'sct_ref_random.npz'))
xr = ref['x']
with jt.no_grad():
    outs_j = model_c(jt.array(xr))

print('-' * 60)
print('[test_sct/2a] fp32 对齐 (torch CPU 参考, eval):')
for i, n in enumerate(NAMES):
    yr = ref[f'y{i}_{n}']
    yj = outs_j[i].numpy()
    assert yj.shape == yr.shape, f'{n} 形状不一致: {yj.shape} vs {yr.shape}'
    d_logit = float(np.abs(yj - yr).max())
    d_sig = float(np.abs(sigmoid(yj) - sigmoid(yr)).max())
    mismatch = float(np.mean((sigmoid(yj) > 0.5) != (sigmoid(yr) > 0.5)))
    ok = d_sig < 1e-4 and mismatch == 0
    print(f'  [{n}] logit max abs diff = {d_logit:.3e} | sigmoid diff = {d_sig:.3e} '
          f'(<1e-4) | 二值化不一致 = {mismatch:.2e} (要求0): {"PASS" if ok else "FAIL"}')
    if not ok:
        failures.append(f'fp32_{n}')

# ---------- fp64 结构等价性 ----------
print('-' * 60)
model64 = convert(npz, out_path=None, verbose=False)
model64.float64()
model64.eval()
ref64 = np.load(os.path.join(DATA, 'sct_ref_random_fp64.npz'))
with jt.no_grad():
    outs64 = model64(jt.array(ref64['x'].astype(np.float64)))
print('[test_sct/2b] fp64 结构等价性 (阈值 1e-6):')
for i, n in enumerate(NAMES):
    d64 = float(np.abs(outs64[i].numpy() - ref64[f'y{i}_{n}']).max())
    ok = d64 < 1e-6
    print(f'  [{n}] max abs diff = {d64:.3e}: {"PASS" if ok else "FAIL"}')
    if not ok:
        failures.append(f'fp64_{n}')

# ---------- [3] train 模式可反向 ----------
print('-' * 60)
model_g = convert(npz, out_path=None, verbose=False)
model_g.train()
# 用小尺寸输入做可反向冒烟: 梯度通路契约与分辨率无关, 且避免 CPU 全尺寸反向的高昂开销
xs = jt.array(xr[:1, :, :64, :64].copy())
outs_g = model_g(xs)
assert isinstance(outs_g, tuple) and len(outs_g) == 6
loss = sum(o.mean() for o in outs_g)
grads = jt.grad(loss, model_g.parameters())
# 注意: 逐个数 grads[i] 触发单独 sync 会让 jittor 重复执行反向子图 (实测 300s+ 跑不完)。
# 拼成单向量一次 sync, 再算全局范数。
flat = jt.concat([g.reshape(-1) for g in grads], dim=0)
gnorm = float(jt.sqrt((flat * flat).sum()))
ok = np.isfinite(gnorm) and gnorm > 0
print(f'[test_sct/3] train 模式前向+反向 (输入 1x3x64x64): loss = {float(loss):.6f}, '
      f'全局梯度范数 = {gnorm:.6e} (有限非零): {"PASS" if ok else "FAIL"}')
if not ok:
    failures.append('backward')

print('=' * 60)
if failures:
    print('test_sct 未通过:', failures)
    sys.exit(1)
print('test_sct 全部通过 ✓')
