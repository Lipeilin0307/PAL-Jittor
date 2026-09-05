# -*- coding: utf-8 -*-
"""ALC Jittor 版验收测试 (在 jittor 环境运行)，与 test_acm.py 同规格:

[1] 形状测试: 随机输入 [2,3,256,256] -> [2,1,256,256]
[2a] fp32 参数对齐 (torch CPU 参考, eval): 随机输入 + val 真实图像(归一化)
     —— logit 级如实报告; sigmoid 决策级 < 1e-4; 二值化不一致像素 = 0
[2b] fp64 结构等价性决定性判据: max abs diff < 1e-6 量级
"""
import os
import sys

import numpy as np
import jittor as jt

jt.flags.use_cuda = 0  # 与 torch CPU 参考对齐

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'tests', 'data')

from model.alc import ALC_No_Sigmoid
from model.convert_alc_weights import convert

failures = []

# ---------- [1] 形状测试 ----------
model = ALC_No_Sigmoid()
model.eval()
x = jt.array(np.random.randn(2, 3, 256, 256).astype(np.float32))
y = model(x)
expect_shape = (2, 1, 256, 256)
ok = tuple(y.shape) == expect_shape
print(f'[test_alc/1] 随机输入 (2,3,256,256) -> 输出 {tuple(y.shape)}, '
      f'期望 {expect_shape}: {"PASS" if ok else "FAIL"}')
if not ok:
    failures.append('shape')

# ---------- [2] 官方权重转换 + 对齐 ----------
npz = os.path.join(DATA, 'alc_official_torch.npz')
model_t = convert(npz, out_path=os.path.join(ROOT, 'work_dirs', 'alc_official_jt.pkl'))
model_t.eval()

sigmoid = lambda a: 1.0 / (1.0 + np.exp(-a))
print('-' * 60)
print('[test_alc/2a] fp32 参数对齐 (torch CPU 参考, eval):')
for tag, fname in [('随机输入', 'alc_ref_random.npz'), ('val真实图像', 'alc_ref_val.npz')]:
    ref = np.load(os.path.join(DATA, fname))
    xr, yr = ref['x'], ref['y']
    with jt.no_grad():
        yj = model_t(jt.array(xr)).numpy()
    assert yj.shape == yr.shape, f'{tag} 输出形状不一致: {yj.shape} vs {yr.shape}'
    d_logit = float(np.abs(yj - yr).max())
    d_sig = float(np.abs(sigmoid(yj) - sigmoid(yr)).max())
    mismatch = float(np.mean((sigmoid(yj) > 0.5) != (sigmoid(yr) > 0.5)))
    print(f'  [{tag}] 输出 {yj.shape}:')
    print(f'    logit 级 max abs diff = {d_logit:.3e}')
    print(f'    sigmoid 级 max abs diff = {d_sig:.3e}  阈值 1e-4: '
          f'{"PASS" if d_sig < 1e-4 else "FAIL"}')
    print(f'    二值化不一致像素比例 = {mismatch:.2e}  (要求 0): '
          f'{"PASS" if mismatch == 0 else "FAIL"}')
    if d_sig >= 1e-4:
        failures.append(f'sigmoid_{fname}')
    if mismatch != 0:
        failures.append(f'binarize_{fname}')

# ---------- fp64 结构等价性 ----------
print('-' * 60)
model64 = convert(npz, out_path=None, verbose=False)
model64.float64()
model64.eval()
ref64 = np.load(os.path.join(DATA, 'alc_ref_random_fp64.npz'))
with jt.no_grad():
    y64 = model64(jt.array(ref64['x'].astype(np.float64))).numpy()
d64 = float(np.abs(y64 - ref64['y']).max())
ok64 = d64 < 1e-6
print(f'[test_alc/2b] fp64 结构等价性: max abs diff = {d64:.3e}, 阈值 1e-6: '
      f'{"PASS" if ok64 else "FAIL"}')
if not ok64:
    failures.append('fp64_structure')

print('=' * 60)
if failures:
    print('test_alc 未通过:', failures)
    sys.exit(1)
print('test_alc 全部通过 ✓')
