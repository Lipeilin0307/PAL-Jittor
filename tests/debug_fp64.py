# -*- coding: utf-8 -*-
"""fp64 结构等价性验证 (jittor 环境):
jittor fp64 vs torch fp64 参考, 若 diff ~1e-10 级, 证明结构/权重迁移完全正确,
fp32 下的 1e-3 级 diff 纯为浮点累加噪声。
"""
import os
import sys
import numpy as np
import jittor as jt

jt.flags.use_cuda = 0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'tests', 'data')

from model.convert_acm_weights import convert

model = convert(os.path.join(DATA, 'acm_official_torch.npz'), out_path=None, verbose=False)
model.float64()
model.eval()

ref = np.load(os.path.join(DATA, 'acm_ref_random_fp64.npz'))
x = jt.array(ref['x'].astype(np.float64))
with jt.no_grad():
    y = model(x).numpy()
d = float(np.abs(y - ref['y']).max())
print('jittor fp64 vs torch fp64 max abs diff = %.3e' % d)
print('FP64_DONE')
