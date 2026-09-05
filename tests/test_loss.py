# -*- coding: utf-8 -*-
"""edgeSCE_loss Jittor 版验收测试 (在 jittor 环境运行):
同一组随机 logits/target/edge (loss_ref.npz, torch 环境生成),
Jittor 版 edgeSCE_loss 输出标量与 PyTorch 版误差 < 1e-5。
"""
import os
import sys

import numpy as np
import jittor as jt

jt.flags.use_cuda = 1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'tests', 'data')

from loss.edge_sce import edgeSCE_loss

THRESH = 1e-5

ref = np.load(os.path.join(DATA, 'loss_ref.npz'))
logits, target, edge = ref['logits'], ref['target'], ref['edge']
loss_torch = float(ref['loss'])

loss_jt = float(edgeSCE_loss(jt.array(logits), jt.array(target), jt.array(edge)))
diff = abs(loss_jt - loss_torch)
ok = diff < THRESH

print(f'[test_loss] torch edgeSCE_loss = {loss_torch:.10f}')
print(f'[test_loss] jittor edgeSCE_loss = {loss_jt:.10f}')
print(f'[test_loss] abs diff = {diff:.3e}, 阈值 {THRESH:.0e}: {"PASS" if ok else "FAIL"}')

print('=' * 60)
if not ok:
    print('test_loss 未通过')
    sys.exit(1)
print('test_loss 通过 ✓')
