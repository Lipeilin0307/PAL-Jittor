# -*- coding: utf-8 -*-
"""在 jittor 环境运行: 将 jittor 对应算子与 torch_op_probe.py 导出的参考逐一对比。
同时内省: InstanceNorm2d 默认是否带仿射参数 / LeakyReLU 参数名 / chunk 可用性。
"""
import os
import sys
import inspect
import numpy as np
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0

REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'tests', 'data', 'op_probe_ref.npz')
d = np.load(REF)
fails = []


def check(name, yj, yt, tol=1e-6):
    yj = np.asarray(yj)
    diff = float(np.abs(yj - yt).max()) if yj.shape == yt.shape else float('nan')
    ok = yj.shape == yt.shape and diff < tol
    print(f'  {name:26s} shape {str(yj.shape):20s} max diff = {diff:.3e}  '
          f'{"PASS" if ok else "FAIL"}')
    if not ok:
        fails.append(name)
    return ok


print('=== 签名/内省 ===')
print('InstanceNorm2d 签名:', inspect.signature(nn.InstanceNorm2d.__init__))
m_inst = nn.InstanceNorm2d(1)
print('InstanceNorm2d(1) state_dict keys:', list(m_inst.state_dict().keys()))
print('LeakyReLU 签名:', inspect.signature(nn.LeakyReLU.__init__))
print('Conv1d 签名:', inspect.signature(nn.Conv1d.__init__))
print('interpolate 签名:', inspect.signature(nn.interpolate))

print('=== 数值对比 (vs torch ref) ===')

# 1) InstanceNorm2d
m_inst.eval()
with jt.no_grad():
    y = m_inst(jt.array(d['in_x'])).numpy()
check('InstanceNorm2d.eval', y, d['in_eval'])
m_inst.train()
y = m_inst(jt.array(d['in_x'])).numpy()
check('InstanceNorm2d.train', y, d['in_train'])

# 2) nearest x2
with jt.no_grad():
    y = nn.interpolate(jt.array(d['nearest_x']), scale_factor=2, mode='nearest').numpy()
check('interpolate nearest x2', y, d['nearest_y'])

# 3) bilinear align_corners=True
y = nn.interpolate(jt.array(d['bil_t_x']), scale_factor=2, mode='bilinear',
                   align_corners=True).numpy()
check('bilinear ac=True x2', y, d['bil_t_y'])

# 4) bilinear align_corners=False
y = nn.interpolate(jt.array(d['bil_t_x']), scale_factor=2, mode='bilinear',
                   align_corners=False).numpy()
check('bilinear ac=False x2', y, d['bil_f_y'])

# 5) Conv1d
c1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
c1.weight.update(jt.array(d['c1_w']))
y = c1(jt.array(d['c1_x'])).numpy()
check('Conv1d k3 p1', y, d['c1_y'])

# 6) AdaptiveAvgPool2d(1)
ap = nn.AdaptiveAvgPool2d(1)
y = ap(jt.array(d['ap_x'])).numpy()
check('AdaptiveAvgPool2d(1)', y, d['ap_y'])

# 7) LeakyReLU
try:
    lr = nn.LeakyReLU()
    y = lr(jt.array(d['lr_x'])).numpy()
    check('LeakyReLU()', y, d['lr_y'])
except Exception as e:
    print('  LeakyReLU() 无参构造失败:', e)
    lr = nn.LeakyReLU(0.01)
    y = lr(jt.array(d['lr_x'])).numpy()
    check('LeakyReLU(0.01)', y, d['lr_y'])

# 8) chunk
try:
    a, b = jt.array(d['ck_x']).chunk(2, dim=1)
    check('chunk[0]', a.numpy(), d['ck_a'])
    check('chunk[1]', b.numpy(), d['ck_b'])
except Exception as e:
    print('  Var.chunk 不可用:', e)
    fails.append('chunk_api')

# 9) matmul 4D + transpose
q, k, v = jt.array(d['mm_q']), jt.array(d['mm_k']), jt.array(d['mm_v'])
attn = jt.matmul(q, k.transpose(-2, -1))
check('matmul attn', attn.numpy(), d['mm_attn'])
check('matmul out', jt.matmul(attn, v).numpy(), d['mm_out'])
try:
    attn2 = q @ k.transpose(-2, -1)
    check('@ 运算符', attn2.numpy(), d['mm_attn'])
except Exception as e:
    print('  @ 运算符不可用:', e)
    fails.append('at_op')

# 10) var unbiased=False: 手写等价 (x-mu)^2 均值
x = jt.array(d['var_x'])
mu = x.mean(-1, keepdims=True)
var_manual = ((x - mu) ** 2).mean(-1, keepdims=True)
check('manual var(biased)', var_manual.numpy(), d['var_y'])
try:
    var_api = x.var(-1, keepdim=True, unbiased=False)
    check('Var.var(unbiased=False)', var_api.numpy(), d['var_y'])
except TypeError as e:
    print('  Var.var 签名不支持 unbiased:', e)

# 11) normalize dim=-1 (L2, eps=1e-12 clamp)
x = jt.array(d['nm_x'])
denom = jt.sqrt((x * x).sum(-1, keepdims=True))
denom = jt.maximum(denom, 1e-12)
y = (x / denom).numpy()
check('manual L2 normalize', y, d['nm_y'])

# 12) softmax dim=3
y = nn.softmax(jt.array(d['sm_x']), dim=3).numpy()
check('softmax dim=3', y, d['sm_y'])

print('=' * 60)
if fails:
    print('探针未过项:', fails)
    sys.exit(1)
print('全部探针通过 ✓')
