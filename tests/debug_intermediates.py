# -*- coding: utf-8 -*-
"""逐层中间输出对比 (jittor 环境): 找出 diff 从哪一层开始放大。"""
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
model.eval()

ref = np.load(os.path.join(DATA, 'acm_ref_random.npz'))
inter = np.load(os.path.join(DATA, 'acm_intermediates.npz'))
x = jt.array(ref['x'])

with jt.no_grad():
    outs = {}
    outs['stem'] = model.stem(x)
    outs['c1'] = model.layer1(outs['stem'])
    outs['c2'] = model.layer2(outs['c1'])
    outs['c3'] = model.layer3(outs['c2'])
    outs['deconv2'] = model.deconv2(outs['c3'])
    outs['fuse2'] = model.fuse2(outs['deconv2'], outs['c2'])
    outs['upc2'] = model.uplayer2(outs['fuse2'])
    outs['deconv1'] = model.deconv1(outs['upc2'])
    outs['fuse1'] = model.fuse1(outs['deconv1'], outs['c1'])
    outs['upc1'] = model.uplayer1(outs['fuse1'])
    outs['pred'] = model.head(outs['upc1'])
    outs['out'] = jt.nn.interpolate(outs['pred'].stop_fuse(), scale_factor=4, mode='bilinear')

for k, v in outs.items():
    a = v.numpy()
    b = inter[k]
    if a.shape != b.shape:
        print(f'{k:8s} SHAPE MISMATCH jt{a.shape} vs torch{b.shape}')
        continue
    d = float(np.abs(a - b).max())
    rel = d / (float(np.abs(b).max()) + 1e-12)
    print(f'{k:8s} shape={a.shape} max_abs_diff={d:.3e} rel={rel:.3e}')
