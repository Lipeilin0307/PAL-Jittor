# -*- coding: utf-8 -*-
"""逐层中间输出导出 (torch 环境), 用于定位迁移 diff 来源。"""
import os
import sys
import numpy as np

PAL_ROOT = r'C:/Users/Alienware/Documents/kimi/workspace/PAL'
DATA = r'C:/Users/Alienware/Documents/kimi/workspace/PAL_jittor/tests/data'
CKPT = os.path.join(PAL_ROOT, 'work_dirs/ACM__SIRST3__masks_coarse__official',
                    'best_mIoU_checkpoint_ACM__SIRST3__masks_coarse__official.pth.tar')
sys.path.insert(0, PAL_ROOT)
import torch
from model.ACM.ACM_no_sigmoid import ACM_No_Sigmoid

model = ACM_No_Sigmoid()
model.load_state_dict(torch.load(CKPT, map_location='cpu')['state_dict'])
model.eval()

ref = np.load(os.path.join(DATA, 'acm_ref_random.npz'))
x = torch.from_numpy(ref['x'])

outs = {}
with torch.no_grad():
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
    outs['out'] = torch.nn.functional.interpolate(outs['pred'], scale_factor=4, mode='bilinear')

np.savez(os.path.join(DATA, 'acm_intermediates.npz'),
         **{k: v.numpy() for k, v in outs.items()})
print('INTERMEDIATES_SAVED', {k: tuple(v.shape) for k, v in outs.items()})
