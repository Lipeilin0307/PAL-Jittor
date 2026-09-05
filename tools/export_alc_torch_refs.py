# -*- coding: utf-8 -*-
"""在 PyTorch 环境 (pal_torch) 中运行, 导出 ALC 参考:
1) 官方 checkpoint state_dict -> alc_official_torch.npz
2) 固定随机输入 [2,3,256,256] 前向 -> alc_ref_random.npz (+ fp64 版)
3) SIRST3 val 000001.png (归一化) 前向 -> alc_ref_val.npz
"""
import os
import sys
import numpy as np

PAL_ROOT = r'C:/Users/Alienware/Documents/kimi/workspace/PAL'
OUT_DIR = r'C:/Users/Alienware/Documents/kimi/workspace/PAL_jittor/tests/data'
CKPT = os.path.join(PAL_ROOT, 'work_dirs/ALC__SIRST3__masks_coarse__official',
                    'best_mIoU_checkpoint_ALC__SIRST3__masks_coarse__official.pth.tar')
VAL_IMG = os.path.join(PAL_ROOT, 'dataset/SIRST3/val/img/000001.png')

sys.path.insert(0, PAL_ROOT)
os.makedirs(OUT_DIR, exist_ok=True)

import torch
from model.ALC.ALC_no_sigmoid import ALC_No_Sigmoid

# ---------- 1) 官方权重 ----------
model = ALC_No_Sigmoid()
sd = torch.load(CKPT, map_location='cpu')['state_dict']
model.load_state_dict(sd, strict=True)
model.eval()
print('[1] 官方 ALC 权重 strict 加载成功, keys =', len(sd))
np.savez(os.path.join(OUT_DIR, 'alc_official_torch.npz'),
         **{k: v.numpy().astype(np.float32) for k, v in sd.items()})

# ---------- 2) 随机输入前向 (fp32 + fp64) ----------
rng = np.random.default_rng(1234)
x_rand = rng.standard_normal((2, 3, 256, 256), dtype=np.float32)
with torch.no_grad():
    y_rand = model(torch.from_numpy(x_rand)).numpy()
print('[2] 随机输入前向:', x_rand.shape, '->', y_rand.shape)
np.savez(os.path.join(OUT_DIR, 'alc_ref_random.npz'), x=x_rand, y=y_rand)

md = ALC_No_Sigmoid().double()
md.load_state_dict(sd)
md.eval()
with torch.no_grad():
    y64 = md(torch.from_numpy(x_rand).double()).numpy()
np.savez(os.path.join(OUT_DIR, 'alc_ref_random_fp64.npz'), x=x_rand, y=y64)
print('    fp64 参考已存; torch fp32 vs fp64 max diff =',
      float(np.abs(y_rand.astype(np.float64) - y64).max()))

# ---------- 3) val 真实图像 (归一化 mean=0.38625179/std=0.14124445) ----------
import cv2
mean, std = 0.38625179, 0.14124445
img = cv2.imread(VAL_IMG, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
img = (img - mean) / std
x_val = np.stack([img] * 3, axis=0)[None].astype(np.float32)
with torch.no_grad():
    y_val = model(torch.from_numpy(x_val)).numpy()
print('[3] val 前向:', x_val.shape, '->', y_val.shape,
      'logit range [%.3f, %.3f]' % (y_val.min(), y_val.max()))
np.savez(os.path.join(OUT_DIR, 'alc_ref_val.npz'), x=x_val, y=y_val)
print('EXPORT_DONE')
