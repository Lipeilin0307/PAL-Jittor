# -*- coding: utf-8 -*-
"""在 PyTorch 环境 (pal_torch) 中运行:
1) 导出官方 ACM checkpoint 的 state_dict -> acm_official_torch.npz
2) 固定随机输入 [2,3,256,256] 的 PyTorch 前向输出 -> acm_ref_random.npz
3) SIRST3 val 真实图像(000001.png)预处理输入 + PyTorch 输出 -> acm_ref_val.npz
4) 随机 logits/target/edge + PyTorch edgeSCE_loss 标量 -> loss_ref.npz
"""
import os
import sys
import numpy as np

PAL_ROOT = r'C:/Users/Alienware/Documents/kimi/workspace/PAL'
OUT_DIR = r'C:/Users/Alienware/Documents/kimi/workspace/PAL_jittor/tests/data'
CKPT = os.path.join(PAL_ROOT, 'work_dirs/ACM__SIRST3__masks_coarse__official',
                    'best_mIoU_checkpoint_ACM__SIRST3__masks_coarse__official.pth.tar')
VAL_IMG = os.path.join(PAL_ROOT, 'dataset/SIRST3/val/img/000001.png')

sys.path.insert(0, PAL_ROOT)
os.makedirs(OUT_DIR, exist_ok=True)

import torch
from model.ACM.ACM_no_sigmoid import ACM_No_Sigmoid

torch.manual_seed(0)

# ---------- 1) 官方权重 ----------
model = ACM_No_Sigmoid()
ckpt = torch.load(CKPT, map_location='cpu')
sd = ckpt['state_dict']
missing, unexpected = model.load_state_dict(sd, strict=True), None
model.eval()
print('[1] 官方权重 load_state_dict(strict=True) 成功, keys =', len(sd))
npz_sd = {k: v.numpy().astype(np.float32) for k, v in sd.items()}
np.savez(os.path.join(OUT_DIR, 'acm_official_torch.npz'), **npz_sd)
print('    -> acm_official_torch.npz  (%d tensors)' % len(npz_sd))

# ---------- 2) 随机输入前向 ----------
rng = np.random.default_rng(1234)
x_rand = rng.standard_normal((2, 3, 256, 256), dtype=np.float32)
with torch.no_grad():
    y_rand = model(torch.from_numpy(x_rand)).numpy()
print('[2] 随机输入前向:', x_rand.shape, '->', y_rand.shape)
np.savez(os.path.join(OUT_DIR, 'acm_ref_random.npz'), x=x_rand, y=y_rand)

# ---------- 3) val 真实图像前向 ----------
import cv2
from PIL import Image
# 复刻 components/cal_mean_std.py: 对 origin/img 逐图求灰度均值/方差再平均
origin_dir = os.path.join(PAL_ROOT, 'dataset/SIRST3/origin/img')
ml, sl = [], []
for fn in os.listdir(origin_dir):
    im = np.array(Image.open(os.path.join(origin_dir, fn)).convert('L'))
    ml.append(im.mean()); sl.append(im.std())
cal_mean, cal_std = float(np.mean(ml)) / 255, float(np.mean(sl)) / 255
print('[3] SIRST3 origin/img cal_mean=%.6f cal_std=%.6f' % (cal_mean, cal_std))

img = cv2.imread(VAL_IMG, cv2.IMREAD_GRAYSCALE)
print('    val 原图尺寸:', img.shape)
img_f = img.astype(np.float32) / 255.0
img_n = (img_f - cal_mean) / cal_std            # A.Normalize 等价
img_rgb = np.stack([img_n, img_n, img_n], axis=0).astype(np.float32)  # [3,H,W]
h, w = img_rgb.shape[1:]
# 居中裁剪/反射填充到 256x256, 确定性预处理, 两环境共享同一数组
H = W = 256
if h >= H:
    top = (h - H) // 2; img_rgb = img_rgb[:, top:top + H, :]
else:
    pad_t = (H - h) // 2; pad_b = H - h - pad_t
    img_rgb = np.pad(img_rgb, ((0, 0), (pad_t, pad_b), (0, 0)), mode='reflect')
if w >= W:
    left = (w - W) // 2; img_rgb = img_rgb[:, :, left:left + W]
else:
    pad_l = (W - w) // 2; pad_r = W - w - pad_l
    img_rgb = np.pad(img_rgb, ((0, 0), (0, 0), (pad_l, pad_r)), mode='reflect')
x_val = img_rgb[None].astype(np.float32)  # [1,3,256,256]
with torch.no_grad():
    y_val = model(torch.from_numpy(x_val)).numpy()
print('    val 前向:', x_val.shape, '->', y_val.shape,
      'pred logit range [%.3f, %.3f]' % (y_val.min(), y_val.max()))
np.savez(os.path.join(OUT_DIR, 'acm_ref_val.npz'), x=x_val, y=y_val)

# ---------- 4) edgeSCE loss 参考 ----------
from loss.Edge_loss import edgeSCE_loss
rng2 = np.random.default_rng(5678)
logits = rng2.standard_normal((2, 1, 256, 256), dtype=np.float32) * 2.0
target = (rng2.random((2, 1, 256, 256)) > 0.98).astype(np.float32)  # 稀疏目标, 类似小目标
edge = np.zeros_like(target)
edge[target > 0] = 1.0
# 再加一些随机 edge 像素
edge[rng2.random((2, 1, 256, 256)) > 0.995] = 1.0
loss_t = edgeSCE_loss(torch.from_numpy(logits), torch.from_numpy(target),
                      torch.from_numpy(edge))
print('[4] torch edgeSCE_loss =', float(loss_t))
np.savez(os.path.join(OUT_DIR, 'loss_ref.npz'),
         logits=logits, target=target, edge=edge, loss=float(loss_t))

print('EXPORT_DONE ->', OUT_DIR)
