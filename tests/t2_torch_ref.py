# -*- coding: utf-8 -*-
"""T2 辅助：torch 环境参考计算（由 test_pal_mechanism.py 以子进程调用）。

输入: --snap 目录（jittor 侧已落盘）:
  snap/update_gt/pred_XXX.npy      float32 [h,w] sigmoid 预测
  snap/update_gt/prev_XXX.npy      float32 [h,w] {0,1} 旧标签
  snap/admit/<ws 相对路径镜像>      no_choose 池快照 (img/points/mask_pred)
输出:
  snap/ref/update_gt/out_XXX.npy   原版 update_gt_update_degen_corr 输出
  snap/ref/admit_new_choose.json   原版 deal_pred_mask_and_true_point_in 判定
"""
import argparse
import json
import os
import sys

import numpy as np

PAL_ROOT = r'C:/Users/Alienware/Documents/kimi/workspace/PAL'
sys.path.insert(0, PAL_ROOT)
from utilts import update_gt_update_degen_corr, deal_pred_mask_and_true_point_in

ap = argparse.ArgumentParser()
ap.add_argument('--snap', required=True)
ap.add_argument('--lose_point_ratio', type=float, required=True)
ap.add_argument('--alarm_point_ration', type=float, default=5)
args = ap.parse_args()

# 1) update_gt 参考
ref_dir = os.path.join(args.snap, 'ref', 'update_gt')
os.makedirs(ref_dir, exist_ok=True)
ug = os.path.join(args.snap, 'update_gt')
for fn in sorted(os.listdir(ug)):
    if fn.startswith('pred_'):
        key = fn[5:-4]
        pred = np.load(os.path.join(ug, fn))
        prev = np.load(os.path.join(ug, f'prev_{key}.npy'))
        h, w = pred.shape
        out = update_gt_update_degen_corr(pred, prev, 0.5, 0.5, [h, w], degen=0.97)
        np.save(os.path.join(ref_dir, f'out_{key}.npy'), out)
        print('update_gt ref:', key, out.shape, float(out.max()))

# 2) 准入判定参考
ad = os.path.join(args.snap, 'admit')
lst = deal_pred_mask_and_true_point_in(
    os.path.join(ad, 'img'), os.path.join(ad, 'mask_pred'),
    os.path.join(ad, 'points'),
    os.path.join(ad, 'img'), os.path.join(ad, 'mask_pred'),
    os.path.join(ad, 'points'),
    lose_point_ratio=args.lose_point_ratio,
    alarm_point_ration=args.alarm_point_ration)
with open(os.path.join(args.snap, 'ref', 'admit_new_choose.json'), 'w') as f:
    json.dump(sorted(lst), f)
print('admit ref:', len(lst), '张')
print('T2_REF_DONE')
