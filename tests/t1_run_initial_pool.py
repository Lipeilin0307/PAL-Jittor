# -*- coding: utf-8 -*-
"""T1 步骤1: 初始池生成运行器（两环境各跑一次）。

  # 参考（原版 utilts）:
  D:/Anaconda/envs/pal_torch/python.exe t1_run_initial_pool.py --impl orig --out <dir>
  # 迁移版:
  D:/Anaconda/envs/jittor/python.exe t1_run_initial_pool.py --impl jt --out <dir>

输出目录结构（对齐原版 PAL 工作区）:
  <out>/train/choose/{img,mask,points}
  <out>/train/no_choose/{img,mask_pred,points}
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.dirname(HERE)
PAL_ROOT = os.path.join(os.path.dirname(JT_ROOT), 'PAL')

ap = argparse.ArgumentParser()
ap.add_argument('--impl', choices=['orig', 'jt'], required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0, help='>0 时只处理前 N 张（用临时镜像目录）')
args = ap.parse_args()

origin_img = os.path.join(PAL_ROOT, 'dataset/SIRST3/origin/img')
origin_pts = os.path.join(PAL_ROOT, 'dataset/SIRST3/origin/masks_coarse')

TRAIN_IMG_DIR = os.path.join(args.out, 'train/choose/img')
TRAIN_MASK_DIR = os.path.join(args.out, 'train/choose/mask')
train_points_dir = os.path.join(args.out, 'train/choose/points')
nc_img_dir = os.path.join(args.out, 'train/no_choose/img')
nc_mask_dir = os.path.join(args.out, 'train/no_choose/mask_pred')
nc_points_dir = os.path.join(args.out, 'train/no_choose/points')
for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_points_dir, nc_img_dir, nc_mask_dir, nc_points_dir]:
    os.makedirs(d, exist_ok=True)

if args.impl == 'orig':
    sys.path.insert(0, PAL_ROOT)
    from utilts import data_inital_make_add_points
else:
    sys.path.insert(0, JT_ROOT)
    from pal.pal_utils import data_inital_make_add_points

if args.limit > 0:
    # 用符号链接子集构建临时 origin（保证两实现看到完全相同的输入清单）
    import shutil
    sub_img = os.path.join(args.out, '_origin_sub/img')
    sub_pts = os.path.join(args.out, '_origin_sub/masks_coarse')
    os.makedirs(sub_img, exist_ok=True)
    os.makedirs(sub_pts, exist_ok=True)
    names = sorted(os.listdir(origin_img))[:args.limit]
    for n in names:
        for src_dir, dst_dir in [(origin_img, sub_img), (origin_pts, sub_pts)]:
            dst = os.path.join(dst_dir, n)
            if not os.path.exists(dst):
                try:
                    os.link(os.path.join(src_dir, n), dst)
                except OSError:
                    shutil.copy(os.path.join(src_dir, n), dst)
    origin_img, origin_pts = sub_img, sub_pts

print(f'[{args.impl}] 输入 {len(os.listdir(origin_img))} 张 -> {args.out}')
data_inital_make_add_points(origin_img, origin_pts, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                            train_points_dir, nc_img_dir, nc_mask_dir, nc_points_dir,
                            crop_size=10)
print('T1_GEN_DONE', args.impl)
