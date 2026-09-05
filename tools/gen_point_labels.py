# -*- coding: utf-8 -*-
"""gen_point_labels.py — Python port of PAL tools/coarse_anno.m & centroid_anno.m (MATLAB-free).

从全监督二值掩码生成单点标签：
  coarse   : 每个连通域内，以质心为中心、标准差=bbox边长/2*0.25 的高斯偏移采样，直到落入掩码内（复刻 coarse_anno.m）
  centroid : 直接取每个连通域质心（floor），不检查是否落在掩码上（复刻 centroid_anno.m）

用法:
  python gen_point_labels.py --masks_dir <二值掩码目录> --out_dir <输出目录> --mode coarse --seed 42
"""
import argparse
import os
import numpy as np
import cv2


def gen_for_mask(mask_bin, mode, rng):
    """mask_bin: uint8 HxW, 0/255。返回同尺寸 0/255 点标签。"""
    H, W = mask_bin.shape
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        (mask_bin > 0).astype(np.uint8), connectivity=8)
    out = np.zeros((H, W), np.uint8)
    for i in range(1, n):  # 0 是背景
        cx, cy = cents[i]          # 0-based 连续坐标 (x=col, y=row)
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if mode == 'centroid':
            x, y = int(np.floor(cx)), int(np.floor(cy))
            if 0 <= x < W and 0 <= y < H:
                out[y, x] = 255
            continue
        # coarse: 高斯偏移采样，直到落入掩码（对应 MATLAB 的 while sum(img_temp.*img)==0）
        tries = 0
        while True:
            gx = rng.normal(0.0, 0.25)
            gy = rng.normal(0.0, 0.25)
            x = int(np.floor(cx + (w / 2.0) * gx))
            y = int(np.floor(cy + (h / 2.0) * gy))
            x = min(max(x, 0), W - 1)
            y = min(max(y, 0), H - 1)
            tries += 1
            if mask_bin[y, x] > 0 or tries > 10000:
                out[y, x] = 255
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mode', choices=['coarse', 'centroid'], default='coarse')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.masks_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.tif')))
    n_img, n_pt, bad = 0, 0, 0
    for f in files:
        m = cv2.imread(os.path.join(args.masks_dir, f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            print('  [skip unreadable]', f)
            continue
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        out = gen_for_mask(m, args.mode, rng)
        # 自检：coarse 模式下每个点必须落在掩码上
        pts = np.argwhere(out > 0)
        if args.mode == 'coarse':
            for (y, x) in pts:
                if m[y, x] == 0:
                    bad += 1
        n_img += 1
        n_pt += len(pts)
        cv2.imwrite(os.path.join(args.out_dir, f), out)
    print('DONE: %d masks -> %s (%s), total points=%d, off-mask violations=%d'
          % (n_img, args.out_dir, args.mode, n_pt, bad))


if __name__ == '__main__':
    main()
