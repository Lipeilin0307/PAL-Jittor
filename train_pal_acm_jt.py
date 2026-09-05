# -*- coding: utf-8 -*-
"""PAL 三阶段渐进式主动学习训练脚本（ACM + Jittor）。

用法::

    D:/Anaconda/envs/jittor/python.exe PAL_jittor/train_pal_acm_jt.py \
        --epochs 400 [--start_epoch 0] [--pal_total_epochs 400] \
        [--init_from work_dirs/acm_official_jt.pkl] \
        [--pal_workspace <path>] [--save_dir <path>] \
        [--limit_init N] [--limit_train N] [--limit_val N]

机制语义（逐项对应 PAL/train_model.py L686-855，choose_dataset_type='masks_coarse'）：
- 初始池：若 <pal_workspace>/train/choose/img 不存在或为空，调用
  pal.pal_utils.data_inital_make_add_points 生成（等价原版 L686-689）。
  --limit_init>0 时用硬链子集做初始化（冒烟用；原版无此参数）。
- 调度（原版 L742）：epoch > int(pal_total*0.2) 且 <= int(pal_total*0.8)
  且 epoch%5==0 → 增强轮：标签自更新(L749-768) → no_choose 池推理
  (L771-788) → 难度准入 deal_pred_mask_and_true_point_in（lose_point_ratio
  线性放宽公式 L747 原样） → 伪标签精细化 deal_gen_mask_error_aera →
  文件迁移 hard_sample_in。epoch > int(pal_total*0.8) 且 %5==0 → 精炼轮：
  只做标签自更新(L815-835)。
- 每 epoch 用当前 choose 池重建训练集（原版 L839-851），随后
  train_one_epoch + val_one_epoch（复用 train_acm_jt.py 骨架）。
- 推理（原版 SirstDataset_test L214-243 + test_pred else 分支）：
  RGB 读图 → pad 32 倍数 → Normalize → 整图前向 → sigmoid → 裁回 [h,w]。
  Jittor 版在 pal_infer() 中以 numpy 等价实现（A.Normalize == (x/255-mean)/std）。

与原版差异清单：
1. PAL 工作区默认落在 PAL_jittor/pal_workspaces/<MODEL>__SIRST3__masks_coarse__<时间戳>/
   （原版写进 dataset/SIRST3/ 下；为避免污染 torch 仓库数据目录而改默认位置，
   目录命名与内部结构与原版本完全一致，可用 --pal_workspace 指到任意位置）。
2. 推理 batch=1 单进程（原版 TEST_BATCH_SIZE=1 + DataLoader num_workers=4），
   数值路径相同； Normalize 用 numpy 手写（与 albumentations 结果逐位一致）。
3. no_choose 推理输出前的 cv2.resize(pred,(w,h)) 原样保留（同尺寸恒等操作）。
4. checkpoint 为 jittor pkl（state_dict + 元信息）；每 epoch 另存
   last_checkpoint.pkl 供 --start_epoch 断点续跑（池状态本身在文件系统，
   天然可续）。
5. best_mIoU 从 -1 起步（首轮必存档；原版从 0 起步在 mIoU 恒 0 时永不存档）。
"""
import argparse
import datetime
import os
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image

import jittor as jt

jt.flags.use_cuda = 1

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
PAL_ROOT = os.environ.get('PAL_ROOT', os.path.join(os.path.dirname(ROOT), 'PAL'))

from model.acm import ACM_No_Sigmoid
from data.sirst3_dataset import SirstDataset, build_train_transform, build_val_transform
from data.cal_mean_std import Calculate_mean_std
from pal.pal_utils import (make_dir, data_inital_make_add_points,
                           deal_pred_mask_and_true_point_in,
                           deal_gen_mask_error_aera, hard_sample_in,
                           update_gt_update_degen_corr)
from train_acm_jt import TeeLogger, train_one_epoch, val_one_epoch

# PAL 调度超参（PAL/train_model.py L60-70 原样）
LOSE_POINT_RATIO_INIT = 0.2
ALARM_POINT_RATION = 5
CLEAR_EPOCH_GAP = 5
CLEAR_INITAL_RATIO = 0.2
FINAL_EPOCH_RATIO = 0.8
THRESH_TB = 0.5
THRESH_K = 0.5
DEGEN = 0.97
PATCH_SIZE = 256


@jt.no_grad()
def pal_infer_one(model, img_path, cal_mean, cal_std):
    """单图推理：对应原版 SirstDataset_test(mode='test') + test_pred else 分支。
    返回 (pred_float32[h,w] 已 sigmoid 并裁回原始尺寸, h, w)。"""
    image = np.array(Image.open(img_path).convert('RGB'))
    h, w, _ = image.shape
    times = 32
    pad_h = (times - h % times) % times
    pad_w = (times - w % times) % times
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    # A.Normalize(mean, std, max_pixel_value=255.0) 的 numpy 等价
    x = image.astype(np.float32) / 255.0
    x = (x - cal_mean) / cal_std
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]  # [1,3,H',W']
    model.eval()
    out = model(jt.array(x))
    out = out[0, :, :h, :w]
    pred = jt.sigmoid(out).numpy()[0].astype(np.float32)
    return pred, h, w


def pal_label_self_update(model, img_dir, mask_dir, cal_mean, cal_std, logger):
    """标签自更新（原版 L749-768 / L815-835）：对 choose 池每张某推理 +
    update_gt_update_degen_corr，结果 *255 覆写 PNG 标签文件。"""
    names = os.listdir(img_dir)
    logger.log(f'    [自更新] choose 池 {len(names)} 张')
    for name in names:
        pred, h, w = pal_infer_one(model, os.path.join(img_dir, name),
                                   cal_mean, cal_std)
        mask_path = os.path.join(mask_dir, name)
        prev_label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
        cur = update_gt_update_degen_corr(pred, prev_label, THRESH_TB, THRESH_K,
                                          [h, w], degen=DEGEN)
        cv2.imwrite(mask_path, cur * 255)


def pal_infer_no_choose(model, nc_img_dir, nc_mask_dir, cal_mean, cal_std, logger):
    """no_choose 池推理（原版 L771-788）：预测 >0.5 -> 255 写入 mask_pred/。"""
    names = os.listdir(nc_img_dir)
    logger.log(f'    [推理] no_choose 池 {len(names)} 张')
    for name in names:
        pred, h, w = pal_infer_one(model, os.path.join(nc_img_dir, name),
                                   cal_mean, cal_std)
        pred = cv2.resize(pred, (int(w), int(h)))  # 原版 L785，同尺寸恒等
        pred = np.where(pred > 0.5, 255, 0)
        cv2.imwrite(os.path.join(nc_mask_dir, name), pred)


def build_pool_datasets(ws, cal_mean, cal_std, args):
    """用当前 choose 池重建训练集（原版 L839-851）+ val 集。"""
    train_ds = SirstDataset(
        image_dir=os.path.join(ws, 'train/choose/img'),
        mask_dir=os.path.join(ws, 'train/choose/mask'),
        patch_size=PATCH_SIZE,
        transform=build_train_transform(cal_mean, cal_std),
        mode='train',
    )
    train_ds.set_attrs(batch_size=args.batch_size, shuffle=True, num_workers=0,
                       drop_last=False, keep_numpy_array=True)
    return train_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--start_epoch', type=int, default=0)
    ap.add_argument('--pal_total_epochs', type=int, default=400,
                    help='PAL 调度分母（原版 NUM_EPOCHS=400）；冒烟可缩小')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--init_from', type=str, default='')
    ap.add_argument('--pal_workspace', type=str, default='')
    ap.add_argument('--save_dir', type=str, default='')
    ap.add_argument('--limit_init', type=int, default=0)
    ap.add_argument('--limit_train', type=int, default=0)
    ap.add_argument('--limit_val', type=int, default=0)
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not args.pal_workspace:
        args.pal_workspace = os.path.join(
            ROOT, 'pal_workspaces', f'ACM__SIRST3__masks_coarse__{ts}')
    if not args.save_dir:
        args.save_dir = os.path.join(ROOT, 'work_dirs', f'pal_{ts}')
    make_dir(args.save_dir)
    logger = TeeLogger(os.path.join(args.save_dir, 'pal_train.log'))

    logger.log('=' * 70)
    logger.log('PAL 三阶段训练 (ACM, Jittor)')
    logger.log(f'args: {vars(args)}')
    logger.log(f'jittor {jt.__version__}, use_cuda={jt.flags.use_cuda}')

    # ---- PAL 工作区目录（结构与原版本完全一致）----
    ws = args.pal_workspace
    TRAIN_IMG_DIR = os.path.join(ws, 'train/choose/img')
    TRAIN_MASK_DIR = os.path.join(ws, 'train/choose/mask')
    train_points_dir = os.path.join(ws, 'train/choose/points')
    nc_img_dir = os.path.join(ws, 'train/no_choose/img')
    nc_mask_dir = os.path.join(ws, 'train/no_choose/mask_pred')
    nc_points_dir = os.path.join(ws, 'train/no_choose/points')
    for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_points_dir,
              nc_img_dir, nc_mask_dir, nc_points_dir]:
        make_dir(d)
    logger.log(f'PAL 工作区: {ws}')

    origin_img = os.path.join(PAL_ROOT, 'dataset/SIRST3/origin/img')
    origin_pts = os.path.join(PAL_ROOT, 'dataset/SIRST3/origin/masks_coarse')
    val_img = os.path.join(PAL_ROOT, 'dataset/SIRST3/val/img')
    val_mask = os.path.join(PAL_ROOT, 'dataset/SIRST3/val/mask')

    logger.log('计算数据集 mean/std ...')
    cal_mean, cal_std = Calculate_mean_std(origin_img)
    logger.log(f'cal_mean={cal_mean:.8f} cal_std={cal_std:.8f}')

    # ---- 初始池（仅当 choose/img 为空）----
    if len(os.listdir(TRAIN_IMG_DIR)) == 0:
        if args.limit_init > 0:
            sub_img = os.path.join(ws, '_origin_sub/img')
            sub_pts = os.path.join(ws, '_origin_sub/masks_coarse')
            make_dir(sub_img); make_dir(sub_pts)
            for n in sorted(os.listdir(origin_img))[:args.limit_init]:
                for s_dir, d_dir in [(origin_img, sub_img), (origin_pts, sub_pts)]:
                    dst = os.path.join(d_dir, n)
                    if not os.path.exists(dst):
                        try:
                            os.link(os.path.join(s_dir, n), dst)
                        except OSError:
                            shutil.copy(os.path.join(s_dir, n), dst)
            origin_img_use, origin_pts_use = sub_img, sub_pts
        else:
            origin_img_use, origin_pts_use = origin_img, origin_pts
        logger.log('生成初始池（data_inital_make_add_points）...')
        t0 = time.time()
        data_inital_make_add_points(origin_img_use, origin_pts_use,
                                    TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_points_dir,
                                    nc_img_dir, nc_mask_dir, nc_points_dir,
                                    crop_size=10)
        logger.log(f'初始池: choose={len(os.listdir(TRAIN_IMG_DIR))} '
                   f'no_choose={len(os.listdir(nc_img_dir))} 耗时 {time.time()-t0:.1f}s')
    else:
        logger.log(f'沿用既有工作区: choose={len(os.listdir(TRAIN_IMG_DIR))} '
                   f'no_choose={len(os.listdir(nc_img_dir))}')

    # ---- 模型/优化器 ----
    model = ACM_No_Sigmoid()
    if args.init_from:
        model.load_state_dict(jt.load(args.init_from))
        logger.log(f'热启动: {args.init_from}')
    optimizer = jt.nn.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ---- val 集（全 epoch 共用）----
    val_ds = SirstDataset(val_img, val_mask, patch_size=None,
                          transform=build_val_transform(cal_mean, cal_std),
                          mode='val')
    val_ds.set_attrs(batch_size=1, shuffle=False, num_workers=0,
                     drop_last=False, keep_numpy_array=True)
    if args.limit_val > 0:
        val_ds.images = val_ds.images[:args.limit_val]

    # ---- 断点恢复 ----
    best_mIoU = -1.0
    if args.start_epoch > 0:
        last = os.path.join(args.save_dir, 'last_checkpoint.pkl')
        ck = jt.load(last)
        model.load_state_dict(ck['state_dict'])
        best_mIoU = ck['best_mIoU']
        logger.log(f'断点恢复: epoch={ck["epoch"]} best_mIoU={best_mIoU:.4f}')

    sched = args.pal_total_epochs
    enh_lo, enh_hi = int(sched * CLEAR_INITAL_RATIO), int(sched * FINAL_EPOCH_RATIO)
    logger.log(f'调度: 预启动 [0,{enh_lo}] 增强 ({enh_lo},{enh_hi}] 每{CLEAR_EPOCH_GAP}ep '
               f'精炼 ({enh_hi},{sched}]（pal_total={sched}）')

    # ---- 主循环 ----
    for epoch in range(args.start_epoch, args.epochs):
        t_ep = time.time()
        logger.log(f'--- epoch {epoch + 1}/{args.epochs} (PAL 调度基准 {sched}) ---')

        if epoch > enh_lo and epoch <= enh_hi:
            # ===== 增强期（原版 L742-806）=====
            if epoch % CLEAR_EPOCH_GAP == 0:
                lose_point_ratio = LOSE_POINT_RATIO_INIT + \
                    (epoch - sched * CLEAR_INITAL_RATIO + 1) / \
                    (sched * (FINAL_EPOCH_RATIO - CLEAR_INITAL_RATIO)) * \
                    (1 - LOSE_POINT_RATIO_INIT)
                logger.log(f'  [增强轮] lose_point_ratio={lose_point_ratio:.4f}')
                logger.log('  开始标签的自更新--------')
                pal_label_self_update(model, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                      cal_mean, cal_std, logger)
                logger.log('  开始认识并学习困难样本-----------')
                pal_infer_no_choose(model, nc_img_dir, nc_mask_dir,
                                    cal_mean, cal_std, logger)
                new_choose_list = deal_pred_mask_and_true_point_in(
                    nc_img_dir, nc_mask_dir, nc_points_dir, TRAIN_IMG_DIR,
                    TRAIN_MASK_DIR, train_points_dir,
                    lose_point_ratio=lose_point_ratio,
                    alarm_point_ration=ALARM_POINT_RATION)
                logger.log(f'  本轮准入 {len(new_choose_list)} 张: {new_choose_list[:10]}'
                           f'{"..." if len(new_choose_list) > 10 else ""}')
                deal_gen_mask_error_aera(nc_mask_dir, nc_points_dir, new_choose_list)
                hard_sample_in(nc_img_dir, nc_mask_dir, nc_points_dir,
                               TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_points_dir,
                               new_choose_list)
                logger.log('  此轮数据转移完成！！！！！ '
                           f'choose={len(os.listdir(TRAIN_IMG_DIR))} '
                           f'no_choose={len(os.listdir(nc_img_dir))}')
        elif epoch > enh_hi and epoch % CLEAR_EPOCH_GAP == 0:
            # ===== 精炼期（原版 L810-835）：只做标签自更新 =====
            logger.log('  [精炼轮] 开始标签的自更新--------')
            pal_label_self_update(model, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                  cal_mean, cal_std, logger)

        # ===== 重建训练集并训练一个 epoch（原版 L839-857）=====
        train_ds = build_pool_datasets(ws, cal_mean, cal_std, args)
        if args.limit_train > 0:
            train_ds.images = train_ds.images[:args.limit_train]
        if len(train_ds) == 0:
            logger.log('  [警告] choose 池为空，跳过本 epoch 训练')
            continue
        train_one_epoch(train_ds, model, optimizer, epoch, logger)

        mIoU, nIoU, PD, FA = val_one_epoch(val_ds, model, epoch, logger)
        if best_mIoU < mIoU:
            best_mIoU = mIoU
            jt.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'best_mIoU': best_mIoU, 'best_nIoU': nIoU,
                     'best_PD': PD, 'best_FA': FA},
                    os.path.join(args.save_dir, 'best_mIoU_checkpoint.pkl'))
            logger.log(f'  [ckpt] best mIoU={best_mIoU:.4f} 已保存')
        jt.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                 'best_mIoU': best_mIoU},
                os.path.join(args.save_dir, 'last_checkpoint.pkl'))
        logger.log(f'  epoch {epoch + 1} 总耗时 {time.time()-t_ep:.1f}s')

    logger.log('=' * 70)
    logger.log(f'训练结束。best mIoU={best_mIoU:.4f}; '
               f'choose={len(os.listdir(TRAIN_IMG_DIR))} '
               f'no_choose={len(os.listdir(nc_img_dir))}')
    logger.log('PAL_TRAIN_DONE')
    logger.close()


if __name__ == '__main__':
    main()
