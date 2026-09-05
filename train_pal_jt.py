# -*- coding: utf-8 -*-
"""PAL 三阶段渐进式主动学习训练脚本（泛化版，支持 ACM / ALC / SCT / ISNet）。

由 train_pal_acm_jt.py 泛化而来，唯一功能差异：新增 --model {ACM,ALC,SCT,ISNet}，
PAL 工作区命名 <model>__SIRST3__masks_coarse__<时间戳>。ALC 与 ACM 同为
单输出网络，loss/val/指标/PAL 机制路径完全一致。
SCT (SCTransNet) 为 6 分支深监督输出网络 (mode='train' 构造)，差异仅在:
  - 训练 loss: 6 个分支 (gt5,gt4,gt3,gt2,d0,out) 各自对原尺寸 targets 算一次
    edgeSCE/guard loss 再取 mean (对齐 PAL/train_model.py L294-304 语义，
    targets 无需下采样，6 分支均已是全分辨率);
  - 指标/val/PAL 推理: 一律取 outs[-1] (out 分支) 过 sigmoid。
ISNet 为 2 分支输出 (out, edge_out)，语义按 train_model.py L285-292/164/192:
  - 训练 loss 只算 out 分支 pred[0] (edge 分支 loss 在原版被注释);
  - 指标/val/PAL 推理取 pred[0] 过 sigmoid。
多输出判定统一用 isinstance(out, (tuple, list)) + 注册表中的 (out_index, loss_mode)
约定，ACM/ALC 单输出路径零改动。

用法::

    D:/Anaconda/envs/jittor/python.exe PAL_jittor/train_pal_jt.py \
        --model ALC --epochs 400 [--pal_total_epochs 400] \
        [--init_from work_dirs/alc_official_jt.pkl] [--limit_init N] ...

机制语义与差异清单见 train_pal_acm_jt.py docstring，此处不再重复。
"""
import argparse
import datetime
import os
import random
import shutil
import sys
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image

import jittor as jt

# PAL_JT_FORCE_CPU=1 可强制 CPU（用于 guard-off 逐位确定性回归测试）
jt.flags.use_cuda = 0 if os.environ.get('PAL_JT_FORCE_CPU') == '1' else 1

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
# 数据集根目录: 需包含 dataset/SIRST3/{origin,val,img_idx}。
# 默认取原版 PAL 仓库布局（本仓库与 PAL/ 同级）；可用环境变量 PAL_ROOT 覆盖，
# 例如 set PAL_ROOT=D:/data/PAL 或 export PAL_ROOT=/root/datasets/PAL。
PAL_ROOT = os.environ.get('PAL_ROOT', os.path.join(os.path.dirname(ROOT), 'PAL'))

from model.acm import ACM_No_Sigmoid
from model.alc import ALC_No_Sigmoid
from model.sct import SCTransNet_No_Sigmoid
from model.isnet import ISNet_No_Sigmoid
from data.sirst3_dataset import SirstDataset, build_train_transform, build_val_transform
from data.cal_mean_std import Calculate_mean_std
from pal.pal_utils import (make_dir, data_inital_make_add_points,
                           deal_pred_mask_and_true_point_in,
                           deal_gen_mask_error_aera, hard_sample_in,
                           update_gt_update_degen_corr)
from loss.edge_sce import edgeSCE_loss, edge_sce_loss_guard
from metrics.metric import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA_2
from train_acm_jt import TeeLogger, val_one_epoch

# 模型注册表: name -> (类, 构造 kwargs, 多输出约定)。
# 多输出约定 = (指标/推理取用分支索引 out_index, 深监督 loss 模式 loss_mode):
#   SCT   : 6 分支 (gt5..d0,out), loss=逐分支取 mean, 指标取 [-1]
#           (train_model.py L294-304);
#   ISNet : (out, edge_out), **loss 只算 out 分支** (edge 分支 loss 原版被注释),
#           指标/推理取 [0] (train_model.py L285-292/164/192);
#   ACM/ALC 单输出, 约定为 None (不进入任何 tuple 分支, 路径与旧版逐字节一致)。
MODELS = {'ACM': (ACM_No_Sigmoid, {}, None),
          'ALC': (ALC_No_Sigmoid, {}, None),
          'SCT': (SCTransNet_No_Sigmoid, {'mode': 'train'}, (-1, 'mean_all')),
          'ISNet': (ISNet_No_Sigmoid, {}, (0, 'first_only'))}

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
def pal_infer_one(model, img_path, cal_mean, cal_std, out_index=-1):
    """单图推理：对应原版 SirstDataset_test(mode='test') + test_pred else 分支。
    多输出网络按 out_index 取分支 (SCT=-1, ISNet=0, 原版 test_pred 的逐模型分支)。
    返回 (pred_float32[h,w] 已 sigmoid 并裁回原始尺寸, h, w)。"""
    image = np.array(Image.open(img_path).convert('RGB'))
    h, w, _ = image.shape
    times = 32
    pad_h = (times - h % times) % times
    pad_w = (times - w % times) % times
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    x = image.astype(np.float32) / 255.0
    x = (x - cal_mean) / cal_std
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
    model.eval()
    out = model(jt.array(x))
    if isinstance(out, (tuple, list)):
        out = out[out_index]     # 多输出网络取指定分支
    out = out[0, :, :h, :w]
    pred = jt.sigmoid(out).numpy()[0].astype(np.float32)
    return pred, h, w


def pal_label_self_update(model, img_dir, mask_dir, cal_mean, cal_std, logger,
                          out_index=-1):
    names = os.listdir(img_dir)
    logger.log(f'    [自更新] choose 池 {len(names)} 张')
    for name in names:
        pred, h, w = pal_infer_one(model, os.path.join(img_dir, name),
                                   cal_mean, cal_std, out_index)
        mask_path = os.path.join(mask_dir, name)
        prev_label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
        cur = update_gt_update_degen_corr(pred, prev_label, THRESH_TB, THRESH_K,
                                          [h, w], degen=DEGEN)
        cv2.imwrite(mask_path, cur * 255)


def pal_infer_no_choose(model, nc_img_dir, nc_mask_dir, cal_mean, cal_std, logger,
                        out_index=-1):
    names = os.listdir(nc_img_dir)
    logger.log(f'    [推理] no_choose 池 {len(names)} 张')
    for name in names:
        pred, h, w = pal_infer_one(model, os.path.join(nc_img_dir, name),
                                   cal_mean, cal_std, out_index)
        pred = cv2.resize(pred, (int(w), int(h)))
        pred = np.where(pred > 0.5, 255, 0)
        cv2.imwrite(os.path.join(nc_mask_dir, name), pred)


def build_pool_datasets(ws, cal_mean, cal_std, args):
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


# --------------------------- PAL-Guard v2 (抗塌缩保护) ---------------------------
# 背景: edgeSCE(top-50% OHEM) + 点监督下训练易陷入全背景塌缩态
# (train_IoU=0 但 loss 平滑下降), 逃离时机是高方差随机变量
# (torch {21,91,106}, jittor {93,94,179,235,286})。塌缩期前景像素 (~0.1%)
# 在 OHEM 难例 mining 中被背景碾压, 逃离靠随机游走。
# v1 服务器 A/B 教训: 退出标准太松 (train_IoU>0.02x2轮) + 硬切回 edgeSCE,
# 模型接不住 -> 退出后二次塌缩, 触发过的两发终点 mIoU 只有 vanilla 一半。
# v2: 1) 退出收紧为 train_IoU>0.05 连续 3 轮; 2) 渐进混合退出 (blend 期
# loss=(1-λ)·edgeSCE+λ·平衡BCE, λ 从 1 线性降至 0); 3) blend 期逐轮塌缩
# 判定, 再塌缩 λ 立即回 1。触发条件不变。默认关闭 (--guard 开启);
# 关闭时训练路径与无 guard 版本严格一致。

class GuardController:
    """塌缩检测/退出状态机 v2 (独立于训练循环, 便于单测)。

    状态机: off -> active -> blend -> off (可多次循环)
      off:    纯 edgeSCE (λ=None)。1-based epoch e1 >= min_epoch 且最近
              window 轮 max(train_IoU) < act_th -> active。
      active: 纯平衡 BCE (λ=1)。train_IoU > exit_th 连续 exit_patience 轮
              -> blend (blend_epochs=0 时直接回 off)。
      blend:  λ_t = 1 - (t-1)/blend_epochs, t=1..blend_epochs
              (即 1.0, 0.9, ..., 0.1; 第 blend_epochs 轮完成后 λ=None)。
              期间每轮仍做塌缩判定: window 内 max < act_th -> 立即回 active。
    update(epoch0, train_iou) 返回**下一轮**应使用的 λ (None=纯 edgeSCE)。
    所有状态切换通过 log_fn 打 [GUARD] 日志。
    """

    def __init__(self, enabled, window=5, min_epoch=40, act_th=0.005,
                 exit_th=0.05, exit_patience=3, blend_epochs=10, log_fn=print):
        self.enabled = enabled
        self.window = window
        self.min_epoch = min_epoch
        self.act_th = act_th
        self.exit_th = exit_th
        self.exit_patience = exit_patience
        self.blend_epochs = blend_epochs
        self.log_fn = log_fn
        self.state = 'off'            # off | active | blend
        self.hist = deque(maxlen=window)
        self.exit_streak = 0
        self.blend_pos = 0            # 当前 blend 轮次 (1-based)
        self.current_lam = None       # 下一轮应使用的 λ
        self.activations = []         # 激活 epoch (1-based)
        self.exits = []               # 退出 (进入 blend 或直接关闭) epoch
        self.blend_done = []          # blend 完成 epoch
        self.recollapses = []         # blend 期再塌缩回退 epoch

    @property
    def active(self):
        """兼容 v1 语义: 全保护期 (λ=1) 视为 active。"""
        return self.state == 'active'

    def _collapsed(self):
        return (len(self.hist) == self.hist.maxlen
                and max(self.hist) < self.act_th)

    def update(self, epoch, train_iou):
        """epoch: 刚结束 epoch 的 0-based 索引; train_iou: 该 epoch train_IoU。
        返回下一轮应使用的 λ (None=纯 edgeSCE)。"""
        if not self.enabled:
            return None
        e1 = epoch + 1
        tiou = float(train_iou)
        self.hist.append(tiou)

        if self.state == 'active':
            if tiou > self.exit_th:
                self.exit_streak += 1
            else:
                self.exit_streak = 0
            if self.exit_streak >= self.exit_patience:
                self.exit_streak = 0
                self.exits.append(e1)
                if self.blend_epochs > 0:
                    self.state = 'blend'
                    self.blend_pos = 0
                    self.log_fn(
                        f'  [GUARD] epoch {e1}: train_IoU={tiou:.4f} 已连续 '
                        f'{self.exit_patience} 轮 > {self.exit_th}, 开始退出: '
                        f'进入 blend 期 ({self.blend_epochs} 轮线性退火, '
                        f'下轮 λ=1.00)')
                else:
                    self.state = 'off'
                    self.log_fn(
                        f'  [GUARD] epoch {e1}: train_IoU={tiou:.4f} 已连续 '
                        f'{self.exit_patience} 轮 > {self.exit_th}, '
                        f'关闭保护 (无 blend), 恢复 edgeSCE(top-50% OHEM)')
        elif self.state == 'blend':
            if self._collapsed():
                self.state = 'active'
                self.exit_streak = 0
                self.recollapses.append(e1)
                self.log_fn(
                    f'  [GUARD] epoch {e1}: blend 期最近 {len(self.hist)} 轮 '
                    f'train_IoU 最大值 {max(self.hist):.5f} < {self.act_th}, '
                    f'再塌缩, λ 立即回 1, 重新进入全保护')
        else:  # off
            if e1 >= self.min_epoch and self._collapsed():
                self.state = 'active'
                self.exit_streak = 0
                self.activations.append(e1)
                self.log_fn(
                    f'  [GUARD] epoch {e1}: 最近 {len(self.hist)} 轮 train_IoU '
                    f'最大值 {max(self.hist):.5f} < {self.act_th}, '
                    f'判定陷入全背景塌缩, 激活保护 (切换为平衡 BCE, '
                    f'pos_weight=min(n_neg/n_pos,1000), 无 OHEM)')

        # 计算下一轮 λ
        if self.state == 'active':
            self.current_lam = 1.0
        elif self.state == 'blend':
            self.blend_pos += 1
            if self.blend_pos > self.blend_epochs:
                self.state = 'off'
                self.current_lam = None
                self.blend_done.append(e1)
                self.log_fn(
                    f'  [GUARD] epoch {e1}: blend {self.blend_epochs} 轮完成, '
                    f'λ=0, 完全恢复 edgeSCE(top-50% OHEM)')
            else:
                self.current_lam = 1.0 - (self.blend_pos - 1) / self.blend_epochs
        else:
            self.current_lam = None
        return self.current_lam


def _guard_mix_loss(pred, targets, edge_t, lam):
    """按 λ 混合 edgeSCE 与平衡 BCE (供 SCT 深监督多分支复用;
    单输出路径在 train_one_epoch_guard 内显式分支, 保证 guard-off 字面等价)。"""
    if lam is None:
        return edgeSCE_loss(pred, targets, edge_t)
    if lam >= 1.0:
        return edge_sce_loss_guard(pred, targets, edge_t)
    return ((1.0 - lam) * edgeSCE_loss(pred, targets, edge_t)
            + lam * edge_sce_loss_guard(pred, targets, edge_t))


def train_one_epoch_guard(train_ds, model, optimizer, epoch, logger,
                          guard_lambda=None, loss_mode='mean_all'):
    """train_acm_jt.train_one_epoch 的本地扩展版: 多返回 (tIoU, tnIoU)。

    guard_lambda=None 时与 train_acm_jt.train_one_epoch 逐字节同语义
    (同算子顺序、同 edgeSCE_loss 调用、同日志格式), 仅返回值不同;
    guard_lambda=1.0 时逐 batch 为纯平衡 BCE (edge_sce_loss_guard);
    0<λ<1 时为 blend: loss = (1-λ)·edgeSCE + λ·平衡BCE。
    loss_mode 仅作用于多输出模型: 'mean_all' (SCT, 逐分支 mean) /
    'first_only' (ISNet, 只算 out 分支 pred[0])。
    不改动共享文件 train_acm_jt.py (train_pal_acm_jt.py 依赖其签名)。
    """
    model.train()
    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    losses = []
    t0 = time.time()
    nb = 0
    for img, mask, edge in train_ds:
        data = jt.array(img)                    # [B,3,256,256] fp32 已归一化
        targets = jt.array(mask).unsqueeze(1)   # [B,1,256,256] {0,1}，对齐原版 train_fn
        edge_t = jt.array(edge)                 # [B,1,256,256] {0,255}
        pred = model(data)
        if isinstance(pred, (tuple, list)):
            if loss_mode == 'first_only':
                # ISNet: loss 只算 out 分支 pred[0] (torch 版 edge 分支 loss 被注释,
                # train_model.py L285-292); guard λ 照常透传; 指标统计 pred[0]
                loss = _guard_mix_loss(pred[0], targets, edge_t, guard_lambda)
                pred = pred[0]
            else:
                # 多输出深监督 (SCT): 6 分支 (gt5,gt4,gt3,gt2,d0,out) 各自对原尺寸
                # targets 算一次 loss 再取 mean (对齐原版 train_model.py L294-304:
                # loss = mean(loss_gt5 + ... + loss_out)); guard λ 照常透传。
                # 6 分支均已是全分辨率, targets 无需下采样。
                loss = sum(_guard_mix_loss(p, targets, edge_t, guard_lambda)
                           for p in pred) / len(pred)
                pred = pred[-1]     # train_IoU/nIoU 只统计 out 分支
        elif guard_lambda is None:
            loss = edgeSCE_loss(pred, targets, edge_t)
        elif guard_lambda >= 1.0:
            loss = edge_sce_loss_guard(pred, targets, edge_t)
        else:
            loss = ((1.0 - guard_lambda) * edgeSCE_loss(pred, targets, edge_t)
                    + guard_lambda * edge_sce_loss_guard(pred, targets, edge_t))
        optimizer.step(loss)
        losses.append(float(loss.numpy().item()))
        with jt.no_grad():
            preds = jt.sigmoid(pred)
        iou_metric.update(preds, targets)
        nIoU_metric.update(preds, targets)
        nb += 1
    _, tIoU = iou_metric.get()
    _, tnIoU = nIoU_metric.get()
    dt = time.time() - t0
    lam_tag = '' if guard_lambda is None else f' [GUARD λ={guard_lambda:.2f}]'
    logger.log(f'  [train] epoch {epoch + 1}: loss_mean={np.mean(losses):.6f} '
               f'(min={np.min(losses):.4f}, max={np.max(losses):.4f}, batches={nb}) '
               f'train_IoU={tIoU:.4f} train_nIoU={tnIoU:.4f} 耗时 {dt:.1f}s{lam_tag}')
    return float(np.mean(losses)), float(tIoU), float(tnIoU)
# ------------------------------------------------------------------------------


def val_one_epoch_ds(val_ds, model, epoch, logger, out_index=-1):
    """train_acm_jt.val_one_epoch 的本地多输出版:
    model(x) 返回 tuple/list 时按 out_index 取分支 (SCT=-1 out 分支, ISNet=0)
    再过 sigmoid 算指标; 单输出模型路径与 val_one_epoch 逐字节同语义。
    不改动共享文件 train_acm_jt.py。
    """
    model.eval()
    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    fa_pd_metric = PD_FA_2(1)
    t0 = time.time()
    with jt.no_grad():
        for img, mask, h, w in val_ds:
            x = jt.array(img)                   # [1,3,H',W'] pad-32
            pred = model(x)
            if isinstance(pred, (tuple, list)):
                pred = pred[out_index]          # 多输出网络取指定分支
            h0, w0 = int(h[0]), int(w[0])
            pred = pred[0, :, :h0, :w0]         # 裁回原始尺寸，对齐原版 val_fn
            y = jt.array(mask).unsqueeze(1)[0, :, :h0, :w0]
            preds = jt.sigmoid(pred)
            iou_metric.update(preds, y)
            nIoU_metric.update(preds, y)
            fa_pd_metric.update(preds, y)
    _, mIoU = iou_metric.get()
    _, nIoU = nIoU_metric.get()
    FA, PD = fa_pd_metric.get(len(val_ds))
    dt = time.time() - t0
    logger.log(f'  [val]   epoch {epoch + 1}: mIoU={mIoU:.4f} nIoU={nIoU:.4f} '
               f'PD={PD:.4f} FA(x1e6)={FA * 1e6:.2f} 耗时 {dt:.1f}s')
    return float(mIoU), float(nIoU), float(PD), float(FA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(MODELS.keys()), default='ACM')
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--start_epoch', type=int, default=0)
    ap.add_argument('--pal_total_epochs', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--init_from', type=str, default='')
    ap.add_argument('--pal_workspace', type=str, default='')
    ap.add_argument('--save_dir', type=str, default='')
    ap.add_argument('--limit_init', type=int, default=0)
    ap.add_argument('--limit_train', type=int, default=0)
    ap.add_argument('--limit_val', type=int, default=0)
    ap.add_argument('--guard', action='store_true',
                    help='PAL-Guard 抗塌缩保护 (默认关闭; 关闭时路径与现版本一致)')
    ap.add_argument('--guard_exit_iou', type=float, default=0.05,
                    help='Guard 退出阈值: train_IoU 超过该值才计入退出连击 (默认 0.05)')
    ap.add_argument('--guard_exit_patience', type=int, default=3,
                    help='Guard 退出连击: 连续 N 轮超阈值才退出全保护 (默认 3)')
    ap.add_argument('--guard_blend_epochs', type=int, default=10,
                    help='Guard 退出后 blend 退火轮数, λ: 1->0 线性 (默认 10; 0=硬切)')
    ap.add_argument('--seed', type=int, default=None,
                    help='设置 random/np.random/jt 种子 (用于确定性回归)')
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        jt.seed(args.seed)

    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not args.pal_workspace:
        args.pal_workspace = os.path.join(
            ROOT, 'pal_workspaces', f'{args.model}__SIRST3__masks_coarse__{ts}')
    if not args.save_dir:
        args.save_dir = os.path.join(ROOT, 'work_dirs', f'pal_{args.model}_{ts}')
    make_dir(args.save_dir)
    logger = TeeLogger(os.path.join(args.save_dir, 'pal_train.log'))

    logger.log('=' * 70)
    logger.log(f'PAL 三阶段训练 ({args.model}, Jittor)')
    logger.log(f'args: {vars(args)}')
    logger.log(f'jittor {jt.__version__}, use_cuda={jt.flags.use_cuda}')

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

    model = MODELS[args.model][0](**MODELS[args.model][1])
    # 多输出约定 (ACM/ALC 为 None, 单输出模型不会进入任何 tuple 分支, 行为不变)
    out_index, loss_mode = MODELS[args.model][2] or (-1, 'mean_all')
    n_param = sum(int(np.prod(p.shape)) for p in model.parameters())
    logger.log(f'{args.model} 参数量: {n_param / 1e6:.3f} M')
    if args.init_from:
        model.load_state_dict(jt.load(args.init_from))
        logger.log(f'热启动: {args.init_from}')
    optimizer = jt.nn.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    val_ds = SirstDataset(val_img, val_mask, patch_size=None,
                          transform=build_val_transform(cal_mean, cal_std),
                          mode='val')
    val_ds.set_attrs(batch_size=1, shuffle=False, num_workers=0,
                     drop_last=False, keep_numpy_array=True)
    if args.limit_val > 0:
        val_ds.images = val_ds.images[:args.limit_val]

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

    guard = GuardController(enabled=args.guard,
                            exit_th=args.guard_exit_iou,
                            exit_patience=args.guard_exit_patience,
                            blend_epochs=args.guard_blend_epochs,
                            log_fn=logger.log)
    if args.guard:
        logger.log(f'PAL-Guard 已开启 (v2): 最近5轮 train_IoU<0.005 且 epoch>=40 '
                   f'-> 平衡BCE(λ=1); train_IoU>{args.guard_exit_iou} '
                   f'连续{args.guard_exit_patience}轮 -> blend '
                   f'{args.guard_blend_epochs}轮线性退火恢复 edgeSCE; '
                   f'blend 期再塌缩 λ 立即回 1')
    guard_lam = None

    for epoch in range(args.start_epoch, args.epochs):
        t_ep = time.time()
        logger.log(f'--- epoch {epoch + 1}/{args.epochs} (PAL 调度基准 {sched}) ---')

        if epoch > enh_lo and epoch <= enh_hi:
            if epoch % CLEAR_EPOCH_GAP == 0:
                lose_point_ratio = LOSE_POINT_RATIO_INIT + \
                    (epoch - sched * CLEAR_INITAL_RATIO + 1) / \
                    (sched * (FINAL_EPOCH_RATIO - CLEAR_INITAL_RATIO)) * \
                    (1 - LOSE_POINT_RATIO_INIT)
                logger.log(f'  [增强轮] lose_point_ratio={lose_point_ratio:.4f}')
                logger.log('  开始标签的自更新--------')
                pal_label_self_update(model, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                      cal_mean, cal_std, logger, out_index)
                logger.log('  开始认识并学习困难样本-----------')
                pal_infer_no_choose(model, nc_img_dir, nc_mask_dir,
                                    cal_mean, cal_std, logger, out_index)
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
            logger.log('  [精炼轮] 开始标签的自更新--------')
            pal_label_self_update(model, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                  cal_mean, cal_std, logger, out_index)

        train_ds = build_pool_datasets(ws, cal_mean, cal_std, args)
        if args.limit_train > 0:
            train_ds.images = train_ds.images[:args.limit_train]
        if len(train_ds) == 0:
            logger.log('  [警告] choose 池为空，跳过本 epoch 训练')
            continue
        _, tIoU, _ = train_one_epoch_guard(train_ds, model, optimizer, epoch,
                                           logger, guard_lambda=guard_lam,
                                           loss_mode=loss_mode)
        # 用本 epoch 的 train_IoU 更新 Guard 状态机 (切换时打 [GUARD] 日志),
        # 返回值即下一轮应使用的 λ (None=纯 edgeSCE, 1.0=纯平衡BCE, 其间=blend)
        guard_lam = guard.update(epoch, tIoU)

        mIoU, nIoU, PD, FA = val_one_epoch_ds(val_ds, model, epoch, logger,
                                              out_index=out_index)
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
