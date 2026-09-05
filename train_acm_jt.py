# -*- coding: utf-8 -*-
"""PAL ACM 的 Jittor 训练脚本（W2-2 端到端冒烟联调）。

用法::

    D:/Anaconda/envs/jittor/python.exe PAL_jittor/train_acm_jt.py \
        --epochs 3 --batch_size 16 --lr 1e-3 [--limit_train N] [--limit_val N]

与 PAL 原版 train_model.py（ACM 分支）的语义对应：
- 数据：train = dataset/SIRST3/origin/img + origin/masks_coarse（点标签直接当监督，
  等价 PAL 预启动期 pre-start 形态：只用初始池、不做标签自更新/样本准入）；
  val = val/img + val/mask（pad-32 整图推理，裁回 [h,w] 后 sigmoid 进指标）。
- 训练步（对应原版 train_fn L508-512 的 else 分支）：
  targets.unsqueeze(1) -> pred = model(data) -> edgeSCE_loss(pred, targets, edge)
  -> backward -> AdamW step。edge 原样传入（{0,255}），取值处理在 loss 内部
  （与原版一致；且原版 loss 的 edge 加权因两行覆盖顺序实际为全像素 ×4，
  jittor 版 loss 已忠实复刻）。
- 优化器：AdamW(lr=1e-3, weight_decay=0.01) 恒定，无 scheduler
  （torch AdamW 默认 weight_decay=0.01；jittor 默认为 0，故显式传入）。
- 每 epoch 末 val，输出 mIoU / nIoU / PD / FA（FA 按原版约定 ×1e6 展示）；
  best mIoU 时保存 checkpoint 到 save_dir。

与原版差异清单（仅工程差异，数值语义一致）：
1. 不迁 AMP：首版全 fp32（原版 torch.cuda.amp.autocast + GradScaler）。
2. torch DataLoader -> jt.dataset.Dataset.set_attrs；keep_numpy_array=True
   （Windows 下 jittor 标量批转换有垃圾值 bug），循环内自行 jt.array 转换。
3. 原版 val 从第 2 个 epoch 才开始（num_start_test_epochs=1），本脚本每 epoch 末都 val。
4. 原版 train_fn 内同步累计 train IoU，本脚本保留该行为（仅日志展示）。
5. val loss 原版恒为 0（eval_losses=0 的 quirk），本脚本不计 val loss，只出四个指标。
6. checkpoint 只含 state_dict + 元信息（epoch/指标），不含 optimizer 状态
   （jt.save/jt.load 语义）。
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np
import jittor as jt

jt.flags.use_cuda = 1

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
PAL_ROOT = os.environ.get('PAL_ROOT', os.path.join(os.path.dirname(ROOT), 'PAL'))

from model.acm import ACM_No_Sigmoid
from loss.edge_sce import edgeSCE_loss
from data.sirst3_dataset import build_train_transform, build_val_transform
from data.cal_mean_std import Calculate_mean_std
from data.utils import get_datasets
from metrics.metric import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA_2

PATCH_SIZE = 256


class TeeLogger:
    """同时写 stdout 与日志文件。"""
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8')

    def log(self, msg):
        print(msg, flush=True)
        self.f.write(msg + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def train_one_epoch(train_ds, model, optimizer, epoch, logger):
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
        loss = edgeSCE_loss(pred, targets, edge_t)
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
    logger.log(f'  [train] epoch {epoch + 1}: loss_mean={np.mean(losses):.6f} '
               f'(min={np.min(losses):.4f}, max={np.max(losses):.4f}, batches={nb}) '
               f'train_IoU={tIoU:.4f} train_nIoU={tnIoU:.4f} 耗时 {dt:.1f}s')
    return float(np.mean(losses))


def val_one_epoch(val_ds, model, epoch, logger):
    model.eval()
    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    fa_pd_metric = PD_FA_2(1)
    t0 = time.time()
    with jt.no_grad():
        for img, mask, h, w in val_ds:
            x = jt.array(img)                   # [1,3,H',W'] pad-32
            pred = model(x)                     # ACM 整图前向（原版 test_pred else 分支）
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
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--save_dir', type=str, default=None)
    ap.add_argument('--limit_train', type=int, default=0, help='>0 时限制训练子集大小')
    ap.add_argument('--limit_val', type=int, default=0, help='>0 时限制验证子集大小')
    ap.add_argument('--init_from', type=str, default='',
                    help='可选：jittor 权重 pkl 路径（如 work_dirs/acm_official_jt.pkl），'
                         '用于热启动冒烟；为空则随机初始化从头训练')
    args = ap.parse_args()

    if args.save_dir is None:
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.save_dir = os.path.join(ROOT, 'work_dirs', f'smoke_{ts}')
    os.makedirs(args.save_dir, exist_ok=True)
    logger = TeeLogger(os.path.join(args.save_dir, 'smoke.log'))

    logger.log('=' * 70)
    logger.log('PAL ACM Jittor 冒烟训练')
    logger.log(f'args: {vars(args)}')
    logger.log(f'jittor {jt.__version__}, use_cuda={jt.flags.use_cuda}')

    # ---- 数据集 ----
    origin_img = os.path.join(PAL_ROOT, 'dataset/SIRST3/origin/img')
    origin_mask = os.path.join(PAL_ROOT, 'dataset/SIRST3/origin/masks_coarse')
    val_img = os.path.join(PAL_ROOT, 'dataset/SIRST3/val/img')
    val_mask = os.path.join(PAL_ROOT, 'dataset/SIRST3/val/mask')
    logger.log('计算数据集 mean/std（对应原版 cal_mean_std，每次启动现算）...')
    cal_mean, cal_std = Calculate_mean_std(origin_img)
    logger.log(f'cal_mean={cal_mean:.8f} cal_std={cal_std:.8f}')

    train_ds, val_ds = get_datasets(
        origin_img, origin_mask, val_img, val_mask,
        patch_size=PATCH_SIZE,
        train_batch_size=args.batch_size,
        test_batch_size=1,
        train_transform=build_train_transform(cal_mean, cal_std),
        val_transform=build_val_transform(cal_mean, cal_std),
        num_workers=0, keep_numpy_array=True,
    )
    if args.limit_train > 0:
        train_ds.images = train_ds.images[:args.limit_train]
    if args.limit_val > 0:
        val_ds.images = val_ds.images[:args.limit_val]
    logger.log(f'train 样本数={len(train_ds)}  val 样本数={len(val_ds)}  '
               f'bs={args.batch_size}')

    # ---- 模型 / 优化器 ----
    model = ACM_No_Sigmoid()
    n_param = sum(int(np.prod(p.shape)) for p in model.parameters())
    logger.log(f'ACM_No_Sigmoid 参数量: {n_param / 1e6:.3f} M')
    if args.init_from:
        model.load_state_dict(jt.load(args.init_from))
        logger.log(f'热启动: 已载入 {args.init_from}')
    optimizer = jt.nn.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ---- 训练 ----
    # 原版 best_mIoU=0 起步（mIoU 恒 0 时永不保存，从头冒烟易触发该边角）；
    # 这里置 -1 保证首个 epoch 必落盘 checkpoint，便于验收 ③。
    best_mIoU = -1.0
    loss_hist = []
    for epoch in range(args.epochs):
        t_ep = time.time()
        loss_mean = train_one_epoch(train_ds, model, optimizer, epoch, logger)
        loss_hist.append(loss_mean)
        if epoch == 0:
            jt.display_memory_info()
        mIoU, nIoU, PD, FA = val_one_epoch(val_ds, model, epoch, logger)
        if best_mIoU < mIoU:
            best_mIoU = mIoU
            ckpt = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mIoU': best_mIoU,
                'best_nIoU': nIoU,
                'best_PD': PD,
                'best_FA': FA,
            }
            ckpt_path = os.path.join(args.save_dir, 'best_mIoU_checkpoint.pkl')
            jt.save(ckpt, ckpt_path)
            logger.log(f'  [ckpt] 新的 best mIoU={best_mIoU:.4f}，已保存 -> {ckpt_path}')
        logger.log(f'  epoch {epoch + 1} 总耗时 {time.time() - t_ep:.1f}s')

    # ---- 收尾验证 ----
    logger.log('-' * 70)
    logger.log(f'loss 序列: {["%.6f" % v for v in loss_hist]}')
    logger.log(f'best mIoU: {best_mIoU:.4f}')
    ckpt_path = os.path.join(args.save_dir, 'best_mIoU_checkpoint.pkl')
    if os.path.exists(ckpt_path):
        ck = jt.load(ckpt_path)
        m2 = ACM_No_Sigmoid()
        m2.load_state_dict(ck['state_dict'])
        logger.log(f'checkpoint 回读验证 OK（epoch={ck["epoch"]}, keys={len(ck["state_dict"])}）')
    else:
        logger.log('警告: 未产生 checkpoint（mIoU 始终为 0）')
    logger.log('SMOKE_DONE')
    logger.close()


if __name__ == '__main__':
    main()
