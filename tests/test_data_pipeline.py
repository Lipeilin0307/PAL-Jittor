# -*- coding: utf-8 -*-
"""
PAL 数据管线 Jittor 迁移版单元测试（用 jittor 环境 python 运行）：

  D:/Anaconda/envs/jittor/python.exe PAL_jittor/tests/test_data_pipeline.py

验收项：
  T1  train Dataset 从 SIRST3 origin/img + masks_coarse 加载，批次形状/取值正确
  T2  val Dataset 全量 1079 张加载不报错，pad-32 契约正确
  T3  cal_mean_std 与 PyTorch 侧结果一致（误差 < 1e-6）
  T4  mIoU/nIoU/PD/FA 指标与 PyTorch 侧逐位一致；edges.py 边缘生成一致；
      指标类可直接吃 jittor.Var
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

PAL_JITTOR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PAL_JITTOR_ROOT)

import jittor as jt
jt.flags.use_cuda = 1

from data.sirst3_dataset import SirstDataset, build_train_transform, build_val_transform
from data.cal_mean_std import Calculate_mean_std
from data.edges import mask_to_onehot, onehot_to_binary_edges
from metrics.metric import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA_2

SIRST3_ROOT = r'C:/Users/Alienware/Documents/kimi/workspace/PAL/dataset/SIRST3'
ORIGIN_IMG = os.path.join(SIRST3_ROOT, 'origin', 'img')
ORIGIN_COARSE = os.path.join(SIRST3_ROOT, 'origin', 'masks_coarse')
VAL_IMG = os.path.join(SIRST3_ROOT, 'val', 'img')
VAL_MASK = os.path.join(SIRST3_ROOT, 'val', 'mask')
TORCH_PY = r'D:/Anaconda/envs/pal_torch/python.exe'
REF_SCRIPT = os.path.join(PAL_JITTOR_ROOT, 'tests', 'ref_torch_parity.py')

PASSED = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    PASSED.append((name, cond, detail))
    print(f'  [{status}] {name}' + (f' | {detail}' if detail else ''))
    if not cond:
        raise AssertionError(f'{name} 失败: {detail}')


def make_metric_fixture(tmpdir):
    """构造确定的 pred/label（含小目标 blob），保证 IoU/PD/FA 都有意义。"""
    rng = np.random.RandomState(42)
    # 3D [B,H,W]，与 PAL 实际评测调用形态一致（train_model.py 评测时 preds/y 均为
    # [1,h,w] 3D）；skimage measure.label 仅支持 1-3 维输入
    label = np.zeros((4, 64, 64), dtype=np.float32)
    pred = np.zeros((4, 64, 64), dtype=np.float32)
    # 每个样本两个标签 blob；pred 一个命中（质心距<3）、一个漏检、一个虚警
    for b in range(4):
        label[b, 10:14, 10:14] = 1.0
        label[b, 40:44, 40:44] = 1.0
        pred[b, 11:15, 11:15] = 0.9          # 命中（质心偏移 ~1.4 < 3）
        pred[b, 50:53, 20:23] = 0.8          # 虚警
        noise = rng.rand(64, 64).astype(np.float32) * 0.3
        pred[b] += noise
    np.savez(os.path.join(tmpdir, 'metric.npz'), pred=pred, label=label)
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[10:14, 10:14] = 1
    mask[40:46, 40:43] = 1
    np.save(os.path.join(tmpdir, 'edge_mask.npy'), mask)
    return pred, label, mask


def main():
    print('=' * 70)
    print('PAL 数据管线 Jittor 迁移版 · 单元测试')
    print('=' * 70)

    # ---------- T3 前置：Jittor 侧均值方差（T1 的 transform 也要用） ----------
    print('\n[T3] cal_mean_std 计算（Jittor 侧，1676 张，需等待）...')
    jt_mean, jt_std = Calculate_mean_std(ORIGIN_IMG)

    # ---------- T1：train Dataset ----------
    print('\n[T1] train Dataset 批次形状与取值检查（origin/img + masks_coarse）')
    train_tf = build_train_transform(jt_mean, jt_std)
    train_ds = SirstDataset(ORIGIN_IMG, ORIGIN_COARSE, patch_size=256,
                            transform=train_tf, mode='train')
    check('T1.0 train 样本数 == 1676', len(train_ds) == 1676, f'len={len(train_ds)}')
    train_ds.set_attrs(batch_size=4, shuffle=False, num_workers=0,
                       drop_last=False, keep_numpy_array=True)
    n_batch = 0
    for img, mask, edge in train_ds:
        b = img.shape[0]
        check(f'T1.1 batch{n_batch} img 形状', img.shape == (b, 3, 256, 256),
              f'img.shape={img.shape} dtype={img.dtype}')
        check(f'T1.1 batch{n_batch} mask 形状', mask.shape == (b, 256, 256),
              f'mask.shape={mask.shape} dtype={mask.dtype}')
        check(f'T1.1 batch{n_batch} edge 形状', edge.shape == (b, 1, 256, 256),
              f'edge.shape={edge.shape} dtype={edge.dtype}')
        check(f'T1.2 batch{n_batch} dtype 全为 float32',
              img.dtype == np.float32 and mask.dtype == np.float32 and edge.dtype == np.float32)
        uniq_m = np.unique(mask)
        check(f'T1.3 batch{n_batch} mask ∈ {{0,1}}',
              set(uniq_m.tolist()) <= {0.0, 1.0}, f'unique={uniq_m.tolist()}')
        check(f'T1.4 batch{n_batch} edge ≥ 0 且取值 ⊆ {{0,255}}',
              bool(edge.min() >= 0) and set(np.unique(edge).tolist()) <= {0.0, 255.0},
              f'edge unique={np.unique(edge).tolist()}')
        check(f'T1.5 batch{n_batch} img 已归一化(大致零均值尺度)',
              bool(abs(img.mean()) < 5.0 and img.std() < 5.0),
              f'mean={img.mean():.4f} std={img.std():.4f}')
        n_batch += 1
        if n_batch >= 2:
            break
    # jt.array 转换冒烟（训练循环里的实际用法）
    img_v, mask_v, edge_v = jt.array(img), jt.array(mask).unsqueeze(1), jt.array(edge)
    check('T1.6 jt.array 转换后形状', tuple(img_v.shape) == (img.shape[0], 3, 256, 256)
          and tuple(mask_v.shape) == (img.shape[0], 1, 256, 256)
          and tuple(edge_v.shape) == (img.shape[0], 1, 256, 256),
          f'img{tuple(img_v.shape)} mask{tuple(mask_v.shape)} edge{tuple(edge_v.shape)}')

    # ---------- T2：val Dataset 全量加载 ----------
    print('\n[T2] val Dataset 全量加载（1079 张，需等待）...')
    val_tf = build_val_transform(jt_mean, jt_std)
    val_ds = SirstDataset(VAL_IMG, VAL_MASK, patch_size=None,
                          transform=val_tf, mode='val')
    check('T2.0 val 样本数 == 1079', len(val_ds) == 1079, f'len={len(val_ds)}')
    val_ds.set_attrs(batch_size=1, shuffle=False, num_workers=0,
                     drop_last=False, keep_numpy_array=True)
    cnt = 0
    for img, mask, h, w in val_ds:
        h0, w0 = int(h[0]), int(w[0])
        hh, ww = img.shape[-2], img.shape[-1]
        assert img.ndim == 4 and img.shape[0] == 1 and img.shape[1] == 3, img.shape
        assert hh % 32 == 0 and ww % 32 == 0, (hh, ww)
        assert h0 <= hh and w0 <= ww and hh - h0 < 32 and ww - w0 < 32, (h0, hh, w0, ww)
        uniq_m = np.unique(mask)
        assert set(uniq_m.tolist()) <= {0.0, 1.0}, uniq_m
        cnt += 1
        if cnt % 400 == 0:
            print(f'  ... 已加载 {cnt}/1079')
    check('T2.1 val 全量迭代完成 1079 个 batch', cnt == 1079, f'cnt={cnt}')
    check('T2.2 val h/w 标量批次取值正确（keep_numpy_array 规避 jittor 标量转换 bug）',
          h0 > 0 and w0 > 0, f'末张 h={h0} w={w0}')

    # ---------- T3/T4：与 PyTorch 侧一致性对比 ----------
    tmpdir = tempfile.mkdtemp(prefix='pal_parity_', dir=os.path.dirname(os.path.abspath(__file__)))
    pred, label, edge_mask = make_metric_fixture(tmpdir)
    ref_json = os.path.join(tmpdir, 'ref.json')
    print('\n[T3/T4] 调用 PyTorch 参照环境计算对照值...')
    subprocess.run(
        [TORCH_PY, REF_SCRIPT, '--img_dir', ORIGIN_IMG,
         '--metric_npz', os.path.join(tmpdir, 'metric.npz'),
         '--edge_npy', os.path.join(tmpdir, 'edge_mask.npy'),
         '--out', ref_json],
        check=True, cwd=os.path.dirname(PAL_JITTOR_ROOT))
    with open(ref_json, 'r', encoding='utf-8') as f:
        ref = json.load(f)

    print('\n[T3] cal_mean_std 一致性')
    check('T3.1 mean 误差 < 1e-6', abs(jt_mean - ref['mean']) < 1e-6,
          f"jittor={jt_mean!r} torch={ref['mean']!r} diff={abs(jt_mean - ref['mean']):.3e}")
    check('T3.2 std 误差 < 1e-6', abs(jt_std - ref['std']) < 1e-6,
          f"jittor={jt_std!r} torch={ref['std']!r} diff={abs(jt_std - ref['std']):.3e}")

    print('\n[T4] 指标一致性（同一构造输入，双环境各算一次）')
    m1 = SigmoidMetric(); m1.update(pred, label)
    pix_acc, miou = m1.get()
    m2 = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5); m2.update(pred, label)
    iou_arr, niou = m2.get()
    m3 = PD_FA_2(1); m3.update(pred, label)
    fa, pd = m3.get(pred.shape[0])

    check('T4.1 mIoU 逐位一致', float(miou) == ref['miou'], f'jittor={float(miou)!r} torch={ref["miou"]!r}')
    check('T4.2 pixAcc 逐位一致', float(pix_acc) == ref['pix_acc'],
          f'jittor={float(pix_acc)!r} torch={ref["pix_acc"]!r}')
    check('T4.3 nIoU 逐位一致', float(niou) == ref['niou'], f'jittor={float(niou)!r} torch={ref["niou"]!r}')
    check('T4.4 逐样本 IoU 逐位一致',
          all(float(a) == b for a, b in zip(iou_arr, ref['iou_arr'])) and len(iou_arr) == len(ref['iou_arr']),
          f'jittor={[float(v) for v in iou_arr]} torch={ref["iou_arr"]}')
    check('T4.5 PD 逐位一致', float(pd) == ref['pd'], f'jittor={float(pd)!r} torch={ref["pd"]!r}')
    check('T4.6 FA 逐位一致', float(fa) == ref['fa'], f'jittor={float(fa)!r} torch={ref["fa"]!r}')

    edge_jt = onehot_to_binary_edges(mask_to_onehot(edge_mask.astype(np.int64), 2), 1, 2)
    check('T4.7 edges.py 边缘生成一致（加权和指纹）',
          float(edge_jt.sum()) == ref['edge_sum'] and
          float(np.sum(edge_jt.astype(np.float64) * np.arange(edge_jt.size, dtype=np.float64).reshape(edge_jt.shape))) == ref['edge_hash'],
          f"sum: jittor={float(edge_jt.sum())!r} torch={ref['edge_sum']!r}")

    # 指标类直接吃 jittor.Var
    m1v = SigmoidMetric(); m1v.update(jt.array(pred), jt.array(label))
    _, miou_v = m1v.get()
    m3v = PD_FA_2(1); m3v.update(jt.array(pred), jt.array(label))
    fav, pdv = m3v.get(pred.shape[0])
    check('T4.8 指标类接受 jittor.Var 输入且结果一致',
          float(miou_v) == float(miou) and float(fav) == float(fa) and float(pdv) == float(pd),
          f'mIoU(Var)={float(miou_v)!r}')

    # ---------- 验收小结 ----------
    print('\n' + '=' * 70)
    print('「数据管线迁移完成」验收小结')
    print('=' * 70)
    n_pass = sum(1 for _, c, _ in PASSED if c)
    print(f'通过检查点：{n_pass}/{len(PASSED)}')
    print(f'cal_mean_std: mean={jt_mean:.8f}, std={jt_std:.8f}（与 PyTorch 侧 diff '
          f'{abs(jt_mean - ref["mean"]):.2e} / {abs(jt_std - ref["std"]):.2e}）')
    print(f'指标对照: mIoU={float(miou)!r}, nIoU={float(niou)!r}, PD={float(pd)!r}, FA={float(fa)!r}（与 PyTorch 侧逐位一致）')
    print(f'train 批次契约: img [B,3,256,256] fp32 | mask [B,256,256] fp32 ∈{{0,1}} | edge [B,1,256,256] fp32 ∈{{0,255}}')
    print(f'val 全量加载: {cnt}/1079 张 OK，pad-32 契约 OK')
    print('结论: 数据管线（Dataset/edges/cal_mean_std）与评测指标（mIoU/nIoU/PD/FA）')
    print('      已迁移至 PAL_jittor/{data,metrics}，全部验收项通过。')
    print('=' * 70)


if __name__ == '__main__':
    main()
