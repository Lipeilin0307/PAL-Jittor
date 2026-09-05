# -*- coding: utf-8 -*-
"""PAL 机制迁移验收测试（jittor 环境运行；torch 参考经子进程调 pal_torch）。

  # 分开运行（每个节控制在数分钟内）:
  python test_pal_mechanism.py --t1   # 初始池确定性比对（默认用已生成产物；--regen 重新生成）
  python test_pal_mechanism.py --t2   # 单轮增强周期小场景 + 跨环境同输入同判定
  python test_pal_mechanism.py --t3   # 短程三阶段贯通（子进程跑 train_pal_acm_jt.py）
  python test_pal_mechanism.py        # 默认 T1+T2

T1: choose/no_choose 文件划分逐文件一致；blob mask/影像/点标签 逐字节一致。
T2: ①标签自更新后点数不减、数值合理 ②update_gt_update_degen_corr 与 torch 原版
    同输入输出逐点一致 ③准入判定 new_choose_list 两版一致 ④精细化+文件迁移正确。
T3: 预启动→第一轮增强完整链路无异常，日志含各阶段动作。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.dirname(HERE)
PAL_ROOT = os.path.join(os.path.dirname(JT_ROOT), 'PAL')
sys.path.insert(0, JT_ROOT)

JT_PY = r'D:/Anaconda/envs/jittor/python.exe'
TORCH_PY = r'D:/Anaconda/envs/pal_torch/python.exe'

FAILS = []


def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        FAILS.append(name)


# ============================= T1 =============================
def t1(regen=False):
    print('== T1 初始池确定性比对 ==')
    ref = os.path.join(HERE, 'pal_t1/ref')
    jtd = os.path.join(HERE, 'pal_t1/jt')
    if regen or not os.path.isdir(os.path.join(ref, 'train/choose/img')):
        for impl, py, out in [('orig', TORCH_PY, ref), ('jt', JT_PY, jtd)]:
            print(f'  生成 {impl} 初始池 ...')
            subprocess.run([py, os.path.join(HERE, 't1_run_initial_pool.py'),
                            '--impl', impl, '--out', out], check=True)
    for split in ['choose', 'no_choose']:
        for sub in ['img', 'mask', 'points', 'mask_pred']:
            d1, d2 = os.path.join(ref, 'train', split, sub), os.path.join(jtd, 'train', split, sub)
            l1 = sorted(os.listdir(d1)) if os.path.isdir(d1) else []
            l2 = sorted(os.listdir(d2)) if os.path.isdir(d2) else []
            check(f'T1 {split}/{sub} 文件清单一致', l1 == l2, f'(ref={len(l1)} jt={len(l2)})')
    n_same = n_tot = 0
    worst = 1.0
    for split, subs in [('choose', ['img', 'mask', 'points']), ('no_choose', ['img', 'points'])]:
        for sub in subs:
            d1, d2 = os.path.join(ref, 'train', split, sub), os.path.join(jtd, 'train', split, sub)
            for fn in sorted(os.listdir(d1)):
                with open(os.path.join(d1, fn), 'rb') as f1, open(os.path.join(d2, fn), 'rb') as f2:
                    b_same = f1.read() == f2.read()
                a = cv2.imread(os.path.join(d1, fn), cv2.IMREAD_GRAYSCALE)
                b = cv2.imread(os.path.join(d2, fn), cv2.IMREAD_GRAYSCALE)
                n_tot += 1
                if b_same and np.array_equal(a, b):
                    n_same += 1
                else:
                    inter = np.logical_and(a > 127, b > 127).sum()
                    union = np.logical_or(a > 127, b > 127).sum()
                    worst = min(worst, inter / max(union, 1))
    check('T1 全部文件逐字节一致', n_same == n_tot,
          f'({n_same}/{n_tot} 字节一致, 最差 IoU={worst})')


# ============================= T2 =============================
def t2():
    print('== T2 单轮增强周期（小场景）==')
    import jittor as jt
    jt.flags.use_cuda = 1
    from pal.pal_utils import (data_inital_make_add_points, update_gt_update_degen_corr,
                               deal_pred_mask_and_true_point_in,
                               deal_gen_mask_error_aera, hard_sample_in)
    from model.acm import ACM_No_Sigmoid
    from train_pal_acm_jt import pal_infer_one

    ws = os.path.join(HERE, 'pal_t2/ws')
    snap = os.path.join(HERE, 'pal_t2/snap')
    if os.path.isdir(ws):
        shutil.rmtree(ws)
    if os.path.isdir(snap):
        shutil.rmtree(snap)
    for d in ['train/choose/img', 'train/choose/mask', 'train/choose/points',
              'train/no_choose/img', 'train/no_choose/mask_pred', 'train/no_choose/points']:
        os.makedirs(os.path.join(ws, d), exist_ok=True)

    # 1) 小初始池（48 张硬链子集：24 张来自 T1 choose + 24 张来自 T1 no_choose，
    #    保证 no_choose 池足够厚，准入判定比对才有意义）
    origin_img = os.path.join(PAL_ROOT, 'dataset/SIRST3/origin/img')
    origin_pts = os.path.join(PAL_ROOT, 'dataset/SIRST3/origin/masks_coarse')
    t1c = sorted(os.listdir(os.path.join(HERE, 'pal_t1/ref/train/choose/img')))
    t1n = sorted(os.listdir(os.path.join(HERE, 'pal_t1/ref/train/no_choose/img')))
    subset = t1c[:24] + t1n[:24]
    sub_img = os.path.join(ws, '_origin_sub/img')
    sub_pts = os.path.join(ws, '_origin_sub/masks_coarse')
    os.makedirs(sub_img); os.makedirs(sub_pts)
    for n in subset:
        for s, d in [(origin_img, sub_img), (origin_pts, sub_pts)]:
            os.link(os.path.join(s, n), os.path.join(d, n))
    data_inital_make_add_points(sub_img, sub_pts,
                                os.path.join(ws, 'train/choose/img'),
                                os.path.join(ws, 'train/choose/mask'),
                                os.path.join(ws, 'train/choose/points'),
                                os.path.join(ws, 'train/no_choose/img'),
                                os.path.join(ws, 'train/no_choose/mask_pred'),
                                os.path.join(ws, 'train/no_choose/points'), crop_size=10)
    n_c = len(os.listdir(os.path.join(ws, 'train/choose/img')))
    n_nc = len(os.listdir(os.path.join(ws, 'train/no_choose/img')))
    print(f'  小初始池: choose={n_c} no_choose={n_nc}')
    check('T2 初始池非空且两侧都有样本', n_c > 0 and n_nc > 0, f'({n_c}/{n_nc})')

    mean, std = 0.38625179, 0.14124445
    model = ACM_No_Sigmoid()
    model.load_state_dict(jt.load(os.path.join(JT_ROOT, 'work_dirs/acm_official_jt.pkl')))
    model.eval()

    # 2) 标签自更新（与 train_pal 同逻辑，另存 npy 供跨环境比对）
    ug_dir = os.path.join(snap, 'update_gt')
    os.makedirs(ug_dir)
    c_img = os.path.join(ws, 'train/choose/img')
    c_mask = os.path.join(ws, 'train/choose/mask')
    prev_bin = {}
    for idx, name in enumerate(sorted(os.listdir(c_img))):
        mask_path = os.path.join(c_mask, name)
        prev = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
        prev_bin[name] = (prev > 0.5)
        pred, h, w = pal_infer_one(model, os.path.join(c_img, name), mean, std)
        if idx < 5:  # 存 5 对做跨环境逐点比对
            np.save(os.path.join(ug_dir, f'pred_{name[:-4]}.npy'), pred)
            np.save(os.path.join(ug_dir, f'prev_{name[:-4]}.npy'),
                    prev.astype(np.float32))
        cur = update_gt_update_degen_corr(pred, prev, 0.5, 0.5, [h, w], degen=0.97)
        cv2.imwrite(mask_path, cur * 255)
    # 2a) 点数不减 + 数值范围
    ok_preserve = ok_range = True
    for name in prev_bin:
        new = cv2.imread(os.path.join(c_mask, name), cv2.IMREAD_GRAYSCALE) / 255
        ok_preserve &= bool((prev_bin[name] <= (new > 0.5)).all())
        ok_range &= new.min() >= 0 and new.max() <= 1.0
    check('T2 标签自更新后原有点/blob 像素全部保留（点数不减）', ok_preserve)
    check('T2 自更新标签数值在 [0,1]', ok_range)

    # 3) no_choose 池推理 -> mask_pred
    nc_img = os.path.join(ws, 'train/no_choose/img')
    nc_pred = os.path.join(ws, 'train/no_choose/mask_pred')
    nc_pts = os.path.join(ws, 'train/no_choose/points')
    for name in sorted(os.listdir(nc_img)):
        pred, h, w = pal_infer_one(model, os.path.join(nc_img, name), mean, std)
        pred = cv2.resize(pred, (int(w), int(h)))
        cv2.imwrite(os.path.join(nc_pred, name), np.where(pred > 0.5, 255, 0))
    n_pred = len(os.listdir(nc_pred))
    check('T2 no_choose 推理 mask_pred 全写出', n_pred == n_nc, f'({n_pred}/{n_nc})')

    # 4) 准入判定（jittor 侧）—— epoch=85, sched=400 -> lose_point_ratio=0.22
    lpr = 0.2 + (85 - 400 * 0.2 + 1) / (400 * 0.6) * 0.8
    ad_dir = os.path.join(snap, 'admit')
    for sub in ['img', 'mask_pred', 'points']:
        shutil.copytree(os.path.join(ws, 'train/no_choose', sub),
                        os.path.join(ad_dir, sub))
    jt_list = sorted(deal_pred_mask_and_true_point_in(
        nc_img, nc_pred, nc_pts, c_img, c_mask,
        os.path.join(ws, 'train/choose/points'),
        lose_point_ratio=lpr, alarm_point_ration=5))
    print(f'  jittor 侧准入 {len(jt_list)}/{n_nc} 张 (lose_point_ratio={lpr:.4f})')

    # 5) torch 原版参考（子进程）
    r = subprocess.run([TORCH_PY, os.path.join(HERE, 't2_torch_ref.py'),
                        '--snap', snap, '--lose_point_ratio', str(lpr)],
                       capture_output=True)
    if r.returncode != 0:
        print((r.stdout or b'')[-2000:].decode('utf-8', errors='replace'))
        print((r.stderr or b'')[-2000:].decode('utf-8', errors='replace'))
    check('T2 torch 参考子进程成功', r.returncode == 0)

    # 5a) update_gt 逐点比对
    worst = 0.0
    for fn in sorted(os.listdir(ug_dir)):
        if fn.startswith('pred_'):
            key = fn[5:-4]
            pred = np.load(os.path.join(ug_dir, fn))
            prev = np.load(os.path.join(ug_dir, f'prev_{key}.npy'))
            mine = update_gt_update_degen_corr(pred, prev, 0.5, 0.5,
                                               list(pred.shape), degen=0.97)
            ref = np.load(os.path.join(snap, 'ref/update_gt', f'out_{key}.npy'))
            worst = max(worst, float(np.abs(mine - ref).max()))
    check('T2 update_gt 与 torch 原版同输入逐点一致', worst < 1e-6,
          f'(max abs diff={worst:.2e})')

    # 5b) 准入清单一致
    ref_list = json.load(open(os.path.join(snap, 'ref/admit_new_choose.json')))
    check('T2 准入判定 new_choose_list 两版一致', jt_list == ref_list,
          f'(jt={len(jt_list)} ref={len(ref_list)})')

    # 6) 精细化 + 迁移
    if len(jt_list) == 0:
        print('  [提示] 本轮准入为 0，用首张 no_choose 样本强制走通精细化/迁移路径')
        jt_list = sorted(os.listdir(nc_img))[:1]
    pts_before = {n: (cv2.imread(os.path.join(nc_pts, n), cv2.IMREAD_GRAYSCALE) > 127)
                  for n in jt_list}
    deal_gen_mask_error_aera(nc_pred, nc_pts, jt_list)
    ok_refine = True
    for n in jt_list:
        refined = cv2.imread(os.path.join(nc_pred, n), cv2.IMREAD_GRAYSCALE) > 127
        ok_refine &= bool((pts_before[n] <= refined).all())  # 原始点保留
    check('T2 精细化后原始点保留在伪标签中', ok_refine)

    hard_sample_in(nc_img, nc_pred, nc_pts, c_img, c_mask,
                   os.path.join(ws, 'train/choose/points'), jt_list)
    ok_mv = True
    for n in jt_list:
        ok_mv &= os.path.exists(os.path.join(c_img, n))
        ok_mv &= os.path.exists(os.path.join(c_mask, n))
        ok_mv &= os.path.exists(os.path.join(ws, 'train/choose/points', n))
        ok_mv &= not os.path.exists(os.path.join(nc_img, n))
        ok_mv &= not os.path.exists(os.path.join(nc_pred, n))
        ok_mv &= not os.path.exists(os.path.join(nc_pts, n))
    check('T2 文件迁移正确（choose 三件套齐 / no_choose 已清空）', ok_mv,
          f'(迁移 {len(jt_list)} 张)')
    print(f'  迁移后 choose={len(os.listdir(c_img))} no_choose={len(os.listdir(nc_img))}')


# ============================= T3 =============================
def t3():
    print('== T3 短程三阶段贯通（pal_total=100, 26 epoch, 首轮增强应在 epoch 26 触发）==')
    ws = os.path.join(HERE, 'pal_t3/ws')
    sd = os.path.join(HERE, 'pal_t3/save')
    if os.path.isdir(ws):
        shutil.rmtree(ws)
    if os.path.isdir(sd):
        shutil.rmtree(sd)
    r = subprocess.run([
        JT_PY, os.path.join(JT_ROOT, 'train_pal_acm_jt.py'),
        '--epochs', '26', '--pal_total_epochs', '100',
        '--init_from', os.path.join(JT_ROOT, 'work_dirs/acm_official_jt.pkl'),
        '--lr', '1e-5', '--pal_workspace', ws, '--save_dir', sd,
        '--limit_init', '128', '--limit_val', '16'],
        capture_output=True)
    out = r.stdout.decode('utf-8', errors='replace') if r.stdout else ''
    err = r.stderr.decode('utf-8', errors='replace') if r.stderr else ''
    print(out[-1500:])
    if r.returncode != 0:
        print(err[-2000:])
    check('T3 训练子进程正常结束', r.returncode == 0
          and 'PAL_TRAIN_DONE' in out)
    log = open(os.path.join(sd, 'pal_train.log'), encoding='utf-8').read()
    check('T3 日志含初始池生成', '初始池' in log or '生成初始池' in log)
    check('T3 日志含增强轮触发（epoch 26, 即调度索引 25）', '[增强轮]' in log)
    check('T3 日志含标签自更新/推理/迁移动作',
          ('自更新' in log) and ('认识并学习困难样本' in log) and ('数据转移完成' in log))
    check('T3 日志含每 epoch val 指标', '[val]' in log and 'mIoU=' in log)
    n_c = len(os.listdir(os.path.join(ws, 'train/choose/img')))
    n_nc = len(os.listdir(os.path.join(ws, 'train/no_choose/img')))
    print(f'  终态: choose={n_c} no_choose={n_nc}')
    check('T3 池文件系统完好', n_c + n_nc == 128, f'({n_c}+{n_nc})')
    check('T3 checkpoint 落盘', os.path.exists(os.path.join(sd, 'last_checkpoint.pkl')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--t1', action='store_true')
    ap.add_argument('--t2', action='store_true')
    ap.add_argument('--t3', action='store_true')
    ap.add_argument('--regen', action='store_true')
    args = ap.parse_args()
    none = not (args.t1 or args.t2 or args.t3)
    if args.t1 or none:
        t1(regen=args.regen)
    if args.t2 or none:
        t2()
    if args.t3:
        t3()
    print('=' * 60)
    if FAILS:
        print('未通过:', FAILS)
        sys.exit(1)
    print('PAL 机制验收测试全部通过 ✓')
