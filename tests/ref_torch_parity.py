# -*- coding: utf-8 -*-
"""
PyTorch 侧参照脚本（用 D:/Anaconda/envs/pal_torch/python.exe 运行）。

调用 PAL 原版仓库代码（components/cal_mean_std.py、components/metric_new_crop.py、
components/edges.py）计算参照值，输出 JSON 供 Jittor 侧单元测试做一致性对比。

用法：
  python ref_torch_parity.py --img_dir <SIRST3/origin/img> \
      --metric_npz <含 pred/label 的 npz> --edge_npy <mask npy> --out <json 路径>
"""
import argparse
import json
import os
import sys

PAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', '..', 'PAL'))
sys.path.insert(0, PAL_ROOT)

import numpy as np
import torch

from components.cal_mean_std import Calculate_mean_std
from components.edges import mask_to_onehot, onehot_to_binary_edges
from components.metric_new_crop import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA_2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--metric_npz', required=True)
    ap.add_argument('--edge_npy', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    result = {}

    # 1. 均值方差
    mean_out, std_out = Calculate_mean_std(args.img_dir)
    result['mean'] = float(mean_out)
    result['std'] = float(std_out)

    # 2. 指标（输入转 torch.Tensor，走原版 update）
    data = np.load(args.metric_npz)
    pred = torch.from_numpy(data['pred'].astype(np.float32))
    label = torch.from_numpy(data['label'].astype(np.float32))

    m1 = SigmoidMetric()
    m1.update(pred, label)
    pix_acc, miou = m1.get()
    result['pix_acc'] = float(pix_acc)
    result['miou'] = float(miou)

    m2 = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)
    m2.update(pred, label)
    iou_arr, niou = m2.get()
    result['iou_arr'] = [float(v) for v in iou_arr]
    result['niou'] = float(niou)

    m3 = PD_FA_2(1)
    m3.update(pred, label)
    fa, pd = m3.get(pred.shape[0])
    result['fa'] = float(fa)
    result['pd'] = float(pd)

    # 3. 边缘生成
    mask = np.load(args.edge_npy).astype(np.int64)
    edge = onehot_to_binary_edges(mask_to_onehot(mask, 2), 1, 2)
    result['edge_sum'] = float(edge.sum())
    result['edge_hash'] = float(np.sum(edge.astype(np.float64) *
                                       np.arange(edge.size, dtype=np.float64).reshape(edge.shape)))

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('[ref_torch_parity] written:', args.out)


if __name__ == '__main__':
    main()
