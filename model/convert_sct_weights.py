# -*- coding: utf-8 -*-
"""torch SCTransNet 权重 -> Jittor SCTransNet 权重转换器 (在 jittor 环境运行)。

用法:
    python convert_sct_weights.py <torch_npz> <out_jt_pkl>

流程:
1) 读取 export_sct_torch_refs.py 导出的 sct_torch_init.npz (torch state_dict 的 numpy 形式)
2) 键名映射: 迁移版刻意保持模块属性名与 torch 完全一致(含死参数 position_embeddings 与
   16 个 q*_attn* 标量), 故 1:1 直映; 仅跳过 torch BN 特有的 num_batches_tracked。
   LayerNorm(手写 WithBias/BiasFree) 的 weight/bias、InstanceNorm2d(affine=False 无参数)、
   eca 的 Conv1d weight 均按同名同 shape 覆盖校验。
3) 逐一 shape 校验后 load 进 SCTransNet_No_Sigmoid, jt.save 落盘。
"""
import os
import sys

import numpy as np
import jittor as jt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.sct import SCTransNet_No_Sigmoid

SKIP_SUFFIX = '.num_batches_tracked'


def convert(npz_path, out_path=None, mode='train', verbose=True):
    data = np.load(npz_path)
    torch_sd = {k: data[k] for k in data.files}

    model = SCTransNet_No_Sigmoid(mode=mode)
    jt_sd = model.state_dict()

    n_loaded, n_skipped = 0, 0
    missing = set(jt_sd.keys())
    for k, v in torch_sd.items():
        if k.endswith(SKIP_SUFFIX):
            n_skipped += 1
            continue
        if k not in jt_sd:
            raise KeyError(f'torch 键在 jittor 模型中不存在: {k}')
        if tuple(jt_sd[k].shape) != tuple(v.shape):
            raise ValueError(f'shape 不匹配 {k}: torch {v.shape} vs jittor {tuple(jt_sd[k].shape)}')
        jt_sd[k].update(jt.array(v))
        missing.discard(k)
        n_loaded += 1

    if missing:
        raise RuntimeError(f'jittor 参数未被 torch 权重覆盖: {sorted(missing)}')

    if verbose:
        print(f'[convert] 载入 {n_loaded} 个张量, 跳过 num_batches_tracked x {n_skipped}, '
              f'jittor 参数全覆盖 ✓')
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        jt.save(model.state_dict(), out_path)
        print(f'[convert] jittor 权重已保存: {out_path}')
    return model


if __name__ == '__main__':
    npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'data',
        'sct_torch_init.npz')
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'work_dirs',
        'sct_torch_init_jt.pkl')
    jt.flags.use_cuda = 1
    convert(npz, out)
    print('CONVERT_DONE')
