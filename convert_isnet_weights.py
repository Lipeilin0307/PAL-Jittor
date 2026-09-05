# -*- coding: utf-8 -*-
"""torch ISNet 权重 -> Jittor ISNet 权重转换器 (在 jittor 环境运行)。

用法:
    python convert_isnet_weights.py <torch_npz> <out_jt_pkl>

说明:
1) 读取 export_isnet_torch_refs.py 导出的 isnet_torch_init.npz (torch state_dict 的 numpy 形式)
2) 键名映射: 迁移版刻意保持模块属性名与 torch 完全一致, 包括:
   - DCN: dcn_row/dcn_colum.{weight,bias,conv_offset_mask.weight/bias} 1:1
     (offset/mask 预测层零初始化已被导出脚本非平凡化, 此处按值直灌)
   - GatedSpatialConv2d: {weight, _gate_conv.0..5.*} 1:1 (去 _ConvNd 继承后键名不变)
   - register_buffer: grad_extractor.{weight_v,weight_h} 1:1
   仅跳过 torch BN 特有的 num_batches_tracked。
3) 逐层打印映射表 + shape 校验, 全覆盖后 jt.save 落盘。
"""
import os
import sys

import numpy as np
import jittor as jt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.isnet import ISNet_No_Sigmoid

SKIP_SUFFIX = '.num_batches_tracked'


def convert(npz_path, out_path=None, verbose=True):
    data = np.load(npz_path)
    torch_sd = {k: data[k] for k in data.files}

    model = ISNet_No_Sigmoid()
    jt_sd = model.state_dict()

    n_loaded, n_skipped = 0, 0
    missing = set(jt_sd.keys())
    for k in sorted(torch_sd.keys()):
        v = torch_sd[k]
        if k.endswith(SKIP_SUFFIX):
            n_skipped += 1
            continue
        if k not in jt_sd:
            raise KeyError(f'torch 键在 jittor 模型中不存在: {k}')
        if tuple(jt_sd[k].shape) != tuple(v.shape):
            raise ValueError(f'shape 不匹配 {k}: torch {v.shape} vs jittor {tuple(jt_sd[k].shape)}')
        jt_sd[k].update(jt.array(v))
        missing.discard(k)
        if verbose:
            print(f'  {k} -> {k}  {tuple(v.shape)}')
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
        os.path.dirname(os.path.abspath(__file__)), 'tests', 'data',
        'isnet_torch_init.npz')
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'work_dirs',
        'isnet_torch_init_jt.pkl')
    jt.flags.use_cuda = 1
    convert(npz, out, verbose=False)
    print('CONVERT_DONE')
