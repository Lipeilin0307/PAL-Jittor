# -*- coding: utf-8 -*-
"""
数据加载辅助（PAL 原版 components/utils_all_edge_copy_paste_final_2_img_path.py
的 Jittor 迁移版）。

差异说明：
1. 删除死 import torch / torchvision（原版 L1-2，从未使用）；
2. torch DataLoader → jittor Dataset.set_attrs。jittor 没有独立的 DataLoader
   类，数据集本身即迭代器；
3. 默认 num_workers=0（Windows 下 jittor 多进程 DataLoader 有坑，见
   workspace/probe_jt_workers.py 结论）；
4. 默认 drop_last=False，与 torch DataLoader 默认行为对齐（jittor 默认 True，
   不显式设置会静默丢尾批）；
5. 默认 keep_numpy_array=True：迭代产出 numpy 批次，由训练循环自行
   `jt.array(...)` 转换。原因：本机实测 jittor 1.3.8.5 (Windows) 在
   keep_numpy_array=False 时，对 val 模式返回的 int 标量字段（h, w）的自动
   转换会产生垃圾值（首个完整 batch 必现）。
"""
import os

try:
    from data.sirst3_dataset import SirstDataset
except ImportError:  # 允许在 data/ 目录内直接运行
    from sirst3_dataset import SirstDataset


def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def get_datasets(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    patch_size,
    train_batch_size,
    test_batch_size,
    train_transform,
    val_transform,
    num_workers=0,
    shuffle_train=True,
    drop_last=False,
    keep_numpy_array=True,
):
    """构建 train/val 数据集并配置好批处理属性，返回值即迭代器。

    对应原版 get_loaders；迭代产出见 data/sirst3_dataset.py 模块 docstring。
    """
    train_ds = SirstDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        patch_size=patch_size,
        transform=train_transform,
        mode='train',
    )
    train_ds.set_attrs(
        batch_size=train_batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=drop_last,
        keep_numpy_array=keep_numpy_array,
    )

    val_ds = SirstDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        patch_size=patch_size,
        transform=val_transform,
        mode='val',
    )
    val_ds.set_attrs(
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        keep_numpy_array=keep_numpy_array,
    )

    return train_ds, val_ds
