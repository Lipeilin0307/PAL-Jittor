# -*- coding: utf-8 -*-
"""
边缘标签生成（PAL 原版 components/edges.py 的纯 numpy 迁移版）。

与 PyTorch 原版的差异（均为去依赖，无数值语义改动）：
1. 删除死 import torch（原文件从未使用）；
2. `from scipy.ndimage.morphology import distance_transform_edt` 改为
   `from scipy.ndimage import distance_transform_edt` —— 同一个函数，
   morphology 命名空间在 scipy>=1.8 已弃用、scipy 2.0 将移除。
"""
from scipy.ndimage import distance_transform_edt
import numpy as np


def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)
    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])
    for i in range(num_classes):
        # 提取轮廓
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = (edgemap > 0).astype(np.uint8)*255
    return edgemap


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == (i) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)
