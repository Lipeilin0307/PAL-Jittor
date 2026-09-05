# -*- coding: utf-8 -*-
"""
SIRST3 数据集（PAL 原版 components/dataset_final_edge_copy_paste_final_2_img_path.py
的 Jittor 迁移版）。

继承 `jittor.dataset.Dataset`，实现 `__len__` / `__getitem__`。

用法（对应原版 torch Dataset + DataLoader）::

    from data.sirst3_dataset import SirstDataset, build_train_transform, build_val_transform
    from data.cal_mean_std import Calculate_mean_std

    cal_mean, cal_std = Calculate_mean_std(origin_img_dir)

    train_ds = SirstDataset(train_img_dir, train_mask_dir, patch_size=256,
                            transform=build_train_transform(cal_mean, cal_std),
                            mode='train')
    # batch_size / shuffle 通过 set_attrs 设置（等价 torch DataLoader 参数）。
    # 必须 keep_numpy_array=True：__getitem__ 返回 numpy，由框架仅做批拼接不做
    # jt.array 转换（本机实测 jittor 1.3.8.5 Windows 版对小 int32 标量字段的
    # 自动 jt.array 转换会产生垃圾值，详见 tests 验收说明）；训练循环里自行
    # `jt.array(...)` 转张量。num_workers 首版设 0（Windows 多进程有坑）。
    train_ds.set_attrs(batch_size=16, shuffle=True, num_workers=0,
                       drop_last=False, keep_numpy_array=True)
    for img, mask, edge in train_ds:      # 全部是 numpy 批次
        img  = jt.array(img)              # [B,3,256,256] float32，已归一化
        mask = jt.array(mask).unsqueeze(1)  # [B,256,256] float32 {0,1}，对齐原版
                                            # train_fn 里 targets.unsqueeze(1) 的契约
        edge = jt.array(edge)             # [B,1,256,256] float32，取值 {0,255}

    val_ds = SirstDataset(val_img_dir, val_mask_dir, patch_size=None,
                          transform=build_val_transform(cal_mean, cal_std),
                          mode='val')
    val_ds.set_attrs(batch_size=1, shuffle=False, num_workers=0,
                     drop_last=False, keep_numpy_array=True)
    for img, mask, h, w in val_ds:        # img [1,3,H',W']（H',W' 已 pad 到 32 倍数）
        ...                                # h, w 为 [1] int32，原始未 pad 尺寸

与 PyTorch 原版的差异（数值语义全部保持一致）：
1. 去掉 ToTensorV2：__getitem__ 直接返回 numpy（img: float32 CHW 已归一化；
   mask: float32 {0,1} HW；edge: float32 {0,255} (1,H,W)），CHW 转置在本类内
   手工完成，训练循环里再 `jt.array(...)`。
2. 原版 train 分支在 transform=None 时会丢弃 random_crop 结果并在 mask.numpy()
   处崩溃（实际使用必然传 transform）；本版在 transform=None 时直接使用裁剪结果，
   属于健壮性修正，不改变有 transform 时的任何行为。
3. 原版 mask 经 ToTensorV2 后为 float64，本版统一 float32（{0,1} 值精确表示，
   无损）。
4. albumentations 增广管线、随机裁剪、EDT 边缘生成（含原版把第 1 行清零的
   `edge[1, :] = 0` 细节）原样保留。
"""
import math
import os
import random

import albumentations as A
import numpy as np
from PIL import Image
from jittor.dataset import Dataset

try:
    from data.edges import onehot_to_binary_edges, mask_to_onehot
except ImportError:  # 允许在 data/ 目录内直接运行
    from edges import onehot_to_binary_edges, mask_to_onehot


def random_crop(img, mask, patch_size):
    h, w, c = img.shape
    mh, mw = mask.shape

    assert (h, w) == (mh, mw), "Image and mask must have the same height and width"

    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w, _ = img.shape

    h_start = random.randint(0, h - patch_size)
    h_end = h_start + patch_size
    w_start = random.randint(0, w - patch_size)
    w_end = w_start + patch_size

    img_patch = img[h_start:h_end, w_start:w_end, :]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch


class SirstDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size, transform=None, mode='None'):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))
        self.mode = mode
        self.patch_size = patch_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127.5).astype(float)

        if (self.mode == 'train'):
            image_patch, mask_patch = random_crop(image, mask, self.patch_size)
            if self.transform is not None:
                augmentations = self.transform(image=image_patch, mask=mask_patch)
                image = augmentations["image"]
                mask = augmentations["mask"]
            else:
                # 原版此处会把裁剪结果丢弃（依赖必然存在的 transform）；
                # 本版直接使用裁剪结果，保证 transform=None 也可用。
                image = image_patch
                mask = mask_patch
            # albumentations Normalize 输出 float32 HWC；原版靠 ToTensorV2 转
            # CHW torch.Tensor，这里手工 transpose 成 float32 CHW numpy。
            image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
            mask = np.asarray(mask, dtype=np.float32)
            mask_2 = mask.astype(np.int64)
            oneHot_label = mask_to_onehot(mask_2, 2)
            edge = onehot_to_binary_edges(oneHot_label, 1, 2)
            edge[1, :] = 0
            edge[-1:, :] = 0
            edge[:, :1] = 0
            edge[:, -1:] = 0
            edge = np.expand_dims(edge, axis=0).astype(np.float32)
            return image, mask, edge

        elif (self.mode == 'val'):
            times = 32
            h, w, c = image.shape
            pad_height = math.ceil(h / times) * times - h
            pad_width = math.ceil(w / times) * times - w
            image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
            mask = np.asarray(mask, dtype=np.float32)
            return image, mask, h, w


def build_train_transform(cal_mean, cal_std):
    """训练增广管线，逐项对应 PAL train_model.py L560-584（去掉 ToTensorV2）。"""
    return A.Compose(
        [
            A.SomeOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightness(limit=0.3, p=0.2),
                A.RandomContrast(limit=0.3, p=0.2),
                A.Rotate(limit=45, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.2),
                A.NoOp(),
                A.NoOp(),
            ], 3, p=0.5),
            A.Normalize(
                mean=cal_mean,
                std=cal_std,
                max_pixel_value=255.0,
            ),
        ],
    )


def build_val_transform(cal_mean, cal_std):
    """验证/测试变换，对应 PAL train_model.py L585-607（去掉 ToTensorV2）。"""
    return A.Compose(
        [
            A.Normalize(
                mean=cal_mean,
                std=cal_std,
                max_pixel_value=255.0,
            ),
        ],
    )
