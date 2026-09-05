# -*- coding: utf-8 -*-
"""
全数据集灰度均值/方差现算（PAL 原版 components/cal_mean_std.py 原样拷贝）。

纯 numpy/PIL 实现，与深度学习框架无关，PyTorch 版与 Jittor 版计算结果应逐位一致。
注意：返回值是 mean/255、std/255（归一化到 [0,1] 尺度），直接供
albumentations.Normalize(mean=cal_mean, std=cal_std, max_pixel_value=255.0) 使用。
"""
import numpy as np
from PIL import Image
import os
#

def Calculate_mean_std(img_dir):
    img_list = os.listdir(img_dir)

    mean_list = []
    std_list = []

    for i in range(len(img_list)):
        #print(i)
        img_path = os.path.join(img_dir,img_list[i])
        img = np.array(Image.open(img_path).convert("L"))
        mean_list.append(img.mean())
        std_list.append(img.std())
    mean_out = np.mean(mean_list)/255
    std_out = np.mean(std_list)/255
    print("路径为：", img_dir)
    print("数据集均值为：", mean_out)
    print("数据集方差为：", std_out)

    return mean_out, std_out
