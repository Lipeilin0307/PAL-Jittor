# -*- coding: utf-8 -*-
"""PAL 渐进式主动学习机制的 Jittor 迁移版（框架无关部分）。

源: PAL/utilts.py（400 行）。本文件包含全部 7 个机制函数：
  make_dir / center_point_inside_contour / process_image /
  data_inital_make_add_points / nc_pred_mask / deal_pred_mask_and_true_point_in /
  nc_correct_pred_mask / deal_gen_mask_error_aera / move_files / hard_sample_in /
  update_gt_update_degen_corr

迁移原则（纪律）：cv2/numpy/shutil 数值逻辑**逐行保留**——Canny(20,40)、
3x3 膨胀腐蚀核、面积 50px 判失败、0.8 点成功率门槛、漏检率/过检警报公式、
33x33/3x3 膨胀邻域、自适应阈值公式、0.97 背景衰减、np.maximum 保原始点。

与原版唯一差异：
1. 删除 torch import；`update_gt_update_degen_corr` 中
   `threshold.cpu().numpy() if isinstance(threshold, torch.Tensor)` 改为
   鸭子类型判断（jittor 侧传入的 pred 本就是 numpy，该分支不触发；
   若上游误传 torch/jittor 张量也能正确转 numpy）。
2. 删除死 import（importlib/random/pydensecrf 注释）与 access_model/check_path
   这两个训练脚本辅助函数（train 脚本内联实现）。
模型推理耦合点不在本文件（原版就在 train_model.py 的 PAL 循环里），
由 train_pal_acm_jt.py 以注入式推理函数实现。
"""

import os
import shutil

import cv2
import numpy as np


def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def center_point_inside_contour(center_point, target_mask):
    (center_y, center_x) = center_point
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_index_XmergeY = {center_y * 1.0 + center_x * 0.0001}
    temp_contour_mask = np.zeros(target_mask.shape, np.uint8)

    overlap_found = False
    for target_contour in target_contours:
        target_contour_mask = np.zeros(target_mask.shape, np.uint8)
        cv2.fillPoly(target_contour_mask, [target_contour], (255))
        target_index = np.where(target_contour_mask == 255)
        target_index_XmergeY = set(target_index[0] * 1.0 + target_index[1] * 0.0001)
        if not center_index_XmergeY.isdisjoint(target_index_XmergeY):
            area = cv2.contourArea(target_contour)
            if area > 50:
                break
            else:
                overlap_found = True
                cv2.fillPoly(temp_contour_mask, [target_contour], (255))
            break

    if not overlap_found:
        temp_contour_mask = temp_contour_mask
    return temp_contour_mask, overlap_found


def process_image(y_and_x, y1_and_y2_and_x1_and_x2, img_shape, image, low_threshold=50, high_threshold=150, kernel_size=(3, 3), sigma=0):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    high_pass = cv2.subtract(image, blurred_image)
    edges = cv2.Canny(high_pass, low_threshold, high_threshold)
    kernel = np.ones((3, 3), np.uint8)
    sparse_edges = cv2.dilate(edges, kernel, iterations=1)
    sparse_edges = cv2.erode(sparse_edges, kernel, iterations=1)

    temp_contour_mask_2 = np.zeros(img_shape, np.uint8)
    (y1, y2, x1, x2) = y1_and_y2_and_x1_and_x2
    temp_contour_mask_2[y1:y2, x1:x2] = sparse_edges
    (y, x) = y_and_x

    refine_mask, flag = center_point_inside_contour((y, x), temp_contour_mask_2)
    refine_mask_out = refine_mask[y1:y2, x1:x2]
    return refine_mask_out, flag


def data_inital_make_add_points(origin_img_dir, origin_points_dir, TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_points_dir, nc_img_dir, nc_mask_dir, nc_points_dir, crop_size=10):
    input_image_path = origin_img_dir
    input_points_path = origin_points_dir

    output_image_path = TRAIN_IMG_DIR
    output_masks_path = TRAIN_MASK_DIR
    output_points_path = train_points_dir

    no_choose_output_image_path = nc_img_dir
    no_choose_output_masks_path = nc_mask_dir
    no_choose_output_points_path = nc_points_dir

    input_img_list = os.listdir(input_image_path)

    for i in range(len(input_img_list)):
        img_path = os.path.join(input_image_path, input_img_list[i])
        points_path = os.path.join(input_points_path, input_img_list[i])

        out_img_path = os.path.join(output_image_path, input_img_list[i])
        out_mask_path = os.path.join(output_masks_path, input_img_list[i])
        out_points_path = os.path.join(output_points_path, input_img_list[i])

        no_choose_out_img_path = os.path.join(no_choose_output_image_path, input_img_list[i])
        no_choose_out_mask_path = os.path.join(no_choose_output_masks_path, input_img_list[i])
        no_choose_out_points_path = os.path.join(no_choose_output_points_path, input_img_list[i])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(points_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"图像或mask读取失败：{input_img_list[i]}")
            continue

        points = np.where(mask == 255)
        if len(points[0]) == 0:
            cv2.imwrite(out_img_path, img)
            cv2.imwrite(out_mask_path, mask)
            cv2.imwrite(out_points_path, mask)
            continue

        merged_result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        correct_point = 0

        for i in range(len(points[0])):
            center_y, center_x = points[0][i], points[1][i]

            half_size = crop_size
            x1 = max(center_x - half_size, 0)
            y1 = max(center_y - half_size, 0)
            x2 = min(center_x + half_size, img.shape[1])
            y2 = min(center_y + half_size, img.shape[0])

            roi = img[y1:y2, x1:x2]

            processed_roi, flag = process_image((center_y, center_x), (y1, y2, x1, x2), mask.shape, roi,
                                                low_threshold=20, high_threshold=40, kernel_size=(3, 3), sigma=0)

            merged_result[y1:y2, x1:x2] = merged_result[y1:y2, x1:x2] + processed_roi

            if flag == True:
                correct_point = correct_point + 1
            else:
                continue

        if (correct_point / len(points[0])) >= 0.8:
            merged_result = merged_result + mask
            merged_result = np.where(merged_result > 0, 255, 0)
            cv2.imwrite(out_img_path, img)
            cv2.imwrite(out_mask_path, merged_result)
            cv2.imwrite(out_points_path, mask)
        else:
            cv2.imwrite(no_choose_out_img_path, img)
            cv2.imwrite(no_choose_out_points_path, mask)

    print("初始数据已生成，共生成样本张数：", len(os.listdir(output_image_path)))


###copy_mask 为点
###target_mask为预测结果
def nc_pred_mask(copy_mask, target_mask, lose_point_ratio=0.2, alarm_point_ration=0.2):
    copy_contours, _ = cv2.findContours(copy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overwrite_contours = []
    un_overwrite_contours = []

    target_index_sets = []
    for target_contour in target_contours:
        target_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(target_contour_mask, [target_contour], (255))
        target_index = np.where(target_contour_mask == 255)
        target_index_XmergeY = set(target_index[0] * 1.0 + target_index[1] * 0.0001)
        target_index_sets.append(target_index_XmergeY)

    for copy_contour in copy_contours:
        copy_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(copy_contour_mask, [copy_contour], (255))
        copy_index = np.where(copy_contour_mask == 255)
        copy_index_XmergeY = set(copy_index[0] * 1.0 + copy_index[1] * 0.0001)

        overlap_found = False
        for target_index_XmergeY in target_index_sets:
            if not copy_index_XmergeY.isdisjoint(target_index_XmergeY):
                overwrite_contours.append(copy_contour)
                overlap_found = True
                break

        if not overlap_found:
            un_overwrite_contours.append(copy_contour)

    flag = False
    if len(un_overwrite_contours) / len(copy_contours) > lose_point_ratio or ((len(target_contours) - len(overwrite_contours)) / len(copy_contours)) > alarm_point_ration:
        flag = flag
    else:
        flag = True
    return flag


def deal_pred_mask_and_true_point_in(nc_img_path, nc_pred_mask_path, nc_points_path, c_img_path, c_mask_path, c_points_path, lose_point_ratio=0.2, alarm_point_ration=0.2):
    nc_img_list = os.listdir(nc_img_path)
    new_choose_list = []
    for i in range(len(nc_img_list)):
        img_path = os.path.join(nc_img_path, nc_img_list[i])
        points_path = os.path.join(nc_points_path, nc_img_list[i])
        pred_path = os.path.join(nc_pred_mask_path, nc_img_list[i])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(points_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"图像或mask读取失败：{nc_img_list[i]}")
            continue

        points = np.where(mask == 255)
        if len(points[0]) == 0:
            new_choose_list.append(nc_img_list[i])
            continue

        flag = nc_pred_mask(mask, pred, lose_point_ratio=lose_point_ratio, alarm_point_ration=alarm_point_ration)
        if flag == True:
            new_choose_list.append(nc_img_list[i])
    return new_choose_list


###注意这边与nc_pred_mask中的相反
###copy_mask 为预测结果
###target_mask为点
def nc_correct_pred_mask(copy_mask, target_mask):
    copy_contours, _ = cv2.findContours(copy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overwrite_contours = []
    un_overwrite_contours = []

    target_index_sets = []
    for target_contour in target_contours:
        target_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(target_contour_mask, [target_contour], (255))
        target_index = np.where(target_contour_mask == 255)
        target_index_XmergeY = set(target_index[0] * 1.0 + target_index[1] * 0.0001)
        target_index_sets.append(target_index_XmergeY)

    for copy_contour in copy_contours:
        copy_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(copy_contour_mask, [copy_contour], (255))
        copy_index = np.where(copy_contour_mask == 255)
        copy_index_XmergeY = set(copy_index[0] * 1.0 + copy_index[1] * 0.0001)

        overlap_found = False
        for target_index_XmergeY in target_index_sets:
            if not copy_index_XmergeY.isdisjoint(target_index_XmergeY):
                overwrite_contours.append(copy_contour)
                overlap_found = True
                break

        if not overlap_found:
            un_overwrite_contours.append(copy_contour)

    copy_contour_mask_out = np.zeros(copy_mask.shape, np.uint8)
    for i in range(len(overwrite_contours)):
        cv2.fillPoly(copy_contour_mask_out, [overwrite_contours[i]], (255))

    copy_contour_mask_out = copy_contour_mask_out + target_mask
    copy_contour_mask_out = np.where(copy_contour_mask_out > 0, 255, 0)

    return copy_contour_mask_out


def deal_gen_mask_error_aera(nc_pred_mask_path, nc_points_path, new_choose_list):
    for i in range(len(new_choose_list)):
        pred_mask_path = os.path.join(nc_pred_mask_path, new_choose_list[i])
        points_path = os.path.join(nc_points_path, new_choose_list[i])

        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        points = cv2.imread(points_path, cv2.IMREAD_GRAYSCALE)

        pred_mask_out = nc_correct_pred_mask(pred_mask, points)
        cv2.imwrite(pred_mask_path, pred_mask_out)
    print("生成的标签精细化完成！！！！！！")


def move_files(file_list, src_path, dst_path):
    for file_name in file_list:
        src_file = os.path.join(src_path, file_name)
        dst_file = os.path.join(dst_path, file_name)
        if os.path.exists(src_file):
            try:
                shutil.copy(src_file, dst_file)
                os.remove(src_file)
            except PermissionError as e:
                print(f"PermissionError: {e}")
        else:
            print(f"File {src_file} does not exist.")


def hard_sample_in(nc_img_path, nc_pred_mask_path, nc_points_path, c_img_path, c_mask_path, c_points_path, new_choose_list):
    move_files(new_choose_list, nc_img_path, c_img_path)
    move_files(new_choose_list, nc_pred_mask_path, c_mask_path)
    move_files(new_choose_list, nc_points_path, c_points_path)


def update_gt_update_degen_corr(pred, gt_masks, thresh_Tb, thresh_k, size, degen=0.9):

    update_gt_masks = gt_masks.copy()

    background_length = 33
    target_length = 3

    num_labels, label_image = cv2.connectedComponents((gt_masks > 0.5).astype(np.uint8))

    background_kernel = np.ones((background_length, background_length), np.uint8)
    target_kernel = np.ones((target_length, target_length), np.uint8)

    pred_max = pred.max()
    max_limitation = size[0] * size[1] * 0.0015

    combined_thresh_mask = np.zeros_like(pred, dtype=np.float32)

    for region_num in range(1, num_labels):
        region_coords = np.argwhere(label_image == region_num)
        centroid = np.mean(region_coords, axis=0).astype(int)

        cur_point_mask = np.zeros_like(pred, dtype=np.uint8)
        cur_point_mask[centroid[0], centroid[1]] = 1

        nbr_mask = cv2.dilate(cur_point_mask, background_kernel) > 0
        targets_mask = cv2.dilate(cur_point_mask, target_kernel) > 0

        region_size_ratio = len(region_coords) / max_limitation
        threshold_start = (pred * nbr_mask).max() * thresh_Tb
        threshold_delta = thresh_k * ((pred * nbr_mask).max() - threshold_start) * region_size_ratio
        threshold = threshold_start + threshold_delta
        # 原版为 isinstance(threshold, torch.Tensor) 时 .cpu().numpy()；
        # jittor 侧 pred 为 numpy，threshold 本就是标量，此分支不触发。
        if hasattr(threshold, 'cpu'):
            threshold = threshold.cpu().numpy()
        elif hasattr(threshold, 'numpy'):
            threshold = threshold.numpy()

        thresh_mask = (pred * nbr_mask > threshold).astype(np.float32)

        num_labels_thresh, label_image_thresh = cv2.connectedComponents(thresh_mask.astype(np.uint8))
        for num_cur in range(1, num_labels_thresh):
            curr_mask = (label_image_thresh == num_cur).astype(np.float32)
            if np.sum(curr_mask * targets_mask) == 0:
                thresh_mask -= curr_mask

        combined_thresh_mask = np.maximum(combined_thresh_mask, thresh_mask)

    target_patch = (update_gt_masks * combined_thresh_mask + pred * combined_thresh_mask) / 2
    background_patch = update_gt_masks * (1 - combined_thresh_mask) * degen
    update_gt_masks = background_patch + target_patch

    update_gt_masks = np.maximum(update_gt_masks, (gt_masks == 1).astype(np.float32))

    return update_gt_masks
