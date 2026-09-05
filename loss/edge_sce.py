# -*- coding: utf-8 -*-
"""edgeSCE_loss 的 Jittor 迁移版。
源: PAL/loss/Edge_loss.py (torch, 41 行)。

组成(与原版逐步对应):
1. SMP SoftBCEWithLogitsLoss(smooth_factor=None, reduction='none')
   == 逐像素 BCE-with-logits(无标签平滑)。已在 torch 侧实测:
   SMP 输出与 F.binary_cross_entropy_with_logits(reduction='none') max diff = 0。
   Jittor 内置 binary_cross_entropy_with_logits 无 reduction 参数(默认取均值),
   故用数值稳定形式手写: max(x,0) - x*z + log(1+exp(-|x|))。
2. edge 加权: 原代码为
       edge[edge == 0] = 1.
       edge[edge > 0] = edge_weight(=4)
   注意第二行会把第一行刚写入的 1 也覆盖(1>0), 因此实际效果是
   **所有像素权重均为 4**(原版的 quirk, edge 图不起区分作用)。
   为保证与原版逐位一致, 此处忠实复刻两步覆盖语义, 不做"修正"。
3. OHEM: 逐像素 loss 展平升序排序, 取索引 int(0.5*numel) 处值为阈值,
   保留 >= 阈值的像素求均值(即最大 50% 像素均值)。
   torch 用布尔索引 loss[loss>=min].mean() == sum(选中)/count(选中),
   这里用乘法掩码实现, 数学上完全等价。
"""
import jittor as jt


def edgeSCE_loss(pred, target, edge):
    # 1) 逐像素 BCE-with-logits (数值稳定形式, 等价 SMP SoftBCEWithLogitsLoss)
    loss_sce = jt.maximum(pred, 0) - pred * target + jt.log(1 + jt.exp(-jt.abs(pred)))

    # 2) 忠实复刻原版两步覆盖: edge==0 -> 1, 然后 edge>0 -> 4 (实际全为 4)
    edge_w = jt.ternary(edge == 0, jt.ones_like(edge), edge)
    edge_w = jt.ternary(edge_w > 0, jt.ones_like(edge_w) * 4.0, edge_w)
    loss_sce = loss_sce * edge_w

    # 3) OHEM: 排序取最大 50% 像素求均值
    flat = loss_sce.reshape(-1)
    sorted_v, _ = jt.sort(flat)
    n = flat.shape[0]
    min_value = sorted_v[int(0.5 * n)]
    mask = (loss_sce >= min_value).float32()
    loss = (loss_sce * mask).sum() / mask.sum()
    return loss


def edge_sce_loss_guard(pred, target, edge=None, pos_weight=None):
    """PAL-Guard 塌缩保护损失: 逐像素平衡 BCE-with-logits (无 OHEM)。

    全背景塌缩态下 (train_IoU 恒 0), edgeSCE 的 top-50% OHEM 让 ~0.1% 的
    前景像素在难例 mining 中被背景碾压, 每 epoch 前景 logit 推进极小。
    本损失对正像素乘 pos_weight 直接抵消类别不均衡, 帮助逃离塌缩。

    与原版 edge 加权 quirk 的关系: 原版全像素实际 x4, 常数缩放不改变
    梯度方向 (等价于调 lr), 这里按已定案语义**不乘 4**, 即纯 BCE。

    参数:
        pred:       模型原始输出 logits [N,1,H,W] (未过 sigmoid)
        target:     0/1 浮点标签, 与 pred 同形
        edge:       忽略 (仅为与 edgeSCE_loss 同签名, 便于无缝切换)
        pos_weight: 正像素权重; None 时按 batch 动态计算
                    min(n_neg / max(n_pos, 1), 1000)
                    (cap 1000 防空 batch / 极端不均衡)

    返回:
        标量 loss = (bce * w).mean(),
        w = 1 + (pos_weight - 1) * 1{target > 0.5}
        与 torch F.binary_cross_entropy_with_logits(pos_weight=p,
        reduction='mean') 语义对齐。全背景 batch 时 w 恒 1, 仍给全图梯度。
    """
    del edge  # 占位, 不使用
    t = (target > 0.5).float32()
    # 数值稳定的逐像素 BCE-with-logits: max(z,0) - z*t + log(1+exp(-|z|))
    bce = jt.maximum(pred, 0) - pred * t + jt.log(1 + jt.exp(-jt.abs(pred)))
    if pos_weight is None:
        n_pos = float(t.sum().item())
        n_neg = float((1.0 - t).sum().item())
        pos_weight = min(n_neg / max(n_pos, 1.0), 1000.0)
    w = 1.0 + (float(pos_weight) - 1.0) * t
    return (bce * w).mean()
