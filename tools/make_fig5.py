# -*- coding: utf-8 -*-
"""make_fig5.py — 从 docs/logs/ 下 4 份 Jittor PAL 训练日志提取每 epoch 的
train loss_mean 与 val mIoU，绘制 2x2 双子轴 Loss/指标曲线（fig5）。

用法（仓库根目录）:
    python tools/make_fig5.py
输出:
    assets/figures/fig5_loss_curves.png
"""
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()

import matplotlib.pyplot as plt

# 与 fig1~fig4 一致的低饱和暖色系
C_LOSS = "#C17C5E"   # 赭（loss）
C_MIOU = "#7A9E7E"   # 灰绿（val mIoU）
C_DARK = "#4A4039"

RUNS = [
    ("ACM",       "docs/logs/acm_train_jittor_run2.log",   0.4712, 254, 94),
    ("ALCNet",    "docs/logs/alcnet_train_jittor_run1.log", 0.5057, 361, 93),
    ("SCTransNet", "docs/logs/sct_train_jittor_run1.log",  0.7052, 313, 18),
    ("ISNet",     "docs/logs/isnet_train_jittor_run1.log", 0.5998, 389, 57),
]

re_train = re.compile(r"\[train\] epoch (\d+): loss_mean=([\d.eE+-]+)")
re_val = re.compile(r"\[val\]\s+epoch (\d+): mIoU=([\d.eE+-]+)")


def parse_log(path):
    ep_loss, ep_miou = {}, {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re_train.search(line)
        if m:
            ep_loss[int(m.group(1))] = float(m.group(2))
            continue
        m = re_val.search(line)
        if m:
            ep_miou[int(m.group(1))] = float(m.group(2))
    eps = sorted(set(ep_loss) & set(ep_miou))
    return eps, [ep_loss[e] for e in eps], [ep_miou[e] for e in eps]


def smooth(y, w=5):
    y = np.asarray(y, dtype=float)
    return np.convolve(y, np.ones(w) / w, mode="valid")


def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="#D9D4CE", linewidth=0.8, alpha=0.9)
    for s in ("top",):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#55504A")
    for s in ("left", "bottom", "right"):
        ax.spines[s].set_color("#B8B2AA")


fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
for ax, (name, rel, best, best_ep, esc_ep) in zip(axes.flat, RUNS):
    eps, loss, miou = parse_log(ROOT / rel)
    ax2 = ax.twinx()

    # 左轴：train loss_mean（动态范围跨 3 个数量级，用对数坐标）
    ax.plot(eps, loss, color=C_LOSS, alpha=0.25, linewidth=0.8)
    sm = smooth(loss, 5)
    ax.plot(np.arange(eps[0], eps[0] + len(sm)), sm, color=C_LOSS, linewidth=2.0,
            label="train loss_mean（左轴，log）")
    ax.set_yscale("log")
    ax.set_ylabel("train loss_mean（对数）", color=C_LOSS, fontsize=10)
    ax.tick_params(axis="y", colors=C_LOSS)

    # 右轴：val mIoU
    ax2.plot(eps, miou, color=C_MIOU, alpha=0.25, linewidth=0.8)
    sm2 = smooth(miou, 5)
    ax2.plot(np.arange(eps[0], eps[0] + len(sm2)), sm2, color=C_MIOU, linewidth=2.0,
             label="val mIoU（右轴）")
    ax2.set_ylabel("val mIoU", color=C_MIOU, fontsize=10)
    ax2.tick_params(axis="y", colors=C_MIOU)
    ax2.spines["top"].set_visible(False)

    # 逃离塌缩 epoch 与 best 标注
    ax.axvline(esc_ep, color=C_DARK, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.annotate(f"逃离 ep{esc_ep}", (esc_ep, ax.get_ylim()[0]), textcoords="offset points",
                xytext=(4, 4), fontsize=8.5, color=C_DARK, fontweight="bold")
    ax2.plot([best_ep], [best], marker="*", markersize="11", color=C_MIOU,
             markeredgecolor=C_DARK, linewidth=0, zorder=5)

    style_ax(ax)
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_title(f"{name}（best mIoU={best:.4f} @ ep{best_ep}，逃离 ep{esc_ep}）",
                 fontsize=11.5, color=C_DARK, fontweight="bold", pad=6)

    # 合并双轴图例
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5, framealpha=0.9)

fig.suptitle("图 5  四个网络 PAL 三阶段训练过程：train loss 与 val mIoU 曲线（Jittor，SIRST3 + masks_coarse，400 epoch）",
             fontsize=12.5, color=C_DARK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "assets/figures/fig5_loss_curves.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print("saved:", out)
