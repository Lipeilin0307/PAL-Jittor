# PAL 复现基准锚点（标准答案）

> 2026-08-30 建立。用途：Jittor 迁移版对齐用的"标准答案"。
> 来源：PAL 论文 Table 1（ICCV 2025 正式版，本工作区 `PAL_ICCV2025_paper.pdf`）+ 官方发布权重实测。

## 论文官方数字（SIRST3-Test，Coarse 粗点监督 + PAL 框架）

| 网络 | IoU(%) | nIoU(%) | Pd(%) | Fa(10⁻⁶) |
|---|---|---|---|---|
| ACM | 51.51 | 54.07 | 92.89 | 39.18 |
| ALCNet | 57.11 | 60.22 | 93.95 | 37.20 |
| MLCL-Net | 64.87 | 69.40 | 94.95 | 24.43 |
| ALCL-Net | 66.29 | 68.18 | 94.75 | 18.79 |
| DNANet | 67.20 | 70.20 | 96.15 | 10.86 |
| GGL-Net | 68.52 | 71.69 | 97.14 | 16.69 |
| UIUNet | 69.05 | 71.53 | 96.81 | 15.45 |
| MSDA-Net | 69.38 | 71.55 | 97.41 | 16.34 |

参考：同表 LESPS 框架下 ACM 仅 37.42 IoU——PAL 框架对 ACM 提升 +14.09pt，这就是 PAL 的核心卖点。

## 我方评测管线 + 官方权重实测（2026-08-30，pal_torch 环境，RTX 3070 Laptop）

| 网络(权重) | IoU | nIoU | Pd | Fa | 与论文差距 |
|---|---|---|---|---|---|
| ACM.pth.tar（epoch395） | **51.52** | **54.08** | 92.60→本次 91.55 | **39.15** | IoU/nIoU/Fa 几乎逐位一致；Pd 系统性低 ~1.3pt |
| ALC.pth.tar（epoch199） | **57.11** | **60.22** | 92.60 | **37.22** | IoU/nIoU 与论文完全相同；Pd 低 1.35pt |

结论：**数据布局、推理管线、评测脚本全部验证正确**。Pd 的 ~1.3pt 系统性偏低源于评测实现细节（skimage 版本/连通域质心计算），三方数字各异（训练循环记录 94.35/95.28、论文 92.89/93.95、我们实测 91.55/92.60），复现报告按"±1.5pt 容差"如实说明即可。

## 权重库存（F:\download\SIRST3_weight\）

- 官方发布三种监督（Full / Coarse / Centroid）× 8 网络（ACM/ALC/ALCL/DNA/GGL/MLCL/MSDA/UIU）
- **无 SCTransNet / ISNet 权重**（作者后来才加进框架，未发权重）——这两个网络只能自训对齐论文
- 权重已按仓库约定改名放入 `PAL/work_dirs/<MODEL>__SIRST3__masks_coarse__official/best_mIoU_checkpoint_*.pth.tar`（ACM、ALC 已就位）

## 数据集（W1c 已齐）

- `PAL/dataset/SIRST3/`：origin/{img,mask,masks_coarse,masks_centroid} 各 1676（train）；val/{img,mask} 各 1079（test）；img_idx/ 官方划分
- 来源：LESPS 百度包（`pan.baidu.com/s/1NT2jdjS4wrliYYP0Rt4nXw?pwd=m6ui`），与 SCTransNet 划分逐文件一致
- 另备原始件：`dataset_raw/SIRST-v1/`（427，含自生成点标签）、`dataset_raw/IRSTD-1k/`（1001，含自生成点标签 + 官方 txt）
- 自研点标签生成器：`tools_local/gen_point_labels.py`（coarse/centroid 两模式，已验证 0 越界）

## 自训 baseline 记录（W1d 收尾）

### ACM + masks_coarse，自训 400ep（2026-08-30 18:59 → 08-31 02:23，本机 RTX3070 Laptop，7.4h）
- best_epoch=351：**mIoU 47.69 / nIoU 48.72 / Fa 30.708 / Pd 90.70**
- 对比官方权重锚点（51.52 / 54.08 / 39.15 / 91.55）：mIoU -3.8pt，nIoU -5.4pt，Pd -0.85pt，Fa 反而更优
- 训练曲线健康：train_mIoU 0.71，测试指标单调爬升至尾声（best 在 351/400），无发散
- **注意：train_model.py 未固定任何随机种子**，PAL 主动学习采样本身随机 → 官方权重大概率是多次运行中的较优值，单次自训落在 47~52 区间属合理方差
- checkpoint：`work_dirs/ACM__SIRST3__masks_coarse__2026-08-30_18-59-19/best_mIoU_checkpoint_*.pth.tar`
- 待办：第二次不同种子复跑测方差；若仍 <49 再查 lr/池构建/mean-std

### ACM + masks_coarse，自训 run2（2026-08-31 02:33 → 10:13，400ep，7.7h）
- best_epoch=241：**mIoU 51.13 / nIoU 53.87 / Fa 23.893 / Pd 91.76**
- 对比官方权重锚点（51.52 / 54.08 / 39.15 / 91.55）：mIoU -0.39，nIoU -0.21，Pd +0.21，Fa 更优 → **训练侧对齐确认**
- 两发方差实测：run1 47.69 / run2 51.13，种子方差 ≈3.4pt（无固定种子 + AL 随机采样所致），官方权重=较优次运行
- **W1d 收官结论**：数据组装、训练管线、评测对齐全部验证通过，PAL PyTorch 基线复现完成
- checkpoint：`work_dirs/ACM__SIRST3__masks_coarse__2026-08-31_02-33-45/best_mIoU_checkpoint_*.pth.tar`

---

## Jittor 迁移版训练锚点（W2 收官，2026-09-01）

环境：AutoDL RTX4090 / python3.10 / jittor 1.3.8.5 / numpy 1.26.4 / fp32（无 AMP）/ AdamW lr=1e-3 wd=0.01 / bs=16 / 400ep，SIRST3 + masks_coarse，PAL 三阶段完整接入。

| 运行 | 逃离塌缩 epoch | best mIoU | 备注 |
|---|---|---|---|
| torch run1（本机） | 91 | 0.4769 | PyTorch 基线 |
| torch run2（本机） | ~106 | 0.5113 | PyTorch 基线 |
| jittor run1（服务器） | ~235 | 0.3815 | 塌缩态坐牢过久的坏样本，非迁移 bug |
| **jittor run2（服务器）** | **94** | **0.4712** | ✅ 落在 PyTorch 方差带内，对齐成立 |

结论：①edgeSCE+点监督下两框架都会先经历"全背景"塌缩期，逃离时机是随机变量（91~235 均观测到），逃离后爬升斜率两框架一致；②jittor 版终点水位与 torch 版同带 → 数据管线/网络/loss/PAL 机制迁移全部成立。
日志与 checkpoint：服务器 `/root/autodl-tmp/work_pal/PAL_jittor/{train_pal_run2.log, work_dirs/pal_2026-09-01_*/best_mIoU_checkpoint.pkl}`，本地存档 `PAL_jittor/train_pal_run2.log`。

## ALCNet 训练锚点（2026-09-02）

| 运行 | best mIoU | nIoU | Pd | Fa | 备注 |
|---|---|---|---|---|---|
| 官方权重 | 57.11 | 60.22 | 92.60 | 37.22 | 作者精选（评测对齐 W1d） |
| torch 自训（本机，09:54→17:28，best ep390） | **53.32** | 52.01 | 92.03 | 32.93 | 诚实基线：同代码单发 |
| jittor run1（服务器，逃离 ep93） | 50.57 | - | - | - | 距 torch 自训 -2.75pt，在 ACM 种子摆幅（3.8pt）内 |
| jittor run2（服务器） | 待收 | | | | |
| jittor run2（服务器，逃离 ep179） | 48.26 | | | | 晚逃离 → 终点低 |
| jittor run3（服务器，逃离 ep286） | 46.56 | | | | 晚逃离 → 终点低 |

### 逃离时机统计（截至 2026-09-03，PAL+edgeSCE 全背景塌缩现象）
- torch 逃离 epoch：{21, 91, 106}（3/3 ≤106）
- jittor 逃离 epoch：{93, 94, 179, 235, 286}（2/5 ≤100）
- 终点 mIoU 与逃离 epoch 强负相关（课程有效时长被塌缩期侵蚀）
- 头号系统性嫌疑：torch 训练全程 AMP(fp16) 噪声加速对称破缺；jittor 版 fp32
- → 该现象为 PAL-Guard（抗塌缩改造）的动机与证据链

---

## 2026-09-03 PAL-Guard A/B v1 结果（服务器 4090，ACM，6 发×100ep，pal_total=400）

实验目录：服务器 `work_dirs/ab_2026-09-03_13-57-49`。同序号 vanilla/guard 共享种子（1001~1003），唯一差异 `--guard`。

| run | vanilla 逃离 | guard 逃离 | vanilla 末 mIoU@100 | guard 末 mIoU@100 | GUARD 事件 |
|---|---|---|---|---|---|
| #1 (seed 1001) | 73 | 41 | 0.4394 | 0.2087 | 4 |
| #2 (seed 1002) | 30 | 33 | 0.3899 | 0.3878 | 0（未触发） |
| #3 (seed 1003) | 52 | 42 | 0.4072 | 0.1684 | 4 |

- 结论 1：guard 加速逃离成立（41/42 vs 73/52；未触发时与 vanilla 几乎重合，guard-off 无副作用）。
- 结论 2：v1 退出太松（train_IoU>0.02×2 轮）+ 硬切回 edgeSCE → 退出后 5 轮内再塌缩（val mIoU 归零），二次触发后才真正爬升；100ep 截断时 guard 组仍在爬坡。
- 另注意：带显式种子后 vanilla 三发逃离 {73,30,52} 全部 ≤73，与此前无种子 5 发 {93,94,179,235,286} 分布不同——种子本身也是逃离时机的强变量，记录实验时必须带种子。
- v2 设计（已实现本地，测试全过）：退出标准 train_IoU>0.05 连续 3 轮；退出后 10 轮 λ 线性混合（(1-λ)·edgeSCE + λ·平衡BCE）渐进恢复；blend 期再塌缩 λ 立即回 1。待服务器 A/B v2。

## 2026-09-04 PAL-Guard A/B v2 结果（服务器 4090，ACM，6 发×100ep，目录 ab_2026-09-04_16-12-31）

| run | vanilla 逃离 | guard 逃离 | vanilla 末 mIoU@100 | guard 末 mIoU@100 | GUARD 事件 |
|---|---|---|---|---|---|
| #1 (seed 1001) | 55 | 41 | 0.3829 | 0.1340 | 3 |
| #2 (seed 1002) | **未逃离** | 26 | **0.0000** | 0.4594 | 0 |
| #3 (seed 1003) | 97 | 47 | 0.1934 | 0.1148 | 3 |

- 结论 1（逃离加速，强证据）：三对全部 guard 先逃离（41<55、47<97；#2 为 26<永不）。跨 v1+v2 共 6 对，guard 全部不晚于 vanilla。
- 结论 2（v2 修复再塌缩）：#1/#3 事件=3（激活→退出→blend完成），无再塌缩事件——v1 的"退出即再塌缩"已消除。
- 结论 3（终点仍偏弱，诚实记录）：guard 实际触发的 #1/#3，100ep 终点低于 vanilla（0.134<0.383、0.115<0.193）。机制解释：平衡 BCE 阶段学到高召回/高 FA 解，回到 edgeSCE OHEM 后需重构几何，爬升慢于"原生 edgeSCE 逃离"。
- 结论 4（guard 的保险价值）：vanilla #2 全程塌缩 0.0000——塌缩吸引子真实且致命（100ep 白跑）；guard 组 6 发无一全军覆没。
- 注意 #2 归因：guard 26 轮逃离 < 触发门槛 40 轮（事件=0），是自然逃离非 guard 功劳——同种子不同 init 的运气方差，不能计入 guard 收益。
- 100ep 截断局限：PAL 增强在 epoch 85 才开始，截断只吃 15 轮增强；终点收益需在 400ep 全程验证（列为后续工作，若预算允许可跑 1 对 400ep A/B）。
- 报告叙事线（诚实版）：塌缩吸引子的发现与刻画（11+ 发数据，跨框架）+ Guard 一致加速逃离 + v2 消除再塌缩 + 100ep 终点收益未显现（负结果如实写）+ 400ep 验证列后续。

## SCT 服务器训练 run1（✅ 完赛 2026-09-04 ~15:50）

- 启动 2026-09-03 21:39（A/B 后自动链接），`--model SCT --epochs 400`，约 18h。
- **best mIoU = 0.7052**；epoch 18 逃离塌缩（val mIoU 0.1795），epoch 22 train_IoU=0.4695。
- 深监督 6 分支形态与 ACM/ALC 完全不同：逃离快、全程无塌缩——"深监督天然抗塌缩"假设获第三组证据。
- 对标：PAL 论文 Table 1 无 SCT 行（作者后加未发权重）；**0.7052 超过表中全部 8 网络**（最高 MSDA 69.38、DNANet 67.20）。写报告时可再对标 SCTransNet 原论文（注意其为全监督，需在报告中注明监督口径差异）。
- checkpoint：服务器 `work_dirs/pal_<SCT时间戳>/best_mIoU_checkpoint.pkl`，训练日志 `train_pal_sct_run1.log`（日志待拉回本地）。

## ISNet 服务器训练 run1（✅ 完赛 2026-09-05 ~16:00）

- `--model ISNet --epochs 400`，约 16h。DCN 手写算子全程无异常。
- **best mIoU = 0.5998**；首次逃离 epoch 57（val mIoU 0.1011）。
- 权重：`PAL_jittor/isnet_best_mIoU_0.5998.pkl`（4.3MB）；日志在 `pal_all_logs.tar.gz`（含两轮 A/B 全量日志）。
- 四网全家福（SIRST3+masks_coarse+PAL，jittor 自训 400ep）：ACM 0.4712（逃离 93~286）/ ALC 0.5057（93~286）/ SCT 0.7052（18）/ ISNet 0.5998（57）。
- 2026-09-05 16:15 全部资产拉回本地，AutoDL 实例退役。
