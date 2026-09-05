# W5 汇总报告：ISNet PyTorch → Jittor 迁移与 PAL 集成

日期：2026-09-04 ｜ 环境：`D:\Anaconda\envs\jittor\python.exe`（jittor 1.3.8.5，CUDA 可用）｜ 参考实现：`PAL/model/ISNet/`（TTOA.py 等）+ `PAL/train_model.py` ISNet 分支 ｜ 集成模板：此前完成的 SCT 迁移

**结论：W5 全部完成。** ISNet 已迁移为纯 Jittor 实现（含手写 DCN），数值上与 PyTorch 参考实现对齐到 fp32 决策级零分歧 / fp64 结构级 1e-14；已作为第四模型接入 `train_pal_jt.py`（`--model ISNet`），冒烟训练与既有回归全部通过。

---

## 1. 交付物清单

| 文件 | 说明 |
|---|---|
| `PAL_jittor/model/isnet.py` | Jittor 版 ISNet_No_Sigmoid 完整实现（约 23 KB）：手写 `deform_conv2d` + `DCN`、`GatedSpatialConv2d`、`TTOA`、vendor `BasicBlock`/`BasicBlock1`、`GetGradientNoPadding`、`TFD`、`_FCNHead`、死模块（SA/SA_att/cw/head2/conv2_1/conv16/res1-3/d1-3/gate1-3） |
| `PAL_jittor/convert_isnet_weights.py` | torch state_dict → jittor 权重转换器（按任务书要求放仓库根，与 ACM/SCT 放 model/ 的惯例不同）。独立运行输出 `CONVERT_DONE`，产物 `work_dirs/isnet_torch_init_jt.pkl` |
| `PAL_jittor/tests/test_isnet.py` | 验收测试 [1] 形状契约 / [2a] fp32 对齐 / [2b] fp64 结构等价 / [3] train 模式反向，全部 PASS（EXIT=0） |
| `PAL_jittor/tools/export_isnet_torch_refs.py` | torch 参考导出器（seed 42 建模型，seed 2024 非平凡化 DCN offset，导出 state_dict + 参考输出 npz） |
| `PAL_jittor/tools/probe_dcn_torch.py` / `probe_dcn_jittor.py` | DCN 双盲探针（前向 + 5 项梯度对拍），数据 `tests/data/probe_dcn.npz` |
| `PAL_jittor/train_pal_jt.py` | 集成改动（见 §6） |

模型规模：**ISNet 参数量 1.096 M**（冒烟日志实测），远小于 SCT/ACM，默认 batch 16 无显存压力。

## 2. DCN（可变形卷积）：实际配置与手写实现要点

### 2.1 PAL 源码中的真实配置（TTOA.py L31-32）

TTOA 模块共 4 个 DCN，**全部为非标准几何**：

- kernel_size = (1,3) 与 (3,1) 两种（十字形），stride=1，padding=(0,1)/(1,0)，dilation=1，groups=1
- **带 mask**（sigmoid 门控）、**带 bias**
- `conv_offset_mask` 源码**零初始化**——这是结构性设定：offset 恒 0 时 DCN 退化为普通卷积，PAL 从头训练靠它起步，Jittor 版必须复刻（已 zero 初始化）

### 2.2 offset 通道布局

经 numpy 双假设对拍确认为**交错式**：第 t 个 tap 的 (y, x) 偏移存于通道 (2t, 2t+1)（与 torchvision 一致）。

### 2.3 手写 `deform_conv2d` 实现链（model/isnet.py L39-93）

纯 jittor 算子组合，无自定义 CUDA kernel：

1. base grid + offset → 采样坐标；`floor` 取四角
2. 双线性加权，**角点出界贡献置 0、不做权重归一化**（与 torchvision 语义逐位一致）
3. gather 用 `x.reindex([b,c,y,x])`（索引 clamp + 有效性掩码）
4. 权重累加用**广播乘 + `sum(dims=(2,3))`**——刻意不用 matmul：`cublasGemmEx` 不支持 fp64，CUDA fp64 会崩 `CUBLAS_STATUS_NOT_SUPPORTED`

### 2.4 DCN 探针验证（与 torchvision 对拍）

固定输入 `tests/data/probe_dcn.npz`，双盲对比前向 + 5 项梯度（x / offset / mask / weight / bias）：

- fp64 下全部 max abs diff ≤ **8.9e-15**（前向与 5 项梯度全过）

### 2.5 导出时的关键处理：DCN offset 非平凡化

源码 `conv_offset_mask` 零初始化 → 随机初始化模型的 offset 恒为 0，**测不到双线性采样路径**。`tools/export_isnet_torch_refs.py` 先用 seed 42 构造模型，再用 seed 2024 将 4 个 DCN 的 `conv_offset_mask` 权重改为 N(0, 0.5)，然后才导出 state_dict 与参考输出——保证对齐测试真正覆盖变形采样分支。

## 3. GatedSpatialConv2d 处理方案

原版继承 `_ConvNd`（jittor 无此类），改为显式参数持有：

- `self.weight`：`xavier_gauss` 初始化，形状与原 conv 一致
- `bias=None`：该层全部使用点均为 `bias=False`（逐一核对源码确认）
- `_gate_conv` 保持 `nn.Sequential` 原结构，**键名与 torch 完全一致**（`_gate_conv.0/3/4...`），权重映射零改写
- `mynn.Norm2d == BatchNorm2d`（PAL config.py L70），直接用 `nn.BatchNorm2d`

## 4. 权重映射摘要

| 项 | 数值 |
|---|---|
| torch state_dict 键数 | **551** |
| jittor 载入键数 | **473** |
| 差值 | 78 = BN 的 `num_batches_tracked` 数（jittor 无此缓冲，转换时按后缀 `.num_batches_tracked` 统一跳过） |

- DCN offset 层 / GatedSpatialConv / `register_buffer`（`grad_extractor.weight_v/weight_h`，用同名普通 `jt.array` 属性承载）/ 死模块，全部键名 **1:1** 映射，无改写、无遗漏
- 逐层映射表可用转换器 verbose 模式打印
- 死模块（SA/SA_att/cw/head2/conv2_1/conv16/res1-3/d1-3/gate1-3）保留的理由：state_dict 全覆盖需要——缺任一键转换器即报缺失

## 5. 数值对齐与验收测试（`tests/test_isnet.py`，全部 PASS，EXIT=0）

本次复跑实测（CPU 对齐模式，`jt.flags.use_cuda=0`，规避 CUDA TF32 污染 fp32 对拍）：

**[1] 形状/契约**：`(2,3,256,256) → (out, edge_out)` 均为 `(2,1,256,256)`；edge_out 值域实测 [0.415, 0.505] ⊂ [0,1]（内部已 sigmoid）✓

**[2a] fp32 对齐**（torch CPU 参考，eval 模式，同源权重）：

| 指标 | out 分支 | edge_out 分支 | 要求 |
|---|---|---|---|
| logit/prob max abs diff | 1.788e-07 | 5.364e-07 | sigmoid 决策级 < 1e-4 |
| sigmoid diff | 8.941e-08 | — | < 1e-4 |
| 二值化不一致像素 | **0** | **0** | 必须为 0 |

**[2b] fp64 结构等价**（真 fp64，阈值 1e-6，输入 1x3x64x64）：out max abs diff = **4.807e-14**，edge_out = **5.851e-14** ✓

**[3] train 模式前向+反向**（1x3x64x64，`loss = out.mean() + edge.mean()`）：loss = 0.0638，全局梯度范数 = 5.45，有限非零 ✓（该项用全新随机初始化模型，loss/梯度范数随初始化有 ±0.01 量级波动，判据为"有限非零"；反向覆盖 DCN reindex gather 路径）

> fp64 判定改用 64×64 小输入（`tests/data/isnet_ref_small_fp64.npz`）的原因：jittor CPU fp64 全尺寸（256）前向超 300 s 不可用；256 全尺寸精度由 [2a] fp32 对齐覆盖，结构等价性由 fp64 小输入承担。

## 6. `train_pal_jt.py` 集成改动

**ISNet 输出契约**（据 train_model.py L285-292/164/192）：返回 `(out, edge_out)`；`out` = 无 sigmoid logits，`edge_out` 内部已 sigmoid；**训练 loss 只算 out 分支**（edge 分支 loss 原版被注释）；指标/val/PAL 推理均取 **[0]**（不同于 SCT 的 [-1]）。

| 行号 | 改动 |
|---|---|
| L2-15 | docstring 更新：四模型说明 + ISNet 分支语义 |
| L51 | `from model.isnet import ISNet_No_Sigmoid` |
| L57-72 | MODELS 注册表改三元组 `(类, kwargs, (out_index, loss_mode))`：ISNet = `(0, 'first_only')`，SCT = `(-1, 'mean_all')`，ACM/ALC = None |
| L87 / L111 / L125 | `pal_infer_one` / `pal_label_self_update` / `pal_infer_no_choose` 增加 `out_index` 透传（L104 取分支） |
| L287 / L294-315 | `train_one_epoch_guard` 增加 `loss_mode` 参数：`'first_only'` 时只对 `pred[0]` 算 loss；Guard v2 的 λ 照常经 `_guard_mix_loss` 透传 |
| L348 / L364 | `val_one_epoch_ds` 增加 `out_index`，`pred = pred[out_index]` |
| L475 / L477 | 建模型 + 解包 `MODELS[args.model][2] or (-1, 'mean_all')` |
| L532 / L535 / L553 / L563 / L569 | 五个调用点透传 `out_index` / `loss_mode` |

**未动范围**：Guard v2（λ blend 状态机）零改动；SCT/ACM/ALC 语义零改动（注册表默认值保持原行为）。

## 7. 冒烟训练（CUDA）

命令：`--model ISNet --epochs 2 --limit_init 64 --limit_train 64 --limit_val 16`（默认 batch 16）

| 项 | 结果 |
|---|---|
| 退出码 | EXIT=0，日志尾部 `PAL_TRAIN_DONE` |
| NaN 检查 | grep `nan/NaN` 计数 **0** |
| train loss | epoch1 **1.8434** → epoch2 **1.6533**（正常下降） |
| train_IoU / val mIoU | 0.0000（点监督 + 2 epoch 起步的正常形态，与 SCT 冒烟同型） |
| 参数量 | 1.096 M |
| 检查点 | `work_dirs/isnet_smoke/`（best + last 均在） |
| 耗时 | epoch1 226 s（含 mean/std 计算与编译），epoch2 46 s |

## 8. 回归

| 测试 | 结果 |
|---|---|
| `tests/test_guard.py 4`（Guard v2 双模式 3-epoch 冒烟） | **PASS**，失败 0 项；guard off=[3.245, 3.040, 2.943] / on 同形态，无 NaN |
| SCT 路径 1-epoch 回归（`work_dirs/sct_smoke_regress`） | EXIT=0，`PAL_TRAIN_DONE`（val PD=0.0909，与集成前同形态） |

## 9. 本次迁移新踩的坑（前人清单之外）

1. **`jt.array(np.float64)` 默认降级 float32**——fp64 测试必须显式 `dtype='float64'`，否则"fp64 对拍"实际在 fp32 下进行，判据失效。
2. **C=1 bias 的 train 模式融合编译错**：`_gate_conv[3].bias`（输出通道 1）在 train 模式触发 `op0_outputstrideN` 融合编译错误 → 已对相关算子 `stop_fuse`（dsn1/2/3.bias、head.block[4].bias 及进 interpolate 的张量同样处理）。
3. **jittor CPU fp64 全尺寸不可用**（256×256 前向 > 300 s）→ fp64 判定改 64×64 小输入，256 全尺寸由 fp32 对齐覆盖。
4. **逐参数取梯度会重复执行反向子图**——应把所有 grads reshape 拼接成单向量后一次 `.sync()`/取 numpy，否则耗时随参数量线性放大。

## 10. 残留说明

- 冒烟产物保留未清理：`pal_workspaces/isnet_smoke`、`work_dirs/isnet_smoke`、`work_dirs/sct_smoke_regress`（体积小，留作复查凭据）；临时日志已删。
- `work_dirs/isnet_torch_init_jt.pkl` 为对齐测试用的随机初始化转换权重，非训练产物。
