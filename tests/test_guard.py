# -*- coding: utf-8 -*-
"""PAL-Guard 抗塌缩机制的本地实测 (tests/test_guard.py)。

四个测试:
  ① test_balanced_bce_numeric   平衡 BCE 数值正确性:
       jittor edge_sce_loss_guard vs numpy 手算 (逐位级) vs torch
       F.binary_cross_entropy_with_logits(pos_weight=p) (子进程, 环境缺失则跳过);
       覆盖自动 pos_weight、显式 pos_weight、cap=1000、全背景 batch。
  ② test_guard_controller_fsm   GuardController v2 状态机:
       39 轮全 0 不触发 / 40 轮触发 / >0.05 连续 3 轮才退出 (0.03 不计连击) /
       blend λ 逐轮数值 (1.0,0.9,...,0.1) / blend 完成回 off / 再触发 /
       blend 期再塌缩 λ 立即回 1 / blend_epochs=0 硬切 / enabled=False 恒关。
  ③ test_guard_off_equivalence  --guard 关闭时与现版本一致:
       (a) 静态: guard_lambda=None 分支字面上就是 edgeSCE_loss 调用,
           --guard 默认关, v2 参数默认 0.05/3/10;
       (b) 进程内: 同权重快照 + 冻结权重 (no-op optimizer), 旧 train_one_epoch
           与 train_one_epoch_guard(guard_lambda=None) 的 loss 序列逐位一致;
       (b2) blend 数值: λ=0.3 时 loss == 0.7·edgeSCE + 0.3·平衡BCE;
       (c) CLI: 同种子两跑 (CPU, 小数据), loss 序列在栈自身噪声底内一致
           (已实测 epoch1 rel≈2e-5, epoch2 rel≈3e-4~7.5e-4, 随 epoch 累积
           发散——来源是 jittor 惰性执行 init 抽签顺序 + 多线程归约, 与
           Guard 无关; 逐位级等价由 ③b 进程内对拍保证), 且日志无 [GUARD]。
  ④ test_smoke_both_modes       --epochs 3 小数据冒烟, guard on/off 各一发,
       无 NaN, 正常跑完; guard-on 仅验证代码路径 (3<40 不会触发, 触发逻辑
       已由 ② 覆盖)。

运行 (Git Bash, 工作目录 PAL_jittor 的上一级):
    D:/Anaconda/envs/jittor/python.exe PAL_jittor/tests/test_guard.py            # 全部
    D:/Anaconda/envs/jittor/python.exe PAL_jittor/tests/test_guard.py 1 2        # 指定编号
torch 对拍子进程解释器可用环境变量 PAL_TORCH_PY 覆盖
(默认 D:/Anaconda/envs/pal_torch/python.exe; 不存在则跳过该子项)。
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAL_JT = os.path.dirname(HERE)                  # PAL_jittor/
sys.path.insert(0, PAL_JT)
# 进程内测试一律 CPU, 避免 GPU 原子归约噪声 (并加速小算例)
os.environ['PAL_JT_FORCE_CPU'] = '1'

JT_PY = sys.executable
TORCH_PY = os.environ.get('PAL_TORCH_PY', 'D:/Anaconda/envs/pal_torch/python.exe')

PASS = []


def _report(name, ok, detail=''):
    PASS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  {detail}' if detail else ''))


# ---------------------------------------------------------------- ① 数值对拍
def test_balanced_bce_numeric():
    import jittor as jt
    from loss.edge_sce import edge_sce_loss_guard

    rng = np.random.RandomState(0)
    logits = (rng.randn(2, 1, 8, 8) * 2.0).astype('float32')
    target = (rng.rand(2, 1, 8, 8) > 0.9).astype('float32')

    def np_bce_pw(z, t, pw):
        bce = np.maximum(z, 0) - z * t + np.log(1 + np.exp(-np.abs(z)))
        w = 1.0 + (pw - 1.0) * t
        return float((bce * w).mean())

    n_pos = float((target > 0.5).sum())
    n_neg = float(target.size - n_pos)
    pw = min(n_neg / max(n_pos, 1.0), 1000.0)

    l_auto = float(edge_sce_loss_guard(jt.array(logits), jt.array(target)).item())
    l_expl = float(edge_sce_loss_guard(jt.array(logits), jt.array(target),
                                       pos_weight=pw).item())
    l_np = np_bce_pw(logits, target, pw)
    print(f'  自动 pos_weight: n_pos={n_pos:.0f} n_neg={n_neg:.0f} pw={pw:.2f}')
    print(f'  jt_auto={l_auto:.10f}  jt_expl={l_expl:.10f}  np={l_np:.10f}')
    assert abs(l_auto - l_np) < 1e-6, f'自动 pos_weight 与 numpy 手算不符: {l_auto} vs {l_np}'
    assert abs(l_expl - l_np) < 1e-6, f'显式 pos_weight 与 numpy 手算不符'

    # cap=1000: 1 正像素 + 10099 负像素 -> 未截断 pw=10099 > 1000
    z2 = rng.randn(1, 1, 101, 100).astype('float32')
    t2 = np.zeros_like(z2); t2[0, 0, 0, 0] = 1.0
    l_cap = float(edge_sce_loss_guard(jt.array(z2), jt.array(t2)).item())
    l_cap_np = np_bce_pw(z2, t2, 1000.0)
    print(f'  cap=1000: jt={l_cap:.10f}  np(pw=1000)={l_cap_np:.10f}')
    assert abs(l_cap - l_cap_np) < 5e-6, 'cap=1000 未生效'  # fp32 在 ~10k 像素求和的精度底

    # 全背景 batch: w 恒 1, 退化为纯 BCE mean, 仍给全图梯度
    l_bg = float(edge_sce_loss_guard(jt.array(logits),
                                     jt.array(np.zeros_like(target))).item())
    bce_bg = np.maximum(logits, 0) + np.log(1 + np.exp(-np.abs(logits)))
    l_bg_np = float(bce_bg.mean())
    print(f'  全背景: jt={l_bg:.10f}  np={l_bg_np:.10f}')
    assert abs(l_bg - l_bg_np) < 1e-6, '全背景 batch 应退化为纯 BCE mean'

    # torch 交叉对拍 (环境缺失则跳过)
    if os.path.exists(TORCH_PY):
        code = (
            'import numpy as np, torch, torch.nn.functional as F;'
            'rng=np.random.RandomState(0);'
            'z=(rng.randn(2,1,8,8)*2.0).astype("float32");'
            't=(rng.rand(2,1,8,8)>0.9).astype("float32");'
            f'pw={pw!r};'
            'l=F.binary_cross_entropy_with_logits(torch.from_numpy(z),'
            'torch.from_numpy(t),pos_weight=torch.tensor(pw));'
            'print(f"{l.item():.10f}")')
        out = subprocess.run([TORCH_PY, '-c', code], capture_output=True,
                             text=True, timeout=120, encoding='utf-8',
                             errors='replace')
        l_torch = float(out.stdout.strip())
        print(f'  torch={l_torch:.10f}  |jt-torch|={abs(l_auto - l_torch):.3e}')
        assert abs(l_auto - l_torch) < 1e-5, '与 torch BCEWithLogits(pos_weight) 不符'
    else:
        print(f'  [skip] torch 解释器不存在: {TORCH_PY}')

    _report('① 平衡 BCE 数值对拍', True,
            f'jt_auto={l_auto:.8f}, np={l_np:.8f}, cap/全背景均过')


# ---------------------------------------------------------------- ② 状态机
def test_guard_controller_fsm():
    """v2 状态机: off -> active -> blend -> off (可多次循环)。

    覆盖: 触发不变(40轮) / 新标准 >0.05 连续3轮才退出 / 单轮回落不退出 /
    0.02~0.05 之间的值不计入连击(v1 标准会误退) / blend λ 逐轮数值 /
    blend 完成回 off / 再触发 / blend 期再塌缩 λ 回 1 / disabled 恒关 /
    blend_epochs=0 硬切 / 窗口边界。
    """
    from train_pal_jt import GuardController
    APX = 1e-12

    logs = []
    g = GuardController(enabled=True, log_fn=logs.append)

    # 1) epoch 1..39 全 0: 不触发 (窗口未满或 e1<40), λ 恒 None
    for e in range(39):
        assert g.update(e, 0.0) is None, f'e1={e + 1} 不应激活'
    # 2) epoch 40: 窗口 [36..40] 全 <0.005 且 e1>=40 -> 激活, 下轮 λ=1
    assert g.update(39, 0.0) == 1.0, 'e1=40 应激活'
    assert g.activations == [40] and g.state == 'active'
    # 3) 新退出标准: >0.05 连续 3 轮才退出; 单轮回落清零; 0.03 不计入连击
    assert g.update(40, 0.06) == 1.0      # streak=1
    assert g.update(41, 0.03) == 1.0      # 0.03<0.05 -> 清零 (v1 的 0.02 标准会误计)
    assert g.update(42, 0.06) == 1.0      # streak=1
    assert g.update(43, 0.055) == 1.0     # streak=2 (2 轮不够)
    lam = g.update(44, 0.07)              # streak=3 -> 退出, 进入 blend 第 1 轮
    assert abs(lam - 1.0) < APX and g.state == 'blend', '退出后 blend 第 1 轮 λ=1'
    assert g.exits == [45]
    # 4) blend λ 逐轮: 第 2..10 轮 = 0.9..0.1, 第 11 次更新完成 -> None
    expect = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    got = [1.0]
    for k, exp in enumerate(expect):
        lam = g.update(45 + k, 0.5)
        got.append(lam)
        assert abs(lam - exp) < APX, f'blend 第 {k + 2} 轮: {lam} != {exp}'
    assert g.update(54, 0.5) is None and g.state == 'off', 'blend 10 轮后应完成'
    assert g.blend_done == [55]
    print(f'  blend λ 调度实测: {[round(x, 2) for x in got]} -> None')
    # 5) off 期再塌缩 5 轮 -> 再触发 (多次循环)
    for e in range(55, 59):
        assert g.update(e, 0.0) is None, f'e1={e + 1} 窗口未全塌缩'
    assert g.update(59, 0.0) == 1.0, 'e1=60 应再次激活'
    assert g.activations == [40, 60]
    # 6) 再次退出后, blend 期再塌缩: λ 立即回 1
    for e in (60, 61, 62):
        assert g.update(e, 0.06) == 1.0
    assert g.state == 'blend' and g.exits == [45, 63]
    lams = [g.update(63 + k, 0.0) for k in range(4)]   # blend 第 2..5 轮
    for got_l, exp in zip(lams, [0.9, 0.8, 0.7, 0.6]):
        assert abs(got_l - exp) < APX, f'blend 中 {got_l} != {exp}'
    lam = g.update(67, 0.0)              # 第 5 个 0 -> 窗口全塌 -> 回退
    assert lam == 1.0 and g.state == 'active', 'blend 期再塌缩 λ 应立即回 1'
    assert g.recollapses == [68]
    # 7) disabled 恒关
    g2 = GuardController(enabled=False)
    for e in range(60):
        assert g2.update(e, 0.0) is None
    # 8) blend_epochs=0: 退出即硬切关闭
    g3 = GuardController(enabled=True, blend_epochs=0, log_fn=logs.append)
    for e in range(39):
        assert g3.update(e, 0.0) is None
    assert g3.update(39, 0.0) == 1.0
    assert g3.update(40, 0.06) == 1.0 and g3.update(41, 0.06) == 1.0
    assert g3.update(42, 0.06) is None and g3.state == 'off', 'blend=0 应直接关闭'
    assert g3.exits == [43]
    # 9) 窗口边界: 有任何一轮 >=0.005 不触发
    g4 = GuardController(enabled=True, log_fn=lambda m: None)
    for e in range(39, 45):
        assert g4.update(e, 0.006 if e == 43 else 0.0) is None

    n_act = sum('激活保护' in m for m in logs)
    n_blend = sum('进入 blend 期' in m for m in logs)
    n_done = sum('轮完成' in m for m in logs)
    n_recol = sum('再塌缩' in m for m in logs)
    n_hard = sum('关闭保护 (无 blend)' in m for m in logs)
    print(f'  日志: 激活×{n_act} 退出进blend×{n_blend} blend完成×{n_done} '
          f'再塌缩回退×{n_recol} 硬切关闭×{n_hard}')
    for m in logs:
        print(' ', m)
    assert (n_act, n_blend, n_done, n_recol, n_hard) == (3, 2, 1, 1, 1)
    _report('② GuardController v2 状态机', True,
            '触发@40 / >0.05×3退出@45 / blend λ:1.0→0.1→完成 / 再触发@60 / '
            'blend再塌缩@68 λ回1 / blend=0硬切 / disabled恒关')


# ---------------------------------------------------------------- ③ guard-off 一致性
def test_guard_off_equivalence():
    # (a) 静态检查: 非 guard 分支字面上就是 edgeSCE_loss; --guard 默认关;
    # v2 新增参数默认值符合规格
    src = open(os.path.join(PAL_JT, 'train_pal_jt.py'), encoding='utf-8').read()
    assert 'guard_lambda is None' in src and \
           'loss = edgeSCE_loss(pred, targets, edge_t)' in src, \
        'guard-off 分支必须字面上调用 edgeSCE_loss'
    assert "ap.add_argument('--guard', action='store_true'" in src, \
        '--guard 必须是 store_true (默认关)'
    for frag in ["--guard_exit_iou', type=float, default=0.05",
                 "--guard_exit_patience', type=int, default=3",
                 "--guard_blend_epochs', type=int, default=10"]:
        assert frag in src, f'v2 参数默认值不符规格: {frag}'
    print('  (a) 静态: guard-off 分支 == edgeSCE_loss 字面调用, --guard 默认关, '
          'v2 参数默认 0.05/3/10 ✓')

    # (b) 进程内: 冻结权重下旧函数 vs 新函数 (guard off) loss 逐位一致。
    # 注意: jittor 惰性执行使 jt.seed 复种无法复现"构造期" init 抽签顺序
    # (已实测: 同种子两次 ACM_No_Sigmoid() 有 62/230 参数不同),
    # 因此用 state_dict 快照对齐权重; 前向 dropout 抽签用 jt.seed 复种对齐
    # (纯 Dropout 算子复种可复现, 已实测)。
    import jittor as jt
    from model.acm import ACM_No_Sigmoid
    from train_acm_jt import train_one_epoch
    from train_pal_jt import train_one_epoch_guard

    class _NoOpt:  # 冻结权重: 隔离前向+loss 路径, 排除反向归约噪声
        def step(self, loss):
            pass

    class _Log:
        def __init__(self):
            self.msgs = []

        def log(self, m):
            self.msgs.append(m)

    rng = np.random.RandomState(7)
    batches = [(rng.randn(2, 3, 256, 256).astype('float32'),
                (rng.rand(2, 256, 256) > 0.5).astype('float32'),
                (rng.randint(0, 2, (2, 1, 256, 256)) * 255).astype('float32'))
               for _ in range(2)]

    m1 = ACM_No_Sigmoid()
    snap = {k: v.numpy().copy() for k, v in m1.state_dict().items()}
    jt.seed(123)
    lg1 = _Log()
    loss_old = train_one_epoch(batches, m1, _NoOpt(), 0, lg1)

    m2 = ACM_No_Sigmoid()
    m2.load_state_dict(snap)
    jt.seed(123)
    lg2 = _Log()
    loss_new, tIoU, tnIoU = train_one_epoch_guard(batches, m2, _NoOpt(), 0, lg2,
                                                  guard_lambda=None)
    print(f'  (b) 进程内冻结权重: old={loss_old!r}  new(guard off)={loss_new!r}')
    assert loss_old == loss_new, \
        f'guard-off loss 序列应与旧函数逐位一致: {loss_old!r} vs {loss_new!r}'
    # 日志格式与旧版逐字节一致 (除耗时字段)
    strip = lambda s: re.sub(r'耗时 .*s$', '', s)
    assert strip(lg1.msgs[0]) == strip(lg2.msgs[0]), 'guard-off 日志格式应与旧版一致'
    print(f'  (b) 日志格式一致 ✓  train_IoU={tIoU:.4f}')

    # (b2) blend 数值: 同权重同批数据, λ=0.3 时
    # loss == 0.7·edgeSCE + 0.3·平衡BCE (前向 dropout 抽签靠 jt.seed 复种对齐,
    # BN train 模式用批统计, 三次调用前向输出一致)
    jt.seed(9)
    lg_off = _Log()
    loss_off, _, _ = train_one_epoch_guard(batches, m2, _NoOpt(), 0, lg_off,
                                           guard_lambda=None)
    jt.seed(9)
    loss_bce, _, _ = train_one_epoch_guard(batches, m2, _NoOpt(), 0, _Log(),
                                           guard_lambda=1.0)
    jt.seed(9)
    lg_mix = _Log()
    loss_mix, _, _ = train_one_epoch_guard(batches, m2, _NoOpt(), 0, lg_mix,
                                           guard_lambda=0.3)
    expect = 0.7 * loss_off + 0.3 * loss_bce
    rel = abs(loss_mix - expect) / max(abs(expect), 1e-12)
    print(f'  (b2) blend λ=0.3: mix={loss_mix:.8f}  '
          f'0.7·SCE+0.3·BCE={expect:.8f}  rel={rel:.2e}')
    assert rel < 1e-5, 'blend 损失应等于两分量线性组合'
    assert '[GUARD λ=0.30]' in lg_mix.msgs[0], 'blend 期日志应带 λ 标记'
    assert '[GUARD' not in lg_off.msgs[0], 'guard-off 日志不应带 λ 标记'

    _report('③ guard-off 与现版本一致 (静态+进程内) + blend 数值', True,
            f'off 逐位相同={loss_old!r}; blend rel={rel:.1e}')


# ------------------------------------------------------------- ③c CLI 两跑
def test_guard_off_cli_determinism():
    """同种子两跑 CLI (CPU 小数据): loss 序列在栈噪声底内一致 + 无 [GUARD]。"""
    with tempfile.TemporaryDirectory(prefix='guard_cli_') as td:
        ws = os.path.join(td, 'ws')
        seqs = []
        for tag in ('a', 'b'):
            env = dict(os.environ, PAL_JT_FORCE_CPU='1')
            cmd = [JT_PY, os.path.join(PAL_JT, 'train_pal_jt.py'),
                   '--model', 'ACM', '--epochs', '2', '--seed', '123',
                   '--limit_init', '32', '--limit_train', '32', '--limit_val', '2',
                   '--pal_workspace', ws,
                   '--save_dir', os.path.join(td, f'sd_{tag}')]
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=280, env=env, cwd=PAL_JT,
                                 encoding='utf-8', errors='replace')
            assert out.returncode == 0, f'CLI 跑失败:\n{out.stderr[-2000:]}'
            assert '[GUARD]' not in out.stdout, 'guard 默认关, 日志不应出现 [GUARD]'
            seq = re.findall(r'\[train\] epoch \d+: loss_mean=([\d.]+)', out.stdout)
            assert len(seq) == 2, f'应有两个 epoch 的 loss, 实际: {seq}'
            seqs.append([float(x) for x in seq])
        print(f'  (c) run A: {seqs[0]}')
        print(f'  (c) run B: {seqs[1]}')
        for i, (x, y) in enumerate(zip(*seqs)):
            rel = abs(x - y) / max(abs(x), 1e-12)
            print(f'      epoch {i + 1}: |Δ|rel = {rel:.3e}')
            # 栈自身跨进程噪声随 epoch 累积 (实测 epoch2 3e-4 ~ 7.5e-4), 阈值取 5e-3
            assert rel < 5e-3, f'超出栈自身噪声底: {rel:.3e}'
        _report('③c CLI 同种子两跑 (噪声底内一致, 无 [GUARD])', True,
                f'max rel diff={max(abs(x - y) / abs(x) for x, y in zip(*seqs)):.2e}')


# ---------------------------------------------------------------- ④ 双模式冒烟
def test_smoke_both_modes():
    """--epochs 3 小数据冒烟: guard off / on 各一发 (GPU), 无 NaN, 跑完。"""
    with tempfile.TemporaryDirectory(prefix='guard_smoke_') as td:
        ws = os.path.join(td, 'ws')
        for tag, extra in [('off', []), ('on', ['--guard'])]:
            env = dict(os.environ)
            env.pop('PAL_JT_FORCE_CPU', None)   # 冒烟走 GPU (快)
            cmd = [JT_PY, os.path.join(PAL_JT, 'train_pal_jt.py'),
                   '--model', 'ACM', '--epochs', '3', '--seed', '7',
                   '--limit_init', '64', '--limit_train', '64', '--limit_val', '4',
                   '--pal_workspace', ws,
                   '--save_dir', os.path.join(td, f'sd_{tag}')] + extra
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=280, env=env, cwd=PAL_JT,
                                 encoding='utf-8', errors='replace')
            assert out.returncode == 0, f'guard {tag} 跑失败:\n{out.stderr[-2000:]}'
            assert 'PAL_TRAIN_DONE' in out.stdout, f'guard {tag} 未正常结束'
            losses = [float(x) for x in re.findall(
                r'\[train\] epoch \d+: loss_mean=([\d.eE+-]+)', out.stdout)]
            assert len(losses) == 3, f'guard {tag} 应有 3 个 epoch: {losses}'
            assert all(np.isfinite(losses)), f'guard {tag} 出现 NaN/Inf: {losses}'
            if tag == 'on':
                assert 'PAL-Guard 已开启' in out.stdout
                # 3 个 epoch 未到 min_epoch=40, 不应触发 (触发逻辑由 ② 覆盖)
                assert '激活保护' not in out.stdout
            else:
                assert '[GUARD]' not in out.stdout
            print(f'  guard {tag}: losses={losses} ✓')
        _report('④ 双模式 3-epoch 冒烟 (无 NaN)', True,
                f'off={losses} on 同形态')


ALL = [test_balanced_bce_numeric, test_guard_controller_fsm,
       test_guard_off_equivalence, test_guard_off_cli_determinism,
       test_smoke_both_modes]

if __name__ == '__main__':
    order = {'1': 0, '2': 1, '3': 2, '3c': 3, '4': 4}
    picks = {order[x] for x in sys.argv[1:] if x in order} if len(sys.argv) > 1 else None
    for i, fn in enumerate(ALL):
        if picks is not None and i not in picks:
            continue
        print(f'=== {fn.__name__} ===')
        fn()
    print()
    bad = [n for n, ok in PASS if not ok]
    print(f'合计 {len(PASS)} 项, 失败 {len(bad)} 项' + (f': {bad}' if bad else ''))
    sys.exit(1 if bad else 0)
