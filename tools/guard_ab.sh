#!/usr/bin/env bash
# guard_ab.sh — PAL-Guard 服务器一键 A/B 实验
#
# 顺序跑 6 发从头训练 (ACM, 各 100 epoch, 默认 pal_total_epochs=400):
#   vanilla x3:  train_pal_jt.py --model ACM --epochs 100
#   guard   x3:  同上 + --guard
# 日志落在 $AB_DIR/ab_{vanilla,guard}_{1,2,3}.log (TeeLogger 另存
# $AB_DIR/{vanilla,guard}_$i/pal_train.log), checkpoint 在各自 save_dir。
# 结尾自动汇总: 每发首次 val mIoU>0.01 的 epoch (逃离塌缩点) + 末轮 mIoU。
#
# 用法 (nohup 友好, 无交互):
#   PAL_JT_PY=/path/to/python nohup bash tools/guard_ab.sh > ab_driver.log 2>&1 &
#   (PAL_JT_PY 缺省 python3; 本脚本 cd 到 PAL_jittor/ 后一切用相对/绝对路径)
# 测试钩子 (正式实验不要用): AB_EPOCHS / AB_NRUNS 可覆盖 epoch 数与每组发数,
# AB_EXTRA_ARGS 追加训练参数 (如限流/种子):
#   AB_EPOCHS=1 AB_NRUNS=1 AB_EXTRA_ARGS="--limit_init 64 --limit_train 64 --limit_val 4" \
#       bash tools/guard_ab.sh   # 全链路冒烟
#
# 注意: 100 epoch + pal_total=400 -> 首轮 PAL 增强在 epoch 索引 85;
# 实验测的就是"从头逃离全背景塌缩", 与设计意图一致。

set -u

PAL_JT_PY=${PAL_JT_PY:-python3}
AB_EPOCHS=${AB_EPOCHS:-100}
AB_NRUNS=${AB_NRUNS:-3}
AB_EXTRA_ARGS=${AB_EXTRA_ARGS:-}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."          # -> PAL_jittor/
ROOT="$(pwd)"

AB_DIR="$ROOT/work_dirs/ab_$(date +%F_%H-%M-%S)"
mkdir -p "$AB_DIR"
echo "[guard_ab] 实验目录: $AB_DIR"
echo "[guard_ab] 解释器: $PAL_JT_PY ($("$PAL_JT_PY" --version 2>&1))"
echo "[guard_ab] 开始时间: $(date '+%F %T')"

run_one() {  # $1=tag(vanilla|guard) $2=run_idx $3=extra_args
    local tag=$1 i=$2 extra=${3:-}
    local save_dir="$AB_DIR/${tag}_${i}"
    local log="$AB_DIR/ab_${tag}_${i}.log"
    echo "[guard_ab] === ${tag} #${i} 开始 $(date '+%F %T') -> $log"
    "$PAL_JT_PY" train_pal_jt.py --model ACM --epochs "$AB_EPOCHS" \
        --save_dir "$save_dir" $AB_EXTRA_ARGS $extra > "$log" 2>&1
    local rc=$?
    echo "[guard_ab] === ${tag} #${i} 结束 $(date '+%F %T') rc=${rc}"
    return $rc
}

FAIL=0
# 同序号 run 共享种子 (1000+i): vanilla/guard 成对比较, 唯一差异是 --guard
for i in $(seq 1 "$AB_NRUNS"); do
    run_one vanilla "$i" "--seed $((1000 + i))" || FAIL=$((FAIL + 1))
    sleep 5
done
for i in $(seq 1 "$AB_NRUNS"); do
    run_one guard "$i" "--guard --seed $((1000 + i))" || FAIL=$((FAIL + 1))
    sleep 5
done

# ----------------------------- 汇总 -----------------------------
# 逃离 epoch: 日志中首个 val mIoU>0.01 的 [val] 行的 epoch 号; 未逃离输出 "-"
first_escape() {  # $1=log
    awk '/\[val\]/ {
            e = $3; sub(":", "", e)
            m = $4; sub("mIoU=", "", m)
            if (m + 0 > 0.01) { print e; found = 1; exit }
         }
         END { if (!found) print "-" }' "$1"
}
last_miou() {     # $1=log -> 最后一个 [val] 行的 mIoU
    awk '/\[val\]/ { m = $4; sub("mIoU=", "", m); last = m }
         END { if (last == "") print "-"; else print last }' "$1"
}
guard_events() {  # $1=log -> [GUARD] 激活/关闭行数
    grep -c '\[GUARD\]' "$1" 2>/dev/null || true
}

SUM="$AB_DIR/ab_summary.txt"
{
    echo "================ PAL-Guard A/B 汇总 ================"
    echo "实验目录: $AB_DIR    汇总时间: $(date '+%F %T')"
    echo "逃离 epoch = 首个 val mIoU>0.01 的 epoch ('-' 表示 100 轮内未逃离)"
    echo
    printf '%-6s | %-10s %-10s | %-10s %-10s | %-8s\n' \
        "run" "vanilla逃离" "guard逃离" "vanilla末mIoU" "guard末mIoU" "GUARD事件"
    printf -- '%s\n' "----------------------------------------------------------------------"
    for i in $(seq 1 "$AB_NRUNS"); do
        vlog="$AB_DIR/ab_vanilla_${i}.log"
        glog="$AB_DIR/ab_guard_${i}.log"
        printf '%-6s | %-10s %-10s | %-10s %-10s | %-8s\n' \
            "#$i" "$(first_escape "$vlog")" "$(first_escape "$glog")" \
            "$(last_miou "$vlog")" "$(last_miou "$glog")" \
            "$(guard_events "$glog")"
    done
    echo
    echo "失败 run 数: $FAIL / $((AB_NRUNS * 2))"
    echo "明细: grep '\\[GUARD\\]' $AB_DIR/ab_guard_*.log 查看每次激活/关闭"
} | tee "$SUM"

echo "[guard_ab] 全部完成, 汇总见 $SUM"
