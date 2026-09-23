#!/usr/bin/env bash
# A 档入口：一把锁里跑两步，两步吃同一份 fixture。
#   step1  PTO  sparse_attn_test              ←→ golden 全输出（attn+RoPE+o_proj 融合边界）
#   step2  vendor npu_sparse_attn_sharedkv    ←→ golden stage-1（只有 attn）
# fixture 由 step1 的 --save-data 落盘，step2 从 data/in 读，所以不是"各造一份随机数"。
#
# 锁内的活儿写成独立脚本再交给 task-submit：`--run` 收多行字符串会被吞掉，
# 表现为任务 exit=0 但一行输出都没有。
set -uo pipefail
D=/data/sunkaixuan/codex_sh/csa_b_tier_20260917
V=/data/sunkaixuan/sunkaixuan_subdir/all_servings/.venv-dsv4-vllm310
PL=/data/sunkaixuan/sunkaixuan_subdir/all_libs/pypto-lib-csa-20260917
PP=/data/sunkaixuan/yj_subdir/kernel-csa-feat-pypto
OUT=/data/sunkaixuan/skx_log_output/csa_b_tier/atier_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"/{pto,vendor,work,tmp,home,ascend,cache}
echo "[a-tier] OUT=$OUT"

cat > "$OUT/inner.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
# task-submit 的卡锁是建议性的，不导出 ASCEND_RT_VISIBLE_DEVICES，锁内必须自己导，
# 否则 torch 的 device 0 落到别人的物理卡上，表现为 507014 aicore timeout。
export ASCEND_RT_VISIBLE_DEVICES=\$TASK_DEVICE
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 SOC_VERSION=ascend910_9392
export PTOAS_ROOT=/usr/local/ptoas/0.61
export PYPTO_ROOT=$PP
export PYPTO_LIB_ROOT=$PL
export PTO_ISA_ROOT=$PP/runtime/build/pto-isa
export PYTHONPATH=$PL:$PL/models/deepseek_v4_flash_mtp:$PP/python:$PP/runtime:$PP/runtime/python
export HOME=$OUT/home TMPDIR=$OUT/tmp
export ASCEND_CACHE_PATH=$OUT/cache ASCEND_WORK_PATH=$OUT/work
export ASCEND_PROCESS_LOG_PATH=$OUT/ascend
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

cd $OUT/work || exit 1
echo "=== step1: PTO sparse_attn_test vs golden (save fixture) ==="
$V/bin/python $D/csa_b_tier.py --out-dir $OUT/pto --no-profile --save-data
s1=\$?
echo "=== step1 exit=\$s1 ==="

DATA=\$(cat $OUT/pto/data_dir.txt 2>/dev/null)
if [ -z "\$DATA" ] || [ ! -d "\$DATA/in" ]; then
    echo "[a-tier] step1 没落下 fixture (data_dir=\$DATA)，step2 退回自造输入"
    DATA=""
fi
echo "=== step2: vendor sparse_attn_sharedkv vs golden stage-1 (fixture=\$DATA) ==="
$V/bin/python $D/csa_a_tier_compare.py --out-dir $OUT/vendor \${DATA:+--data-dir "\$DATA"}
s2=\$?
echo "=== step2 exit=\$s2 ==="
[ "\$s1" = 0 ] && [ "\$s2" = 0 ]
EOF
chmod +x "$OUT/inner.sh"

task-submit --device auto --max-time "${MAX_TIME:-3600}" --timeout "${WAIT_TIMEOUT:-4200}" \
  --run "bash $OUT/inner.sh" 2>&1 | tee "$OUT/task.log"
echo "[a-tier] 产物: $OUT"
