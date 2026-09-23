#!/usr/bin/env bash
# 原生 vllm-ascend 三层 vs 换成 PTO CSA 的三层，各采一份 decode 段 trace。
#
# 两轮**同一个 cp310 venv、同样的 serve 参数**，唯一差别是 PTO_CSA —— 否则比的就不是
# CSA 这一处改动。串行跑：机器是共用的，两轮同时排队会互相抢卡，端口也会撞
# （run_dsv4_mtp_vllm.sh 在提交前挑端口，两条都还没 bind 时会挑到同一个）。
#
# 默认**开 aclgraph**（真实部署就是开的）。EAGER=1 可退回 --enforce-eager 做对照。
set -uo pipefail
H=/data/sunkaixuan/codex_sh/dsv4_vllm_20260917
LOG=/data/sunkaixuan/skx_log_output/csa_b_tier
MODEL=/data/sunkaixuan/skx_log_output/dsv4_vllm/models/official-l3
V310=/data/sunkaixuan/sunkaixuan_subdir/all_servings/.venv-dsv4-vllm310
TAG="${TAG:-aclgraph}"

common() {
    export MODEL DEVICE_NUM=1 BATCH_SIZE=4 MTP=0
    export PROMPT_TOKENS=4096 MAX_TOKENS=24
    export PROFILE=1 PROFILE_TOKENS="${PROFILE_TOKENS:-8}" PROFILE_WARMUP=1
    # 空的 SERVE_EXTRA 就是 aclgraph（vllm-ascend 默认 PIECEWISE + npugraph_ex）。
    export SERVE_EXTRA="${EAGER:+--enforce-eager}"
    export DSV4_VENV=$V310
    export VLLM_ASCEND_PROFILER_LEVEL="${VLLM_ASCEND_PROFILER_LEVEL:-Level1}"
    export MAX_TIME=7200 WAIT_TIMEOUT=7200
    # 采 trace 时不要开逐入参校验：它每步做大量 .item()，会把 host 侧时间线搅浑。
    unset PTO_CSA_VERIFY PTO_CSA_DUMP PTO_CSA_PROBE
}

echo "===== 第 1 轮: 原生 vllm-ascend（vendor CSA）  mode=${EAGER:+eager}${EAGER:-aclgraph} ====="
( common
  export PORT=8131
  unset PTO_CSA PTO_CSA_REPORT PYPTO_ROOT PYPTO_LIB_ROOT
  cd $H && bash run_dsv4_mtp_vllm.sh ) > "$LOG/trace_native_$TAG.log" 2>&1
echo "第 1 轮 exit=$? $(grep -E '^\[dsv4\] status' "$LOG/trace_native_$TAG.log" | tail -1)"

echo "===== 第 2 轮: PTO CSA 替换  mode=${EAGER:+eager}${EAGER:-aclgraph} ====="
( common
  export PORT=8132
  export PTO_CSA=1
  export PTO_CSA_REPORT=$LOG/trace_pto_${TAG}_report.json
  export PYPTO_ROOT=/data/sunkaixuan/yj_subdir/kernel-csa-feat-pypto
  export PYPTO_LIB_ROOT=/data/sunkaixuan/sunkaixuan_subdir/all_libs/pypto-lib-csa-20260917
  export PTOAS_ROOT=/usr/local/ptoas/0.61
  cd $H && bash run_dsv4_mtp_vllm.sh ) > "$LOG/trace_pto_$TAG.log" 2>&1
echo "第 2 轮 exit=$? $(grep -E '^\[dsv4\] status' "$LOG/trace_pto_$TAG.log" | tail -1)"

echo "===== 产物 ====="
for f in "$LOG/trace_native_$TAG.log" "$LOG/trace_pto_$TAG.log"; do
    o=$(grep -o 'OUT=/data[^ ]*' "$f" | head -1 | cut -d= -f2)
    echo "$f -> $o"
    [ -n "$o" ] && find "$o/profile" -name 'trace_view.json' -printf '   %p  %s bytes\n' 2>/dev/null
    # 替换在 aclgraph 下会不会被静默跳过/算错，看逐步对拍
    [ -n "$o" ] && grep -E '\[pto-csa\]' "$o/server.log" 2>/dev/null | head -8
done
