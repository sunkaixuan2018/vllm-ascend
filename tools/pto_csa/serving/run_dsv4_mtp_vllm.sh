#!/usr/bin/env bash
# DSV4-Flash MTP vLLM 用例入口（myserver）—— 拿设备锁，然后把整轮交给 task-submit 执行。
#
# 卡由 task-submit 分配并在整轮期间持有：服务启动、健康等待、客户端压测、回收都在同一个
# task 里，避免"每次迭代重新抢锁"导致和别人的任务交叉占卡。
#
# 用法:
#   ./run_dsv4_mtp_vllm.sh                      # 默认 DP=卡数, bs=4
#   BATCH_SIZE=8 ./run_dsv4_mtp_vllm.sh
#   DEVICE_NUM=8 ./run_dsv4_mtp_vllm.sh         # 只要 8 张卡（DP8）
#   DEVICES=1,3,5,7 ./run_dsv4_mtp_vllm.sh      # 指定卡号（精确锁）
#   MODEL=/data/models/dsv4-flash-0731-dspark-w8a8 ./run_dsv4_mtp_vllm.sh
#   PROFILE=1 ./run_dsv4_mtp_vllm.sh            # 同时开 torch profiler（产物在 $OUT/profile）
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

MODEL="${MODEL:-/data/models/dsv4-flash-w8a8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE_NUM="${DEVICE_NUM:-16}"
# 机器是多人共用的，8113（CI 机用的那个）可能已被别人的服务占着。没显式指定就从 8113
# 往上找一个空闲端口——放在排队之前挑，免得拿到卡才发现端口冲突、白占一轮锁。
if [ -z "${PORT:-}" ]; then
    for p in $(seq 8113 8140); do
        ss -ltn | awk '{print $4}' | grep -q ":${p}$" || { PORT=$p; break; }
    done
    : "${PORT:?8113-8140 全被占用，手动指定 PORT=}"
fi
PROFILE="${PROFILE:-0}"
MAX_TIME="${MAX_TIME:-5400}"          # task-submit 默认 300s，远不够一次 DSV4 起服务
WAIT_TIMEOUT="${WAIT_TIMEOUT:-5400}"
RUN_ROOT="${RUN_ROOT:-/data/sunkaixuan/skx_log_output/dsv4_vllm}"

TS=$(date +%Y%m%d_%H%M%S)
TAG="dp${DEVICE_NUM}_bs${BATCH_SIZE}"
OUT="$RUN_ROOT/dsv4_mtp_vllm_${TS}_${TAG}"
mkdir -p "$OUT"
# The AICPU/CCECPU device log defaults to a directory shared with every other
# user on the box. The driver only fopens it, so the directory has to exist.
export ASCEND_PROCESS_LOG_PATH="${ASCEND_PROCESS_LOG_PATH:-$OUT/ascend}"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
printf '%s\n' "$OUT" > "$RUN_ROOT/latest_out.txt"

# --device auto 只在白名单内挑卡（本机默认白名单是奇数卡）；要整机 16 卡得放开白名单。
if [ -n "${DEVICES:-}" ]; then
    DEV_ARGS=(--device "$DEVICES")
else
    DEV_ARGS=(--device auto --device-num "$DEVICE_NUM")
    [ "$DEVICE_NUM" -gt 8 ] && DEV_ARGS=(--ignore-whitelist "${DEV_ARGS[@]}")
fi

echo "[dsv4] OUT=$OUT"
echo "[dsv4] device log -> $ASCEND_PROCESS_LOG_PATH"
echo "[dsv4] model=$MODEL bs=$BATCH_SIZE device_num=$DEVICE_NUM profile=$PROFILE layers=${NUM_LAYERS:-all}"
task-submit --list 2>&1 | sed -n '1,8p'

task-submit "${DEV_ARGS[@]}" \
    --timeout "$WAIT_TIMEOUT" --max-time "$MAX_TIME" \
    --env OUT="$OUT" \
    --env MODEL="$MODEL" \
    --env PORT="$PORT" \
    --env BATCH_SIZE="$BATCH_SIZE" \
    --env PROFILE="$PROFILE" \
    --env PROMPT_TOKENS="${PROMPT_TOKENS:-8192}" \
    --env MAX_TOKENS="${MAX_TOKENS:-106}" \
    --env MAX_MODEL_LEN="${MAX_MODEL_LEN:-8704}" \
    --env SEED="${SEED:-1807}" \
    --env HEALTH_TRIES="${HEALTH_TRIES:-500}" \
    --env NUM_LAYERS="${NUM_LAYERS:-}" \
    --env SERVE_EXTRA="${SERVE_EXTRA:-}" \
    --env MTP="${MTP:-1}" \
    --env SPEC_CONFIG="${SPEC_CONFIG:-}" \
    --env PTO_CSA_PROBE="${PTO_CSA_PROBE:-}" \
    --env PTO_ATTN_PROBE="${PTO_ATTN_PROBE:-}" \
    --env GPU_UTIL="${GPU_UTIL:-}" \
    --env PTO_ATTN_COMPARE="${PTO_ATTN_COMPARE:-}" \
    --env PTO_ATTN_REPLACE="${PTO_ATTN_REPLACE:-}" \
    --env PTO_ATTN_SEQ="${PTO_ATTN_SEQ:-}" \
    --env PTO_ATTN_TP="${PTO_ATTN_TP:-}" \
    --env PTO_DSPARK_SPEC_TOKENS="${PTO_DSPARK_SPEC_TOKENS:-}" \
    --env PYPTO_CACHE="${PYPTO_CACHE:-}" \
    --env DSV4_VENV="${DSV4_VENV:-}" \
    --env PTO_CSA="${PTO_CSA:-}" \
    --env PTO_CSA_MODE="${PTO_CSA_MODE:-}" \
    --env PTO_CSA_REPORT="${PTO_CSA_REPORT:-}" \
    --env PTO_CSA_DUMP="${PTO_CSA_DUMP:-}" \
    --env PTO_CSA_VERIFY="${PTO_CSA_VERIFY:-}" \
    --env PROFILE_TOKENS="${PROFILE_TOKENS:-}" \
    --env PROFILE_WARMUP="${PROFILE_WARMUP:-}" \
    --env VLLM_ASCEND_PROFILER_LEVEL="${VLLM_ASCEND_PROFILER_LEVEL:-}" \
    --env PYPTO_ROOT="${PYPTO_ROOT:-}" \
    --env PYPTO_LIB_ROOT="${PYPTO_LIB_ROOT:-}" \
    --env ASCEND_PROCESS_LOG_PATH="$ASCEND_PROCESS_LOG_PATH" \
    --env PTOAS_ROOT="${PTOAS_ROOT:-}" \
    --run "bash $HERE/dsv4_case_inner.sh" 2>&1 | tee "$OUT/task_submit.log"
rc=${PIPESTATUS[0]}

echo
echo "[dsv4] status: $(cat "$OUT/status.txt" 2>/dev/null || echo "(无 status.txt)")"
echo "[dsv4] 产物目录: $OUT"
echo "[dsv4]   server.log / wait_health.log  服务端日志与健康等待过程"
echo "[dsv4]   client.log  + client/         客户端压测输出（TPOT/TTFT/吞吐）"
echo "[dsv4]   run_meta.txt                 本轮实际参数与 vllm-ascend commit"
[ "$PROFILE" = "1" ] && echo "[dsv4]   profile/                     torch profiler 产物"
exit "$rc"
