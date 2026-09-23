#!/usr/bin/env bash
# Qwen3-14B 压测 + profiling 入口 —— 拿设备锁，把整轮交给 task-submit 执行。
#
# 这是 CI 机上真正跑出过数据的那条用例（profile-util/profilling/ascendc_profile.py），
# DSV4 那条见 run_dsv4_mtp_vllm.sh。
#
# 用法:
#   ./run_qwen3_vllm.sh                       # 单卡，默认并发 16 / 32 请求 / 输出 128 token
#   VERBOSE=1 ./run_qwen3_vllm.sh             # 逐请求、逐算子明细
#   PROMPT=prompt/prompt_1024.txt ./run_qwen3_vllm.sh
#   DEVICE_NUM=2 ./run_qwen3_vllm.sh          # TP=2
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODEL="${MODEL:-/data/models/Qwen3-14B}"
DEVICE_NUM="${DEVICE_NUM:-1}"
MAX_TIME="${MAX_TIME:-3600}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-3600}"
RUN_ROOT="${RUN_ROOT:-/data/sunkaixuan/skx_log_output/dsv4_vllm}"

# 多人共用，8113（CI 用的那个）可能被占；排队前先挑一个空闲端口
if [ -z "${PORT:-}" ]; then
    for p in $(seq 8113 8140); do
        ss -ltn | awk '{print $4}' | grep -q ":${p}$" || { PORT=$p; break; }
    done
    : "${PORT:?8113-8140 全被占用，手动指定 PORT=}"
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT="$RUN_ROOT/qwen3_vllm_${TS}_tp${DEVICE_NUM}"
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$RUN_ROOT/latest_out.txt"

if [ -n "${DEVICES:-}" ]; then
    DEV_ARGS=(--device "$DEVICES")
else
    DEV_ARGS=(--device auto --device-num "$DEVICE_NUM")
fi

echo "[qwen3] OUT=$OUT"
echo "[qwen3] model=$MODEL tp=$DEVICE_NUM port=$PORT"

task-submit "${DEV_ARGS[@]}" \
    --timeout "$WAIT_TIMEOUT" --max-time "$MAX_TIME" \
    --env OUT="$OUT" \
    --env MODEL="$MODEL" \
    --env PORT="$PORT" \
    --env SERVED_NAME="${SERVED_NAME:-qwen3}" \
    --env MAX_MODEL_LEN="${MAX_MODEL_LEN:-5500}" \
    --env CONCURRENCY="${CONCURRENCY:-16}" \
    --env REQUESTS="${REQUESTS:-32}" \
    --env OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}" \
    --env PROMPT="${PROMPT:-}" \
    --env VERBOSE="${VERBOSE:-0}" \
    --env PROFILE="${PROFILE:-1}" \
    --run "bash $HERE/qwen3_case_inner.sh" 2>&1 | tee "$OUT/task_submit.log"
rc=${PIPESTATUS[0]}

echo
echo "[qwen3] status: $(cat "$OUT/status.txt" 2>/dev/null || echo "(无 status.txt)")"
echo "[qwen3] 产物: $OUT"
exit "$rc"
