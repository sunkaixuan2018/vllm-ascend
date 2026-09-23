#!/usr/bin/env bash
# Qwen3-14B 压测 + profiling 用例执行体 —— 在 task-submit 已分配的卡上跑完整一轮。
#
# 对标 CI 机的 vllm-qwen3 容器（docker inspect 抠出来的原始参数）：
#   vllm serve <model> --served-model-name qwen3 --trust-remote-code
#     --profiler-config '{"profiler":"torch","torch_profiler_dir":...,"torch_profiler_with_stack":false}'
#     --async-scheduling --tensor-parallel-size 1 --max-model-len 5500
#     --max-num-batched-tokens 40960 --port 8113 --block-size 128
#     --gpu-memory-utilization 0.9 --enable-logging-iteration-details --no-enable-prefix-caching
# CI 用 docker 起，本机无 docker 权限，改成 setsid 原生进程（对应关系见 dsv4_case_inner.sh 头注释）。
#
# 客户端就是 CI 那份 profile-util/profilling/ascendc_profile.py 的移植版。
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env_dsv4_vllm.sh"

: "${OUT:?OUT 未设置}"
MODEL="${MODEL:-/data/models/Qwen3-14B}"
SERVED_NAME="${SERVED_NAME:-qwen3}"
PORT="${PORT:-8113}"
URL="http://127.0.0.1:${PORT}"
DEVICES="${DEVICES:-${TASK_DEVICE:?既没给 DEVICES 也没有 TASK_DEVICE}}"
if [[ "$DEVICES" == *-* ]]; then
    DEVICES=$(awk -F- '{s="";for(i=$1;i<=$2;i++)s=s (s?",":"") i;print s}' <<<"$DEVICES")
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-5500}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-40960}"
CONCURRENCY="${CONCURRENCY:-16}"
REQUESTS="${REQUESTS:-32}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
PROMPT="${PROMPT:-}"
VERBOSE="${VERBOSE:-0}"
PROFILE="${PROFILE:-1}"
HEALTH_TRIES="${HEALTH_TRIES:-300}"
CLIENT="${CLIENT:-$HERE/ascendc_profile.py}"

TP=$(awk -F, '{print NF}' <<<"$DEVICES")
mkdir -p "$OUT/profile"
SERVER_LOG="$OUT/server.log"
SERVER_PID=""

collect_meta() {
    {
        echo "OUT=$OUT"
        echo "MODEL=$MODEL"
        echo "SERVED_NAME=$SERVED_NAME"
        echo "ASCEND_RT_VISIBLE_DEVICES=$DEVICES"
        echo "TP=$TP"
        echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
        echo "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
        echo "CONCURRENCY=$CONCURRENCY"
        echo "REQUESTS=$REQUESTS"
        echo "OUTPUT_TOKENS=$OUTPUT_TOKENS"
        echo "PROFILE=$PROFILE"
        echo "VLLM_SRC=$DSV4_VLLM_SRC"
        echo "VLLM_ASCEND_SRC=$DSV4_VLLM_ASCEND_SRC"
        git -C "$DSV4_VLLM_ASCEND_SRC" log -1 --format="VLLM_ASCEND_COMMIT=%H" 2>/dev/null
        date --iso-8601=seconds
    } > "$OUT/run_meta.txt"
    npu-smi info > "$OUT/device_before.txt" 2>&1 || true
}

stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 0
    kill -TERM -- "-$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
    pkill -KILL -u "$(id -u)" -f "vllm serve $MODEL .*--port $PORT" 2>/dev/null
}
trap stop_server EXIT

start_server() {
    if ss -ltn | awk '{print $4}' | grep -q ":${PORT}\$"; then
        echo "端口 ${PORT} 已被占用" >&2; return 1
    fi
    export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"

    EXTRA=()
    if [ "$PROFILE" = "1" ]; then
        # 这个 release 的 profiler 走 --profiler-config，不是 VLLM_TORCH_PROFILER_DIR
        EXTRA+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$OUT/profile\", \"torch_profiler_with_stack\": false}")
    fi

    setsid vllm serve "$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --trust-remote-code \
        --async-scheduling \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --port "$PORT" \
        --block-size 128 \
        --gpu-memory-utilization 0.9 \
        --enable-logging-iteration-details \
        --no-enable-prefix-caching \
        ${EXTRA[@]+"${EXTRA[@]}"} \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$OUT/server_pid.txt"
    echo "[server] pid=$SERVER_PID devices=$DEVICES tp=$TP model=$MODEL"
}

wait_for_health() {
    for i in $(seq 1 "$HEALTH_TRIES"); do
        if curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "$URL/health" >/dev/null 2>&1; then
            echo "READY iteration=$i" | tee -a "$OUT/wait_health.log"; return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "STOPPED iteration=$i" | tee -a "$OUT/wait_health.log"; return 1
        fi
        (( i % 10 == 0 )) && { echo "WAIT iteration=$i"; tail -n 40 "$SERVER_LOG"; } >> "$OUT/wait_health.log"
        sleep 3
    done
    echo "TIMEOUT after $HEALTH_TRIES tries" | tee -a "$OUT/wait_health.log"; return 1
}

main() {
    test -d "$MODEL" || { echo "模型目录不存在: $MODEL" >&2; return 1; }
    test -f "$CLIENT" || { echo "客户端不存在: $CLIENT" >&2; return 1; }
    collect_meta

    start_server || return 1
    if ! wait_for_health; then
        echo "status=server_unhealthy" > "$OUT/status.txt"
        tail -n 120 "$SERVER_LOG" >&2
        return 1
    fi

    curl --noproxy '*' -fsS "$URL/v1/models" > "$OUT/models.json" 2>/dev/null || true

    ARGS=(--base-url "$URL" --model "$SERVED_NAME"
          --profile-dir "$OUT/profile" --server-log "$SERVER_LOG"
          --concurrency "$CONCURRENCY" --requests "$REQUESTS"
          --output-tokens "$OUTPUT_TOKENS")
    [ -n "$PROMPT" ] && ARGS+=(-p "$PROMPT")
    [ "$VERBOSE" = "1" ] && ARGS+=(-v)

    local rc=0
    if python3 "$CLIENT" "${ARGS[@]}" > "$OUT/client.log" 2>&1; then
        echo "status=passed" > "$OUT/status.txt"
    else
        echo "status=client_failed" > "$OUT/status.txt"
        tail -n 60 "$OUT/client.log" >&2
        rc=1
    fi

    npu-smi info > "$OUT/device_after.txt" 2>&1 || true
    date --iso-8601=seconds > "$OUT/end_time.txt"
    tail -n 60 "$OUT/client.log"
    return "$rc"
}

main "$@"
