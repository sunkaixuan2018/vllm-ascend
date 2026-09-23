#!/usr/bin/env bash
# DSV4-Flash MTP vLLM 单用例执行体 —— 在 task-submit 已分配的卡上跑完整一轮。
#
# 由 run_dsv4_mtp_vllm.sh 通过 task-submit --run 调起，不要直接裸跑：
# 裸跑不持有设备锁，会和别人的任务抢同一批卡。
#
# 与 CI 机 run_dsv4_mtp_vllm_sweep.sh 的对应关系（CI 用 docker，本机无 docker 权限）：
#   docker run -d …                  → setsid vllm serve（独立进程组，便于整组回收）
#   docker inspect .State.Running    → kill -0 "$SERVER_PID"
#   docker logs <c>                  → $OUT/server.log
#   docker stop/rm                   → kill -TERM/-KILL 整个进程组
#   npu-lock --status                → task-submit --list
#   DEVICES=0,…,15（写死）           → $TASK_DEVICE（task-submit 实际分配的卡）
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env_dsv4_vllm.sh"

# PTO 层级替换打开时只需要 PyPTO 工具链（DSL + 编译器 + 运行时）。kernel 源码内联在
# vllm_ascend/attention/pto_kernels/ 下，走包内相对 import，pypto-lib 不上路径。
# 解释器由 env_own.sh 固定（own_stack 的 cp311）；_task_interface 是按该 ABI 编的。
# The attention-level entry points need the same toolchain, so any of them arms it.
PTO_NEEDS_PYPTO=0
for _v in "${PTO_CSA:-}" "${PTO_ATTN_COMPARE:-}" "${PTO_ATTN_REPLACE:-}"; do
    [ -n "$_v" ] && [ "$_v" != "0" ] && PTO_NEEDS_PYPTO=1
done
if [ "$PTO_NEEDS_PYPTO" = "1" ]; then
    : "${PYPTO_ROOT:?PTO 替换打开时必须给 PYPTO_ROOT}"
    export PTO_ISA_ROOT="${PTO_ISA_ROOT:-$PYPTO_ROOT/runtime/build/pto-isa}"
    export PYTHONPATH="$PYPTO_ROOT/python:$PYPTO_ROOT/runtime:$PYPTO_ROOT/runtime/python${PYTHONPATH:+:$PYTHONPATH}"
    echo "[dsv4-vllm] PTO 工具链已上路径 pypto=$PYPTO_ROOT ptoas=${PTOAS_ROOT:-}"
    echo "[dsv4-vllm] python=$(command -v python)"
fi

: "${OUT:?OUT 未设置}"
MODEL="${MODEL:?MODEL 未设置}"
PORT="${PORT:-8113}"
URL="http://127.0.0.1:${PORT}"
DEVICES="${DEVICES:-${TASK_DEVICE:?既没给 DEVICES 也没有 TASK_DEVICE}}"
# task-submit 给的卡号可能是 "2,4,6" 也可能是 "2-6" 区间形式；
# ASCEND_RT_VISIBLE_DEVICES 只认逗号列表，这里统一展开。
if [[ "$DEVICES" == *-* ]]; then
    DEVICES=$(awk -F- '{s="";for(i=$1;i<=$2;i++)s=s (s?",":"") i;print s}' <<<"$DEVICES")
fi
# MTP=0 关掉投机解码（截层冒烟时用：MTP 权重在 checkpoint 的第 43 层，截层后索引对不上）。
# 默认 JSON 单独放一个变量：直接写在 ${VAR:-默认} 里会被它自己的右花括号提前结束展开，
# 置空时反而得到一个孤立的 "}"，vllm 报
# "argument --speculative-config: Value } cannot be converted"。
MTP="${MTP:-1}"
SPEC_CONFIG_DEFAULT='{"method":"mtp","num_speculative_tokens":1}'
SPEC_CONFIG="${SPEC_CONFIG:-$SPEC_CONFIG_DEFAULT}"
[ "$MTP" = "1" ] || SPEC_CONFIG=""
BATCH_SIZE="${BATCH_SIZE:-4}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8192}"
MAX_TOKENS="${MAX_TOKENS:-106}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8704}"
SEED="${SEED:-1807}"
PROFILE="${PROFILE:-0}"
HEALTH_TRIES="${HEALTH_TRIES:-500}"
CLIENT="${CLIENT:-$HERE/dsv4_mtp_vllm_client.py}"
SERVE_EXTRA="${SERVE_EXTRA:-}"
# 只跑前 N 层。整模 43 层的权重按 EP 均分后仍放不下现有卡数，截断后可在 8 卡上跑通全链路。
# vllm-ascend 的 load_weights 取 params_dict 之前有 is_pp_missing_parameter 守卫，落在
# [start_layer, end_layer) 之外的层会被跳过，所以多出来的层不会报 KeyError。
NUM_LAYERS="${NUM_LAYERS:-}"

EXTRA_ARGS=()
# 注意：截层不要用 --hf-overrides。实测 `--hf-overrides {"num_hidden_layers":3}` 会让
# 注意力 linear 拿不到量化方法，加载权重时报
# KeyError: 'model.layers.0.self_attn.wo_b.scale'；同一份 checkpoint 不加这个开关能正常
# 载入（只是整模放不下小卡数）。要截层就用 make_ci_model_dir.sh 生成一份改过
# num_hidden_layers 的 config.json，把 MODEL 指过去。
if [ -n "$NUM_LAYERS" ]; then
    EXTRA_ARGS+=(--hf-overrides "{\"num_hidden_layers\": $NUM_LAYERS}")
fi
# 空 SPEC_CONFIG 表示不开 MTP 投机解码（截层冒烟时用得上：MTP 权重在 checkpoint 的
# 第 43 层，截层后索引对不上）。
if [ -n "$SPEC_CONFIG" ]; then
    EXTRA_ARGS+=(--speculative-config "$SPEC_CONFIG")
fi
if [ -n "$SERVE_EXTRA" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS+=($SERVE_EXTRA)
fi

DP=$(awk -F, '{print NF}' <<<"$DEVICES")
TP="${TP:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((PROMPT_TOKENS * BATCH_SIZE))}"

mkdir -p "$OUT/profile"
SERVER_LOG="$OUT/server.log"
SERVER_PID=""

collect_meta() {
    {
        echo "OUT=$OUT"
        echo "MODEL=$MODEL"
        echo "ASCEND_RT_VISIBLE_DEVICES=$DEVICES"
        echo "TP=$TP"
        echo "DP=$DP"
        echo "EP=$DP via --enable-expert-parallel"
        echo "LOCAL_BS=$BATCH_SIZE"
        echo "PROMPT_TOKENS=$PROMPT_TOKENS"
        echo "MAX_TOKENS=$MAX_TOKENS"
        echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
        echo "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
        echo "SPECULATIVE_CONFIG=${SPEC_CONFIG:-(disabled)}"
        echo "PROFILE=$PROFILE"
        echo "HF_OVERRIDES_NUM_LAYERS=${NUM_LAYERS:-none}"
        # 真正生效的层数看模型自己的 config.json（截层是写在那里的，不是命令行开关）
        python3 -c "
import json
d = json.load(open('$MODEL/config.json'))
print('MODEL_NUM_HIDDEN_LAYERS=%s' % d.get('num_hidden_layers'))
print('MODEL_MTP_LAYERS=%s' % d.get('num_nextn_predict_layers'))
" 2>/dev/null
        echo "HCCL_CONNECT_TIMEOUT=$HCCL_CONNECT_TIMEOUT"
        echo "VLLM_SRC=$DSV4_VLLM_SRC"
        echo "VLLM_ASCEND_SRC=$DSV4_VLLM_ASCEND_SRC"
        git -C "$DSV4_VLLM_ASCEND_SRC" log -1 --format="VLLM_ASCEND_COMMIT=%H" 2>/dev/null
        date --iso-8601=seconds
    } > "$OUT/run_meta.txt"
    task-submit --list > "$OUT/task_queue_before.txt" 2>&1 || true
    npu-smi info > "$OUT/device_before.txt" 2>&1 || true
}

stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 0
    kill -TERM -- "-$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
    # DP 子进程偶尔挂在父进程之外，按模型路径兜底清一遍自己的残留。
    pkill -KILL -u "$(id -u)" -f "vllm serve $MODEL .*--port $PORT" 2>/dev/null
    echo "server stopped (pid=$SERVER_PID)" >> "$OUT/server_stop.txt"
}
trap stop_server EXIT

start_server() {
    if ss -ltn | awk '{print $4}' | grep -q ":${PORT}\$"; then
        echo "port ${PORT} is already in use" >&2
        ss -ltn | grep ":${PORT}" >&2 || true
        return 1
    fi
    # 0.20.2 不再读 VLLM_TORCH_PROFILER_DIR，profiler 走 --profiler-config.* 命令行。
    # torch_profiler_with_stack 默认就是 true，显式写出来是因为它同时决定了
    # vllm-ascend 那层开不开 with_stack（没有 Python 栈，trace 里看不出某个 kernel
    # 是 vendor 下发的还是 PTO 下发的）。
    PROFILE_ARGS=()
    if [ "$PROFILE" = "1" ]; then
        PROFILE_ARGS=(--profiler-config.profiler=torch
                      --profiler-config.torch_profiler_dir="$OUT/profile"
                      --profiler-config.torch_profiler_with_stack=true)
    fi
    export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"

    setsid vllm serve "$MODEL" \
        --served-model-name dsv4 \
        --trust-remote-code \
        --async-scheduling \
        --tensor-parallel-size "$TP" \
        --data-parallel-size "$DP" \
        --data-parallel-size-local "$DP" \
        --data-parallel-backend mp \
        --enable-expert-parallel \
        --enable-ep-weight-filter \
        --expert-placement-strategy linear \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$BATCH_SIZE" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --port "$PORT" \
        --block-size 128 \
        --gpu-memory-utilization "${GPU_UTIL:-0.9}" \
        --kv-cache-dtype fp8 \
        --enable-logging-iteration-details \
        --no-enable-prefix-caching \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"} \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$OUT/server_pid.txt"
    echo "[server] pid=$SERVER_PID devices=$DEVICES dp=$DP bs=$BATCH_SIZE"
}

wait_for_health() {
    for i in $(seq 1 "$HEALTH_TRIES"); do
        if curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "$URL/health" >/dev/null 2>&1; then
            # /health 会在 engine 还没初始化完时就返回 200（DP + 多 API server 下
            # API server 先于 engine core 起来），此时打真实请求全是 400。
            # 所以再用一个 1-token 的真实补全确认引擎真的能干活。
            if curl --noproxy '*' --connect-timeout 2 --max-time 30 -fsS                  -H 'Content-Type: application/json'                  -d "{\"model\": \"dsv4\", \"prompt\": \"hi\", \"max_tokens\": 1}"                  "$URL/v1/completions" >/dev/null 2>&1; then
                echo "READY iteration=$i (engine 已可服务)" | tee -a "$OUT/wait_health.log"
                return 0
            fi
            if (( i % 10 == 0 )); then
                {
                    echo "--- health OK 但补全被拒 iteration=$i"
                    echo "  /v1/models:"
                    curl --noproxy '*' --max-time 10 -sS "$URL/v1/models" 2>&1 | head -c 600
                    echo
                    echo "  400 响应体:"
                    curl --noproxy '*' --max-time 20 -sS -H 'Content-Type: application/json'                         -d "{\"model\": \"dsv4\", \"prompt\": \"hi\", \"max_tokens\": 1}"                         "$URL/v1/completions" 2>&1 | head -c 800
                    echo
                } >> "$OUT/wait_health.log"
            fi
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "STOPPED iteration=$i" | tee -a "$OUT/wait_health.log"
            return 1
        fi
        if (( i % 10 == 0 )); then
            { echo "WAIT iteration=$i"; tail -n 60 "$SERVER_LOG"; } >> "$OUT/wait_health.log"
        fi
        sleep 3
    done
    echo "TIMEOUT after $HEALTH_TRIES tries" | tee -a "$OUT/wait_health.log"
    return 1
}

main() {
    test -d "$MODEL" || { echo "模型目录不存在: $MODEL" >&2; return 1; }
    test -f "$CLIENT" || { echo "客户端脚本不存在: $CLIENT" >&2; return 1; }
    collect_meta

    start_server || return 1
    if ! wait_for_health; then
        echo "status=server_unhealthy" > "$OUT/status.txt"
        tail -n 200 "$SERVER_LOG" >&2
        return 1
    fi

    curl --noproxy '*' -fsS "$URL/version"   > "$OUT/version.json"    2>/dev/null || true
    curl --noproxy '*' -fsS "$URL/v1/models" > "$OUT/models.json"     2>/dev/null || true
    curl --noproxy '*' -fsS "$URL/metrics"   > "$OUT/metrics_before.txt" 2>/dev/null || true
    npu-smi info > "$OUT/device_before_client.txt" 2>&1 || true

    local rc=0
    # PROFILE=1 时要的是 trace 不是 TPOT 统计：换成只录 decode 段的客户端。
    if [ "$PROFILE" = "1" ]; then
        if python3 "$HERE/profile_decode_client.py" \
                --base-url "$URL" \
                --model dsv4 \
                --profile-dir "$OUT/profile" \
                --out "$OUT/profile_client.json" \
                --batch-size "$BATCH_SIZE" \
                --prompt-tokens "$PROMPT_TOKENS" \
                --max-tokens "$MAX_TOKENS" \
                --profile-tokens "${PROFILE_TOKENS:-8}" \
                --warmup "${PROFILE_WARMUP:-1}" \
                --seed "$SEED" \
                > "$OUT/client.log" 2>&1; then
            echo "status=passed" > "$OUT/status.txt"
        else
            echo "status=client_failed" > "$OUT/status.txt"
            tail -n 80 "$OUT/client.log" >&2
            rc=1
        fi
        curl --noproxy '*' -fsS "$URL/metrics" > "$OUT/metrics_after.txt" 2>/dev/null || true
        cp -f "$SERVER_LOG" "$OUT/server_final.log" 2>/dev/null || true
        date --iso-8601=seconds > "$OUT/end_time.txt"
        return "$rc"
    fi
    if python3 "$CLIENT" \
            --base-url "$URL" \
            --model dsv4 \
            --output-dir "$OUT/client" \
            --batch-size "$BATCH_SIZE" \
            --prompt-tokens "$PROMPT_TOKENS" \
            --max-tokens "$MAX_TOKENS" \
            --warmup-batches "${WARMUP_BATCHES:-5}" \
            --measured-batches "${MEASURED_BATCHES:-1}" \
            --steady-skip "${STEADY_SKIP:-5}" \
            --timeout "${CLIENT_TIMEOUT:-3600}" \
            --seed "$SEED" \
            > "$OUT/client.log" 2>&1; then
        echo "status=passed" > "$OUT/status.txt"
    else
        echo "status=client_failed" > "$OUT/status.txt"
        tail -n 80 "$OUT/client.log" >&2
        rc=1
    fi

    curl --noproxy '*' -fsS "$URL/metrics" > "$OUT/metrics_after.txt" 2>/dev/null || true
    npu-smi info > "$OUT/device_after_client.txt" 2>&1 || true
    cp -f "$SERVER_LOG" "$OUT/server_final.log" 2>/dev/null || true
    date --iso-8601=seconds > "$OUT/end_time.txt"
    return "$rc"
}

main "$@"
