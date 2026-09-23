#!/usr/bin/env bash
# 等官方权重分片下完 → 组装 3 层模型目录 → 提交用例跑。
#
# 全程 detached 在服务器上跑，不依赖 ssh 链路。日志:
#   /data/sunkaixuan/skx_log_output/dsv4_vllm/auto_after_fetch.log
set -uo pipefail
HERE=/data/sunkaixuan/codex_sh/dsv4_vllm_20260917
L=/data/sunkaixuan/skx_log_output/dsv4_vllm
SRC=$L/models/official/DeepSeek-V4-Flash-w8a8-mtp
NEED="00001 00002 00003 00004 00005 00070"
DEADLINE=$(( $(date +%s) + ${MAX_WAIT_H:-8} * 3600 ))

say() { echo "[$(date +%F' '%T)] $*"; }

# safetensors 头部记了数据区长度，据此判断文件是否下全（避免拿半截文件去跑）
complete() {
    python3 - "$1" <<'PY'
import json, os, struct, sys
p = sys.argv[1]
try:
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
    sys.exit(0 if os.path.getsize(p) == 8 + n + end else 1)
except Exception:
    sys.exit(1)
PY
}

say "开始等待 ${NEED}"
while :; do
    missing=""
    for n in $NEED; do
        f=$SRC/quant_model_weights-$n-of-00070.safetensors
        complete "$f" || missing="$missing $n"
    done
    tk=$(stat -c%s "$SRC/tokenizer.json" 2>/dev/null || echo 0)
    [ "$tk" -lt 5000000 ] && missing="$missing tokenizer"

    if [ -z "$missing" ]; then
        say "分片与 tokenizer 齐全"
        break
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        say "超时放弃，仍缺:$missing"
        exit 1
    fi
    if ! pgrep -u "$(id -u)" -f "FilePath=quant_model_weights" >/dev/null &&
       ! pgrep -u "$(id -u)" -f "fetch_official_shards[.]sh" >/dev/null; then
        say "下载进程都没了但仍缺:$missing —— 重新拉一遍"
        "$HERE/fetch_official_shards.sh" ${missing// tokenizer/} >> "$L/fetch_shards.log" 2>&1
    fi
    sleep 120
done

say "组装模型目录"
out=$(LAYERS=3 MTP=0 "$HERE/make_official_l3_dir.sh" 2>&1)
echo "$out"
if ! grep -q "分片齐全" <<<"$out"; then
    say "组装未通过，停在这里"
    exit 1
fi

say "提交用例（2 卡 / bs=4 / 3 层 / 不开 MTP）"
cd "$HERE" || exit 1
MODEL=$L/models/official-l3 \
DEVICE_NUM=2 BATCH_SIZE=4 MTP=0 \
HEALTH_TRIES=400 MAX_TIME=5400 WAIT_TIMEOUT=5400 \
    ./run_dsv4_mtp_vllm.sh
rc=$?
O=$(cat "$L/latest_out.txt" 2>/dev/null)
say "用例结束 rc=$rc  产物=$O  status=$(cat "$O/status.txt" 2>/dev/null || echo 无)"
echo AUTO_AFTER_FETCH_DONE
