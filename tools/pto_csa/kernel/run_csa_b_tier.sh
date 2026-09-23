#!/usr/bin/env bash
# B 档入口：拿设备锁，在锁内跑 PTO 版 CSA 并采 trace。
#
# 注意 task-submit 的卡锁是建议性的、不导出 ASCEND_RT_VISIBLE_DEVICES，
# 锁内必须自己 export，否则会跑到别人的物理卡上（表现为 507014 aicore timeout）。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_ROOT="${RUN_ROOT:-/data/sunkaixuan/skx_log_output/csa_b_tier}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$RUN_ROOT/pto_$TS"
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$RUN_ROOT/latest_out.txt"

echo "[b-tier] OUT=$OUT"
task-submit --device auto --max-time "${MAX_TIME:-2400}" --timeout "${WAIT_TIMEOUT:-2700}" \
    --run "export ASCEND_RT_VISIBLE_DEVICES=\$TASK_DEVICE; \
           source $HERE/env_pto_csa.sh >/dev/null; \
           cd \$PYPTO_MODEL_DIR && \$PTO_PY $HERE/csa_b_tier.py --out-dir $OUT ${EXTRA_ARGS:-}" \
    2>&1 | tee "$OUT/task_submit.log"
rc=${PIPESTATUS[0]}

echo
echo "[b-tier] 产物: $OUT"
[ -f "$OUT/trace.json" ] && echo "[b-tier] trace: $OUT/trace.json ($(stat -c%s "$OUT/trace.json") 字节)"
exit "$rc"
