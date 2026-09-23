#!/usr/bin/env bash
# 第 4 步：kernel 模式三步验证（handoff 的判据，一步不减）。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/data/sunkaixuan/skx_log_output/own_stack/kernel_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"/{work,tmp,home,ascend,cache}
echo "[kernel] OUT=$OUT"

cat > "$OUT/inner.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
# task-submit 的卡锁是建议性的，不导出 ASCEND_RT_VISIBLE_DEVICES，锁内必须自己导。
export ASCEND_RT_VISIBLE_DEVICES=\$TASK_DEVICE
source $HERE/env_own.sh >/dev/null 2>&1
export HOME=$OUT/home TMPDIR=$OUT/tmp
export ASCEND_CACHE_PATH=$OUT/cache ASCEND_WORK_PATH=$OUT/work
export ASCEND_PROCESS_LOG_PATH=$OUT/ascend
export PYTHONPATH="\$PYPTO_LIB_ROOT:\$PYPTO_LIB_ROOT/models/deepseek_v4_flash_mtp:\$PYPTO_ROOT/python:\$PYPTO_ROOT/runtime:\$PYPTO_ROOT/runtime/python\${PYTHONPATH:+:\$PYTHONPATH}"
PY=\$OWN_VENV/bin/python
echo "[inner] \$(\$PY -V 2>&1)  device=\$TASK_DEVICE"
cd $OUT/work || exit 1
\$PY $HERE/kernel_mode_check.py --out-dir $OUT
rc=\$?
echo "=== kernel_mode_check exit=\$rc ==="
# 段错误(139)时按 handoff 的要求补一份 backtrace
if [ "\$rc" = 139 ] || [ "\$rc" = 134 ]; then
    echo "=== 段错误，抓 backtrace ==="
    command -v gdb >/dev/null 2>&1 &&
      gdb -batch -ex run -ex "bt 60" --args \$PY $HERE/kernel_mode_check.py --out-dir $OUT/gdb 2>&1 | tail -70 ||
      echo "  本机没有 gdb"
fi
exit \$rc
EOF
chmod +x "$OUT/inner.sh"

task-submit --device auto --max-time "${MAX_TIME:-2400}" --timeout "${WAIT_TIMEOUT:-3000}" \
  --run "bash $OUT/inner.sh" 2>&1 | tee "$OUT/task.log"
echo "[kernel] 产物: $OUT"
