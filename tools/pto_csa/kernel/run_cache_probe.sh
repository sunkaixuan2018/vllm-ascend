#!/usr/bin/env bash
# PyPTO JIT 编译缓存的诊断入口：问清楚缓存为什么没落盘、重复调用有没有命中。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/data/sunkaixuan/skx_log_output/own_stack/cacheprobe_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"/{work,tmp,home,ascend,cache}
echo "[cache-probe] OUT=$OUT"

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
\$PY $HERE/cache_probe.py --out-dir $OUT
rc=\$?
echo "=== cache_probe exit=\$rc ==="
# 段错误(139)时补一份 backtrace，省得再跑一轮
if [ "\$rc" = 139 ] || [ "\$rc" = 134 ]; then
    echo "=== 段错误，抓 backtrace ==="
    command -v gdb >/dev/null 2>&1 &&
      gdb -batch -ex run -ex "bt 60" --args \$PY $HERE/cache_probe.py --out-dir $OUT/gdb 2>&1 | tail -70 ||
      echo "  本机没有 gdb"
fi
exit \$rc
EOF
chmod +x "$OUT/inner.sh"

task-submit --device auto --max-time "${MAX_TIME:-2400}" --timeout "${WAIT_TIMEOUT:-3000}" \
  --run "bash $OUT/inner.sh" 2>&1 | tee "$OUT/task.log"
echo "[cache-probe] 产物: $OUT"
