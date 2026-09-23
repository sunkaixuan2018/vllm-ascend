#!/usr/bin/env bash
# 新栈的三道验证，一把锁里串行跑完：
#   1 adapter   适配器离线自查（不需要模型权重）      判据 within_tol=true, max_abs≈1.5e-5
#   2 gate      PyPTO dispatch 与 torch_npu 同进程共存  判据 canary 完好、vendor 重跑逐位相同
#   3 atier     PTO 全路径对 golden + vendor 注意力对 golden stage-1
# 这三道都不碰模型权重，跑完才谈整网。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/data/sunkaixuan/skx_log_output/own_stack/checks_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"/{adapter,gate,atier/pto,atier/vendor,work,tmp,home,ascend,cache}
echo "[own-checks] OUT=$OUT"

cat > "$OUT/inner.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
# task-submit 的卡锁是建议性的，不导出 ASCEND_RT_VISIBLE_DEVICES，锁内必须自己导，
# 否则 torch 的 device 0 落到别人的物理卡上，表现为 507014 aicore timeout。
export ASCEND_RT_VISIBLE_DEVICES=\$TASK_DEVICE
source $HERE/env_own.sh >/dev/null 2>&1
export HOME=$OUT/home TMPDIR=$OUT/tmp
export ASCEND_CACHE_PATH=$OUT/cache ASCEND_WORK_PATH=$OUT/work
export ASCEND_PROCESS_LOG_PATH=$OUT/ascend
# pypto-lib 的模型脚本用平铺 import（from config import FLASH），模型目录本身也要在路径上。
export PYTHONPATH="\$PYPTO_LIB_ROOT:\$PYPTO_LIB_ROOT/models/deepseek_v4_flash_mtp:\$PYPTO_ROOT/python:\$PYPTO_ROOT/runtime:\$PYPTO_ROOT/runtime/python\${PYTHONPATH:+:\$PYTHONPATH}"
PY=\$OWN_VENV/bin/python
echo "[inner] python=\$(\$PY -V 2>&1)  device=\$TASK_DEVICE"
cd $OUT/work || exit 1

echo; echo "=== 1/3 adapter 离线自查 ==="
\$PY $HERE/csa_adapter_check.py --out-dir $OUT/adapter
a=\$?; echo "=== adapter exit=\$a ==="

echo; echo "=== 2/3 共存门禁 ==="
\$PY $HERE/pto_in_vllm_process.py --out-dir $OUT/gate
g=\$?; echo "=== gate exit=\$g ==="

echo; echo "=== 3/3a A 档 step1: PTO 全路径对 golden（落盘 fixture）==="
\$PY $HERE/csa_b_tier.py --out-dir $OUT/atier/pto --no-profile --save-data
p=\$?; echo "=== atier-pto exit=\$p ==="

DATA=\$(cat $OUT/atier/pto/data_dir.txt 2>/dev/null)
[ -d "\$DATA/in" ] || DATA=""
echo; echo "=== 3/3b A 档 step2: vendor 注意力对 golden stage-1 (fixture=\$DATA) ==="
\$PY $HERE/csa_a_tier_compare.py --out-dir $OUT/atier/vendor \${DATA:+--data-dir "\$DATA"}
v=\$?; echo "=== atier-vendor exit=\$v ==="

echo; echo "=== 汇总 adapter=\$a gate=\$g atier_pto=\$p atier_vendor=\$v ==="
[ "\$a" = 0 ] && [ "\$g" = 0 ] && [ "\$p" = 0 ] && [ "\$v" = 0 ]
EOF
chmod +x "$OUT/inner.sh"

task-submit --device auto --max-time "${MAX_TIME:-3600}" --timeout "${WAIT_TIMEOUT:-4200}" \
  --run "bash $OUT/inner.sh" 2>&1 | tee "$OUT/task.log"
echo "[own-checks] 产物: $OUT"
