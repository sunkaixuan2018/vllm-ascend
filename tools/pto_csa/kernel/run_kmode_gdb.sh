#!/usr/bin/env bash
# A 路：把 kernel 模式的段错误定位到具体栈帧。
set -uo pipefail
D=/data/sunkaixuan/codex_sh/csa_b_tier_20260917
V=/data/sunkaixuan/sunkaixuan_subdir/all_servings/.venv-dsv4-vllm310
PL=/data/sunkaixuan/sunkaixuan_subdir/all_libs/pypto-lib-csa-20260917
PP=/data/sunkaixuan/yj_subdir/kernel-csa-feat-pypto
OUT=/data/sunkaixuan/skx_log_output/csa_b_tier/kmode_gdb_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"/{work,tmp,home,ascend,cache}
echo "[kmode] OUT=$OUT"

cat > "$OUT/inner.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
export ASCEND_RT_VISIBLE_DEVICES=\$TASK_DEVICE
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 SOC_VERSION=ascend910_9392
export PTOAS_ROOT=/usr/local/ptoas/0.61
export PYPTO_ROOT=$PP PYPTO_LIB_ROOT=$PL
export PTO_ISA_ROOT=$PP/runtime/build/pto-isa
export PYPTO_ALLOW_TORCH_NPU=2.10.0
export PYTHONPATH=$PL:$PL/models/deepseek_v4_flash_mtp:$PP/python:$PP/runtime:$PP/runtime/python
export HOME=$OUT/home TMPDIR=$OUT/tmp
export ASCEND_CACHE_PATH=$OUT/cache ASCEND_WORK_PATH=$OUT/work
export ASCEND_PROCESS_LOG_PATH=$OUT/ascend
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd $OUT/work || exit 1
gdb -batch -nx \
    -ex "set pagination off" -ex "set confirm off" \
    -ex "run" \
    -ex "echo \n===== BACKTRACE =====\n" -ex "bt 60" \
    -ex "echo \n===== THREADS =====\n" -ex "thread apply all bt 25" \
    -ex "echo \n===== REGS =====\n" -ex "info registers rip rsp" \
    --args $V/bin/python -X faulthandler $D/eager_path_check.py --out-dir $OUT
EOF
chmod +x "$OUT/inner.sh"
task-submit --device auto --max-time 2400 --timeout 3000 --run "bash $OUT/inner.sh" 2>&1 | tee "$OUT/task.log"
echo "[kmode] 产物: $OUT"
