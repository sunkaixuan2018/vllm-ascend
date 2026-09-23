#!/usr/bin/env bash
# 补齐官方权重仓的小文件（tokenizer 等）。
# 加 --speed-time/--speed-limit：ModelScope 偶发"连接不断但也不传"的挂死，
# 只靠 --retry 救不回来（实测 tokenizer.json 卡在 463KB 一个半小时）。
set -uo pipefail
D=/data/sunkaixuan/skx_log_output/dsv4_vllm/models/official/DeepSeek-V4-Flash-w8a8-mtp
R=Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
for f in "$@"; do
    echo "[$(date +%H:%M:%S)] $f"
    curl -sSL --noproxy '*' --retry 10 --retry-delay 5 \
         --speed-time 60 --speed-limit 1024 \
         -o "$D/$f" \
         "https://www.modelscope.cn/api/v1/models/$R/repo?Revision=master&FilePath=$f" \
         -w "  %{size_download} 字节, %{speed_download} B/s\n"
done
