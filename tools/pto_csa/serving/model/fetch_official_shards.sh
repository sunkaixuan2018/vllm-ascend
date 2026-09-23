#!/usr/bin/env bash
# 从 ModelScope 拉 Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp 的指定分片（可断点续传）。
#
# 这是 vllm-ascend v0.20.2rc1 官方 nightly 配置指向的权重
# （tests/e2e/nightly/single_node/models/configs/DeepSeek-V4-Flash-W8A8-A3.yaml）。
# 它和 /data/models/dsv4-flash-w8a8 的量化配方不同：wo_a/wo_b 是 FLOAT 不量化，
# wq_a/wq_b/wkv/indexer.wq_b 是 W8A8_DYNAMIC 且 scale 命名为 weight_scale/weight_offset
# —— 正是本 release 的 loader 期望的形状。本地那份把 wo_b 也量化了并用 .scale 命名，
# 所以加载时报 KeyError: model.layers.0.self_attn.wo_b.scale。
#
# 用法: ./fetch_official_shards.sh 4 5        # 拉第 4、5 个分片
#       SHARDS_ALL=1 ./fetch_official_shards.sh   # 拉能跑前三层的全部 8 个
set -uo pipefail
REPO="${REPO:-Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp}"
DST="${DST:-/data/sunkaixuan/skx_log_output/dsv4_vllm/models/official/DeepSeek-V4-Flash-w8a8-mtp}"
mkdir -p "$DST"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [ "${SHARDS_ALL:-0}" = "1" ]; then
    set -- 1 2 3 4 5 68 69 70
fi

for n in "$@"; do
    # 10# 强制十进制：printf %05d 遇到 00070 会按八进制解析成 56，下到错的分片
    f=$(printf "quant_model_weights-%05d-of-00070.safetensors" "$((10#$n))")
    url="https://www.modelscope.cn/api/v1/models/$REPO/repo?Revision=master&FilePath=$f"
    echo "[$(date +%H:%M:%S)] 下载 $f"
    curl -sSL --noproxy "*" -C - --retry 5 --retry-delay 5 -o "$DST/$f" "$url" \
        -w "  完成 %{size_download} 字节, 平均 %{speed_download} B/s, 用时 %{time_total}s\n"
done
echo "=== 当前已下载 ==="
ls -la "$DST"/*.safetensors 2>/dev/null | awk "{printf \"  %-52s %.2f GiB\n\", \$9, \$5/1073741824}"
echo FETCH_DONE
