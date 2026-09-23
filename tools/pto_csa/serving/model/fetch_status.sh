#!/usr/bin/env bash
# 官方权重下载进度
D=/data/sunkaixuan/skx_log_output/dsv4_vllm/models/official/DeepSeek-V4-Flash-w8a8-mtp
NEED="00001 00002 00003 00004 00005 00070"
tot=0; done_n=0
for n in $NEED; do
    f=$D/quant_model_weights-$n-of-00070.safetensors
    s=$(stat -c%s "$f" 2>/dev/null || echo 0); tot=$((tot+s))
    pct=$(awk -v s=$s "BEGIN{printf \"%.0f\", s/4283969536*100}")
    [ "$pct" -ge 99 ] && { done_n=$((done_n+1)); mark="完成"; } || mark="${pct}%"
    printf "  分片 %s  %6.2f GiB  %s\n" "$n" "$(awk -v s=$s "BEGIN{print s/1073741824}")" "$mark"
done
printf "合计 %.2f GiB / 约 23.9 GiB   完整分片 %d/6\n" "$(awk -v t=$tot "BEGIN{print t/1073741824}")" "$done_n"
echo "活动下载: $(pgrep -u $(id -u) -cf "FilePath=quant_model_weights") 个"
echo "tokenizer.json: $(stat -c%s $D/tokenizer.json 2>/dev/null || echo 0) 字节"
