#!/usr/bin/env bash
# 用已下载的官方分片拼一个「前 N 层」的可加载模型目录。
#
# 源是 ModelScope 的 Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp —— vllm-ascend v0.20.2rc1 官方
# nightly 配置指定的权重。它和 /data/models/dsv4-flash-w8a8 的量化配方不同：
#   wq_a/wq_b/wkv/indexer.wq_b = W8A8_DYNAMIC，scale 名为 weight_scale/weight_offset
#   wo_a/wo_b/compressor/weights_proj/norm/ape/attn_sink = FLOAT（无 scale）
# 本地那份把 wo_b 也量化了且用 .scale 命名，所以 loader 报
# KeyError: model.layers.0.self_attn.wo_b.scale。
#
# 整仓 279 GiB / 70 个分片，只下前 N 层需要的那几片即可：
#   layers 0-2 在分片 1-5，embed/norm/head 在分片 70，mtp 在 68-70。
#
# 用法: LAYERS=3 ./make_official_l3_dir.sh
set -uo pipefail
SRC="${SRC:-/data/sunkaixuan/skx_log_output/dsv4_vllm/models/official/DeepSeek-V4-Flash-w8a8-mtp}"
LAYERS="${LAYERS:-3}"
MTP="${MTP:-0}"
DST="${DST:-/data/sunkaixuan/skx_log_output/dsv4_vllm/models/official-l${LAYERS}}"

test -f "$SRC/quant_model_weights.safetensors.index.json" || {
    echo "缺少索引: $SRC/quant_model_weights.safetensors.index.json" >&2; exit 1; }

mkdir -p "$DST"
find "$DST" -maxdepth 1 -type l -delete
# quant_model_description.json 必须带上：官方 config.json 里没有 quantization_config，
# 量化信息全在这个文件里（modelslim 约定）。缺了它 --quantization ascend 会在
# ModelConfig 校验阶段就失败，并打印"Found JSON files in model directory"列表。
for f in tokenizer.json tokenizer_config.json generation_config.json configuration.json quant_model_description.json; do
    [ -f "$SRC/$f" ] && ln -sfn "$SRC/$f" "$DST/$f"
done

python3 - "$SRC" "$DST" "$LAYERS" "$MTP" <<'PYEOF'
import json, os, re, sys, struct

src, dst, layers, mtp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
idx = json.load(open(os.path.join(src, "quant_model_weights.safetensors.index.json")))["weight_map"]

# 分片是否已下全：safetensors 头部记录了数据区长度，据此判断文件完整
def complete(path):
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
        return os.path.getsize(path) == 8 + n + end
    except Exception:
        return False

present, partial = set(), set()
for f in sorted(set(idx.values())):
    p = os.path.join(src, f)
    if not os.path.exists(p):
        continue
    (present if complete(p) else partial).add(f)

def wanted(name):
    if name.startswith("mtp."):
        return mtp
    m = re.match(r"layers\.(\d+)\.", name)
    return int(m.group(1)) < layers if m else True

keep = {k: f for k, f in idx.items() if wanted(k)}
missing_files = sorted({f for f in keep.values()} - present)
keep = {k: f for k, f in keep.items() if f in present}

for f in present:
    os.path.exists(os.path.join(dst, f)) or os.symlink(os.path.join(src, f), os.path.join(dst, f))

json.dump({"metadata": {"total_size": 0}, "weight_map": keep},
          open(os.path.join(dst, "model.safetensors.index.json"), "w"))

cfg = json.load(open(os.path.join(src, "config.json")))
cfg["num_hidden_layers"] = layers
cfg["num_nextn_predict_layers"] = 1 if mtp else 0
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)

print(f"DST={dst}")
print(f"  层数 {layers}  MTP {'开' if mtp else '关'}")
print(f"  已完整分片 {len(present)} 个，写入索引的张量 {len(keep)}")
if partial:
    print(f"  ⚠ 下载未完成的分片（已跳过）: {sorted(partial)}")
if missing_files:
    print(f"  ⚠ 还缺这些分片，模型跑不起来: {missing_files}")
else:
    print("  ✓ 前 %d 层所需分片齐全" % layers)
PYEOF
