#!/usr/bin/env bash
# 造一个与 CI 机一致的 DSV4 模型目录：权重全部符号链接到共享目录，只把 config.json
# 换成原版。
#
# 为什么需要这一步：/data/models/dsv4-flash-w8a8/config.json 是本机被改过的
# "yyd-recipes" 版（12443 字节，quantization_config.ignore 有 279 条），而 CI 机那份是
# 原版（2400 字节，ignore 为空，sha256 前缀 3f3df8b6）。带 ignore 的那份会让
# vllm-ascend 给注意力 linear 选到未量化方法，于是模型里没有 wo_b 的 scale 参数，
# 加载权重时报 KeyError: model.layers.0.self_attn.wo_b.scale。
# 原版就躺在同目录的 config.json.pre-yyd-recipes，哈希与 CI 的完全一致。
# 共享目录不能改（别人在用），所以在自己的目录里拼一个。
set -uo pipefail
SRC="${SRC:-/data/models/dsv4-flash-w8a8}"
DST="${DST:-/data/sunkaixuan/skx_log_output/dsv4_vllm/models/dsv4-flash-w8a8-ci}"
ORIG_CONFIG="${ORIG_CONFIG:-$SRC/config.json.pre-yyd-recipes}"

test -d "$SRC" || { echo "源目录不存在: $SRC" >&2; exit 1; }
test -f "$ORIG_CONFIG" || { echo "原版 config 不存在: $ORIG_CONFIG" >&2; exit 1; }

mkdir -p "$DST"
find "$DST" -maxdepth 1 -type l -delete
for f in "$SRC"/*; do
    b=$(basename "$f")
    [ "$b" = "config.json" ] && continue
    [ "$b" = "config.json.pre-yyd-recipes" ] && continue
    ln -sfn "$f" "$DST/$b"
done
cp -f "$ORIG_CONFIG" "$DST/config.json"

# LAYERS=N 只保留前 N 层：整模权重按 EP 均分后放不下小卡数，截层能在现有卡上跑通全链路。
# 必须写进 config.json，不能用 vllm 的 --hf-overrides —— 后者会让注意力 linear 拿不到
# 量化方法，加载时报 KeyError: model.layers.0.self_attn.wo_b.scale（同 checkpoint 不
# 截层可正常载入）。截层后 MTP 权重（checkpoint 第 43 层）索引对不上，配套跑时把
# SPEC_CONFIG 置空关掉投机解码。
if [ -n "${LAYERS:-}" ]; then
    python3 - "$DST/config.json" "$LAYERS" <<'PY'
import json, sys
p, n = sys.argv[1], int(sys.argv[2])
d = json.load(open(p))
d["num_hidden_layers"] = n
d["num_nextn_predict_layers"] = 0
json.dump(d, open(p, "w"), indent=2)
print("num_hidden_layers ->", n, "| mtp layers -> 0")
PY
fi

echo "DST=$DST"
sha256sum "$DST/config.json" | cut -c1-16
python3 -c "
import json
q = json.load(open(\"$DST/config.json\")).get(\"quantization_config\", {})
print(\"ignore entries:\", len(q.get(\"ignore\") or []))
"
ls "$DST" | wc -l
