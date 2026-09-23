#!/usr/bin/env python3
"""从 pypto-lib 上游重新同步内联的 kernel 源码。

本包各子包是 pypto-lib 对应模型目录的内联副本，与上游的唯一差异是 import 形式：
上游用裸顶层 import（靠 PYTHONPATH 解析），这里是包内相对 import。本脚本重放同一套
改写，所以升级上游不需要手工改 import。

    python -m vllm_ascend.attention.pto_kernels.resync /path/to/pypto-lib
    python -m vllm_ascend.attention.pto_kernels.resync /path/to/pypto-lib --variant dspark

改完务必看 `git diff`：上游新增本地模块时闭包会自动带上；若上游把某个 `from golden
import ...` 提到了模块级，这里会变成一个包外依赖，届时要么一起内联 golden，要么
把那处调用挪回函数内。
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _variants() -> dict[str, tuple[str, str]]:
    from vllm_ascend.attention.pto_kernels import VARIANTS

    return VARIANTS


def closure(src: Path, local: set[str], entry: str) -> set[str]:
    """entry 可达的本地模块集合。认不出的名字留在集合外，当外部依赖处理。"""
    seen: set[str] = set()
    stack = [entry]
    while stack:
        mod = stack.pop()
        if mod in seen or mod not in local:
            continue
        seen.add(mod)
        tree = ast.parse((src / f"{mod}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                stack += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                stack.append(node.module.split(".")[0])
    return seen


def sync_one(lib_root: Path, name: str, subdir: str, entries, dry_run: bool) -> bool:
    src = lib_root / subdir
    if not src.is_dir():
        print(f"  {name}: 找不到 {src}", file=sys.stderr)
        return False

    local = {p.stem for p in src.glob("*.py")}
    mods = set().union(*(closure(src, local, e) for e in entries))
    names = "|".join(sorted(local))
    pat_from = re.compile(rf"^(\s*)from ({names}) import ", re.M)
    pat_import = re.compile(rf"^(\s*)import ({names})$", re.M)

    dst = HERE / name
    stale = {p.stem for p in dst.glob("*.py")} - mods - {"__init__"}
    if stale:
        print(f"  {name}: 上游已不再需要，请手工删除 {sorted(stale)}")

    for mod in sorted(mods):
        text = (src / f"{mod}.py").read_text(encoding="utf-8")
        text = pat_from.sub(r"\1from .\2 import ", text)
        text = pat_import.sub(r"\1from . import \2", text)
        if not dry_run:
            (dst / f"{mod}.py").write_text(text, encoding="utf-8", newline="\n")

    print(f"  {name}: {len(mods)} 个模块")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="从 pypto-lib 重新同步内联的 PTO kernel 源码")
    ap.add_argument("pypto_lib", type=Path, help="pypto-lib 仓的根目录")
    ap.add_argument("--variant", action="append", help="只同步这个子包，可重复；缺省全同步")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    variants = _variants()
    wanted = args.variant or sorted(variants)
    unknown = [v for v in wanted if v not in variants]
    if unknown:
        print(f"未知子包 {unknown}，已有 {sorted(variants)}", file=sys.stderr)
        return 1

    commit = subprocess.check_output(
        ["git", "-C", str(args.pypto_lib), "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"上游 {args.pypto_lib} @ {commit}")

    ok = all(sync_one(args.pypto_lib, v, *variants[v], args.dry_run) for v in wanted)
    if not ok:
        return 1

    if not args.dry_run and set(wanted) == set(variants):
        init = HERE / "__init__.py"
        text = init.read_text(encoding="utf-8")
        text = re.sub(r"^上游 commit：.*$", f"上游 commit：{commit}", text, flags=re.M)
        text = re.sub(
            r'^PYPTO_LIB_COMMIT = ".*"$', f'PYPTO_LIB_COMMIT = "{commit}"', text, flags=re.M
        )
        init.write_text(text, encoding="utf-8", newline="\n")
    elif not args.dry_run:
        print("只同步了部分子包，PYPTO_LIB_COMMIT 未更新（它记的是整体版本）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
