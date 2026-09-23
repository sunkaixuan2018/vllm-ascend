#!/usr/bin/env python3
"""只把 **decode** 段录进 trace 的客户端。

直接 start_profile 再发请求，录到的绝大部分是 prefill：4096 个 prompt token 的一轮
prefill 比后面每一步 decode 都重得多，CSA 的 decode 路径会被埋掉。所以这里用流式请求，
**等第一个 token 吐出来**（说明 prefill 已经结束）再开 profiler，录够若干步再关。

用法:
    python profile_decode_client.py --base-url http://127.0.0.1:8113 \
        --profile-dir <dir> --batch-size 4 --prompt-tokens 4096 \
        --max-tokens 24 --profile-tokens 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

# 机器上的 proxy 变量会让 127.0.0.1 也走代理，这里显式关掉。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url: str, payload=None, timeout: float = 60.0):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with _OPENER.open(req, timeout=timeout) as r:
        return r.status, r.read()


def build_prompt(n_tokens: int, seed: int) -> str:
    """造一段长度大致可控的 prompt；词表固定，保证两次运行 prompt 一致。"""
    rng = random.Random(seed)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliet", "kilo", "lima", "mike", "november"]
    return " ".join(rng.choice(words) for _ in range(n_tokens))


class Stream(threading.Thread):
    """一条流式补全请求，记录收到的 token 数与首 token 时间。"""

    def __init__(self, base_url, model, prompt, max_tokens, seed, idx, first_token_evt):
        super().__init__(daemon=True)
        self.base_url, self.model, self.prompt = base_url, model, prompt
        self.max_tokens, self.seed, self.idx = max_tokens, seed, idx
        self.first_token_evt = first_token_evt
        self.tokens = 0
        self.error = None
        self.ttft = None

    def run(self):
        payload = {"model": self.model, "prompt": self.prompt,
                   "max_tokens": self.max_tokens, "temperature": 0.0,
                   "seed": self.seed, "stream": True}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/completions", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        try:
            with _OPENER.open(req, timeout=1800) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    chunk = json.loads(body)
                    if not chunk.get("choices"):
                        continue
                    if chunk["choices"][0].get("text"):
                        self.tokens += 1
                        if self.tokens == 1:
                            self.ttft = time.perf_counter() - t0
                            self.first_token_evt.set()
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"


def wait_for_trace(profile_dir: Path, timeout: float):
    """torch_npu 在 stop 之后异步落盘，这里等 trace_view.json 出现并停止增长。"""
    deadline = time.time() + timeout
    last = (None, -1)
    while time.time() < deadline:
        hits = sorted(profile_dir.rglob("trace_view.json"))
        if hits:
            size = hits[-1].stat().st_size
            if size > 0 and last == (hits[-1], size):
                return hits
            last = (hits[-1], size)
        time.sleep(3)
    return sorted(profile_dir.rglob("trace_view.json"))


def main() -> int:
    p = argparse.ArgumentParser(description="只录 decode 段的 profiling 客户端")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="dsv4")
    p.add_argument("--profile-dir", required=True)
    p.add_argument("--out", default="", help="结果 json")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--prompt-tokens", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--profile-tokens", type=int, default=8,
                   help="开 profiler 之后再录多少个 decode token")
    p.add_argument("--warmup", type=int, default=1, help="先跑几轮不采集的请求")
    p.add_argument("--seed", type=int, default=1807)
    p.add_argument("--trace-timeout", type=float, default=900.0)
    args = p.parse_args()

    prof_dir = Path(args.profile_dir)
    prof_dir.mkdir(parents=True, exist_ok=True)
    r: dict = {"base_url": args.base_url, "batch_size": args.batch_size,
               "prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
               "profile_tokens": args.profile_tokens, "stage": "warmup"}

    def save():
        text = json.dumps(r, indent=2, ensure_ascii=False)
        if args.out:
            Path(args.out).write_text(text)
        print(text, flush=True)

    prompt = build_prompt(args.prompt_tokens, args.seed)

    try:
        for w in range(args.warmup):
            evt = threading.Event()
            s = Stream(args.base_url, args.model, prompt, 4, args.seed, 0, evt)
            s.start()
            s.join(timeout=600)
            if s.error:
                raise RuntimeError(f"warmup {w} 失败: {s.error}")
        r["stage"] = "launch"
        save()

        evt = threading.Event()
        streams = [Stream(args.base_url, args.model, prompt, args.max_tokens,
                          args.seed + i, i, evt) for i in range(args.batch_size)]
        for s in streams:
            s.start()

        # 第一个 token 到手 = prefill 结束，从这里开始才是 decode
        if not evt.wait(timeout=600):
            raise RuntimeError("等不到第一个 token，prefill 没完成")
        r["stage"] = "start_profile"
        save()
        base = max(s.tokens for s in streams)
        post(f"{args.base_url}/start_profile")
        t0 = time.perf_counter()

        # 录够 profile_tokens 步就停；请求本身继续跑完
        deadline = t0 + 300
        while time.perf_counter() < deadline:
            if max(s.tokens for s in streams) - base >= args.profile_tokens:
                break
            if all(not s.is_alive() for s in streams):
                break
            time.sleep(0.02)
        captured = max(s.tokens for s in streams) - base
        post(f"{args.base_url}/stop_profile", timeout=600)
        r["profiled"] = {"decode_tokens_captured": captured,
                         "wall_s": round(time.perf_counter() - t0, 3)}
        r["stage"] = "drain"
        save()

        for s in streams:
            s.join(timeout=900)
        r["requests"] = [{"idx": s.idx, "tokens": s.tokens,
                          "ttft_s": round(s.ttft, 3) if s.ttft else None,
                          "error": s.error} for s in streams]
        errs = [s.error for s in streams if s.error]
        if errs:
            raise RuntimeError(f"请求失败: {errs[:2]}")

        r["stage"] = "wait_trace"
        save()
        traces = wait_for_trace(prof_dir, args.trace_timeout)
        r["traces"] = [{"path": str(t), "bytes": t.stat().st_size} for t in traces]
        r["ok"] = bool(traces)
        r["stage"] = "complete"
    except Exception as e:  # noqa: BLE001
        r["ok"] = False
        r["error"] = f"{type(e).__name__}: {e}"
    finally:
        save()
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
