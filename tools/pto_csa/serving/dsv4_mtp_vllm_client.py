#!/usr/bin/env python3
"""Streaming vLLM benchmark client for DeepSeek-V4 MTP serving."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty input")
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
        "std": std,
        "cv_pct": std / mean * 100.0 if mean else 0.0,
    }


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "dsv4-mtp-vllm-client"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def tokenize(base_url: str, model: str, prompt: str, timeout: float) -> list[int]:
    endpoint = base_url.rstrip("/") + "/tokenize"
    result = request_json(endpoint, {"model": model, "prompt": prompt}, timeout)
    token_ids = result.get("tokens") or result.get("token_ids")
    if not isinstance(token_ids, list):
        raise RuntimeError(f"unexpected tokenize response keys: {sorted(result)}")
    return [int(token_id) for token_id in token_ids]


def detokenize(base_url: str, model: str, token_ids: list[int], timeout: float) -> str:
    endpoint = base_url.rstrip("/") + "/detokenize"
    result = request_json(endpoint, {"model": model, "tokens": token_ids}, timeout)
    text = result.get("prompt") or result.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"unexpected detokenize response keys: {sorted(result)}")
    return text


def make_prompt(
    *,
    base_url: str,
    model: str,
    target_tokens: int,
    timeout: float,
) -> tuple[str, list[int]]:
    seed_text = (
        "DeepSeek V4 MTP performance prompt. "
        "We keep the content plain and deterministic for serving measurement. "
    )
    prompt = seed_text
    token_ids = tokenize(base_url, model, prompt, timeout)
    while len(token_ids) < target_tokens:
        multiplier = max(2, int(target_tokens / max(1, len(token_ids))) + 1)
        prompt = (prompt + "\n") * multiplier
        token_ids = tokenize(base_url, model, prompt, timeout)
    if len(token_ids) == target_tokens:
        return prompt, token_ids

    # vllm 0.20.2 的 /v1/completions 不接受 prompt_token_ids（只认 prompt / prompt_embeds），
    # 所以把截断后的 id 经 /detokenize 转回字符串再发。往返可能差一两个 token。
    exact_ids = token_ids[:target_tokens]
    text = detokenize(base_url, model, exact_ids, timeout)
    roundtrip = tokenize(base_url, model, text, timeout)
    if len(roundtrip) != target_tokens:
        print(
            f"[prompt] detokenize 往返后长度 {len(roundtrip)}，目标 {target_tokens}",
            flush=True,
        )
    return text, roundtrip


def stream_one(
    *,
    request_id: int,
    endpoint: str,
    payload: dict[str, Any],
    barrier: threading.Barrier,
    batch_epoch: float,
    timeout: float,
) -> dict[str, Any]:
    barrier.wait()
    started = time.perf_counter()
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "dsv4-mtp-vllm-client"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    token_ids: list[int] = []
    token_times: list[float] = []
    prompt_token_ids: list[int] | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    text_parts: list[str] = []
    multi_token_chunks = 0
    response_status: int | None = None
    error: str | None = None

    try:
        with opener.open(request, timeout=timeout) as response:
            response_status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if prompt_token_ids is None and choice.get("prompt_token_ids") is not None:
                    prompt_token_ids = [int(value) for value in choice["prompt_token_ids"]]
                delta_ids = [int(value) for value in choice.get("token_ids") or []]
                if len(delta_ids) > 1:
                    multi_token_chunks += 1
                now = time.perf_counter()
                token_ids.extend(delta_ids)
                token_times.extend([now - batch_epoch] * len(delta_ids))
                if choice.get("text"):
                    text_parts.append(choice["text"])
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        ended = time.perf_counter()
    except Exception as exc:  # noqa: BLE001
        ended = time.perf_counter()
        error = f"{type(exc).__name__}: {exc}"

    intervals_ms = [
        (token_times[index] - token_times[index - 1]) * 1000.0
        for index in range(1, len(token_times))
    ]
    return {
        "request_id": request_id,
        "http_status": response_status,
        "error": error,
        "start_offset_s": started - batch_epoch,
        "end_offset_s": ended - batch_epoch,
        "latency_s": ended - started,
        "ttft_ms": (token_times[0] - (started - batch_epoch)) * 1000.0 if token_times else None,
        "prompt_token_ids": prompt_token_ids,
        "prompt_tokens": len(prompt_token_ids) if prompt_token_ids is not None else None,
        "token_ids": token_ids,
        "token_times_s": token_times,
        "token_intervals_ms": intervals_ms,
        "completion_tokens": len(token_ids),
        "finish_reason": finish_reason,
        "usage": usage,
        "text": "".join(text_parts),
        "multi_token_chunks": multi_token_chunks,
        "token_fingerprint": sha256_json(token_ids) if token_ids else None,
    }


def run_batch(
    *,
    batch_index: int,
    batch_size: int,
    endpoint: str,
    payload: dict[str, Any],
    output_dir: Path,
    timeout: float,
    steady_skip: int,
) -> dict[str, Any]:
    batch_dir = output_dir / f"batch_{batch_index:03d}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    barrier = threading.Barrier(batch_size + 1)
    batch_epoch = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [
            executor.submit(
                stream_one,
                request_id=index,
                endpoint=endpoint,
                payload=payload,
                barrier=barrier,
                batch_epoch=batch_epoch,
                timeout=timeout,
            )
            for index in range(batch_size)
        ]
        barrier.wait()
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - batch_epoch
    results.sort(key=lambda item: item["request_id"])

    ok = [result for result in results if result["error"] is None]
    complete = [result for result in ok if result["completion_tokens"] == payload["max_tokens"]]
    aligned = len(complete) == batch_size
    batch_step_times_s: list[float] = []
    if aligned:
        for token_index in range(payload["max_tokens"]):
            batch_step_times_s.append(
                statistics.median(result["token_times_s"][token_index] for result in complete)
            )
    batch_intervals_ms = [
        (batch_step_times_s[index] - batch_step_times_s[index - 1]) * 1000.0
        for index in range(1, len(batch_step_times_s))
    ]
    steady_intervals_ms = batch_intervals_ms[steady_skip:]
    ttft_ms = [float(result["ttft_ms"]) for result in ok if result["ttft_ms"] is not None]
    latency_ms = [float(result["latency_s"]) * 1000.0 for result in ok]
    raw_stats = describe(batch_intervals_ms)
    steady_stats = describe(steady_intervals_ms)
    prompt_lengths = sorted({result["prompt_tokens"] for result in ok})
    output_fingerprints = sorted(
        {result["token_fingerprint"] for result in ok if result["token_fingerprint"]}
    )
    metrics = {
        "batch_index": batch_index,
        "batch_size": batch_size,
        "batch_wall_s": wall_s,
        "requests_succeeded": len(ok),
        "requests_completed_max_tokens": len(complete),
        "prompt_lengths": prompt_lengths,
        "finish_reasons": sorted({str(result["finish_reason"]) for result in ok}),
        "total_completion_tokens": sum(result["completion_tokens"] for result in ok),
        "overall_output_tok_s": sum(result["completion_tokens"] for result in ok) / wall_s
        if wall_s
        else 0.0,
        "raw_intervals_ms": raw_stats,
        "steady_intervals_ms": steady_stats,
        "raw_single_request_tok_s": 1000.0 / raw_stats["mean"] if raw_stats.get("mean") else None,
        "raw_batch_tok_s": batch_size * 1000.0 / raw_stats["mean"]
        if raw_stats.get("mean")
        else None,
        "steady_single_request_tok_s": 1000.0 / steady_stats["mean"]
        if steady_stats.get("mean")
        else None,
        "steady_batch_tok_s": batch_size * 1000.0 / steady_stats["mean"]
        if steady_stats.get("mean")
        else None,
        "ttft_ms": describe(ttft_ms),
        "request_latency_ms": describe(latency_ms),
        "output_token_fingerprints": output_fingerprints,
        "multi_token_chunks": sum(result["multi_token_chunks"] for result in ok),
        "execution_valid": len(ok) == batch_size and len(complete) == batch_size,
        "errors": [result["error"] for result in results if result["error"] is not None],
    }

    with (batch_dir / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (batch_dir / "step_times.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["interval_index", "interval_ms", "is_steady"])
        for index, value in enumerate(batch_intervals_ms):
            writer.writerow([index, f"{value:.9f}", int(index >= steady_skip)])
    with (batch_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8113")
    parser.add_argument("--model", default="dsv4")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=106)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measured-batches", type=int, default=1)
    parser.add_argument("--steady-skip", type=int, default=5)
    parser.add_argument("--pause-between-batches", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=1807)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    base_url = args.base_url.rstrip("/")
    prompt, prompt_ids = make_prompt(
        base_url=base_url,
        model=args.model,
        target_tokens=args.prompt_tokens,
        timeout=args.timeout,
    )
    prompt_payload: dict[str, Any]
    if prompt:
        prompt_payload = {"prompt": prompt}
    else:
        prompt_payload = {"prompt_token_ids": prompt_ids}
    payload: dict[str, Any] = {
        "model": args.model,
        **prompt_payload,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": args.seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "ignore_eos": True,
    }
    endpoint = base_url + "/v1/completions"
    with (output_dir / "prompt_info.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "prompt_tokens": len(prompt_ids),
                "prompt_token_fingerprint": sha256_json(prompt_ids),
                "used_prompt_token_ids": not bool(prompt),
                "request_payload_without_prompt": {
                    key: value
                    for key, value in payload.items()
                    if key not in {"prompt", "prompt_token_ids"}
                },
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )

    warmup_dir = output_dir / "warmup"
    measured_dir = output_dir / "measured"
    warmup_dir.mkdir()
    measured_dir.mkdir()
    for index in range(1, args.warmup_batches + 1):
        metrics = run_batch(
            batch_index=index,
            batch_size=args.batch_size,
            endpoint=endpoint,
            payload=payload,
            output_dir=warmup_dir,
            timeout=args.timeout,
            steady_skip=args.steady_skip,
        )
        print(
            f"warmup batch={index} valid={metrics['execution_valid']} "
            f"steady_ms={metrics['steady_intervals_ms'].get('mean')}",
            flush=True,
        )
        time.sleep(args.pause_between_batches)

    measured: list[dict[str, Any]] = []
    for index in range(1, args.measured_batches + 1):
        metrics = run_batch(
            batch_index=index,
            batch_size=args.batch_size,
            endpoint=endpoint,
            payload=payload,
            output_dir=measured_dir,
            timeout=args.timeout,
            steady_skip=args.steady_skip,
        )
        measured.append(metrics)
        print(
            f"measured batch={index} valid={metrics['execution_valid']} "
            f"steady_ms={metrics['steady_intervals_ms'].get('mean')} "
            f"steady_batch_tok_s={metrics['steady_batch_tok_s']}",
            flush=True,
        )
        if index < args.measured_batches:
            time.sleep(args.pause_between_batches)

    steady_means = [
        float(batch["steady_intervals_ms"]["mean"])
        for batch in measured
        if batch["steady_intervals_ms"].get("mean") is not None
    ]
    summary = {
        "base_url": base_url,
        "model": args.model,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "warmup_batches": args.warmup_batches,
        "measured_batches": args.measured_batches,
        "steady_skip": args.steady_skip,
        "seed": args.seed,
        "measured": measured,
        "all_measured_valid": all(batch["execution_valid"] for batch in measured),
        "steady_interval_ms_across_batches": describe(steady_means),
    }
    if steady_means:
        mean_ms = statistics.fmean(steady_means)
        summary["steady_single_request_tok_s"] = 1000.0 / mean_ms
        summary["steady_batch_tok_s"] = args.batch_size * 1000.0 / mean_ms
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_measured_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
