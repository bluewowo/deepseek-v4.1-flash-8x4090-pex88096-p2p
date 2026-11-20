#!/usr/bin/env python3
"""Matched cold workloads with a long prefill arriving during short decodes."""
import argparse
import json
import re
import runpy
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from threading import Barrier, Event
from uuid import uuid4

from tokenizers import Tokenizer

_benchmark = runpy.run_path(str(Path(__file__).with_name("benchmark-pex88096.py")))
CODES = _benchmark["CODES"]
HTTP = _benchmark["HTTP"]
counters = _benchmark["counters"]
prompt_for = _benchmark["prompt_for"]


def post(url, body):
    body.update(
        model="deepseek-v4.1-flash-1017",
        cache_salt=uuid4().hex,
        stream=True,
        chat_template_kwargs={"thinking": False},
    )
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return HTTP.open(request, timeout=240)


def stream(url, prompt, seed, budget, ready=None, start_gate=None):
    body = dict(
        messages=[dict(role="user", content=prompt)], seed=seed,
        temperature=0.7 if budget > 8 else 0, top_p=0.95,
        max_tokens=budget, ignore_eos=True, return_token_ids=True,
        stream_options={"include_usage": True},
    )
    if start_gate:
        start_gate.wait(timeout=40)
    start = time.monotonic()
    tokens = []
    events = []
    finish = usage = None
    with post(url, body) as response:
        for raw in response:
            if not raw.startswith(b"data:") or raw[5:].strip() == b"[DONE]":
                continue
            event = json.loads(raw[5:])
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", ()):
                finish = choice.get("finish_reason") or finish
                ids = choice.get("token_ids") or []
                if ids:
                    events.append((time.monotonic(), len(ids)))
                    tokens.extend(ids)
                    if ready and len(events) >= 2:
                        ready.set()
    if finish != "length" or len(tokens) != budget or usage is None:
        raise RuntimeError((finish, len(tokens), budget, usage))
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0):
        raise RuntimeError("Unexpected request prefix cache hit")
    return dict(start=start, events=events, input=usage["prompt_tokens"],
                output=len(tokens), _token_ids=tokens)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))] if ordered else None


def measure(url, tokenizer, concurrency, length, seed, late=False):
    short_prompts = [prompt_for(tokenizer, 8192, seed+i+1, False).replace(
        "35 distinct", "120 distinct") for i in range(concurrency-1)]
    if late and concurrency < 3:
        raise ValueError("Need an early decode plus a late short request")
    late_prompt = short_prompts.pop() if late else None
    long_prompt = prompt_for(tokenizer, length, seed, False)
    before = counters(url)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        if short_prompts:
            gate = Barrier(len(short_prompts))
            signals = [Event() for _ in short_prompts]
            jobs = [pool.submit(stream, url, prompt, seed+i+1, 768, signal, gate)
                    for i, (prompt, signal) in enumerate(zip(short_prompts, signals))]
            for signal in signals:
                if not signal.wait(timeout=45):
                    raise RuntimeError("Short request failed to enter decode")
        else:
            jobs = []
        long_job = pool.submit(stream, url, long_prompt, seed, 1)
        if late_prompt is not None:
            # The long prompt is locally encoded and reaches EngineCore before
            # the late short request is submitted on the same host.
            time.sleep(0.8)
            jobs.append(pool.submit(stream, url, late_prompt, seed+concurrency-1, 768))
        long = long_job.result()
        short = [job.result() for job in jobs]
    after = counters(url)
    long.pop("_token_ids")
    for request in short:
        answer = tokenizer.decode(request.pop("_token_ids"))
        request["codes_correct"] = re.findall(r"\b[0-9]{8}\b", answer)[:3] == list(CODES)
    delta = {key: after[key]-before.get(key, 0) for key in after}
    if (delta["vllm:prompt_tokens_cached_total"] != 0
        or delta["vllm:request_prefill_time_seconds_count"] != concurrency):
        raise RuntimeError(("Expected cold workload and exclusive bench", delta))
    begin = long["start"]
    first = long["events"][0][0]
    gaps = [
        end-start for request in short
        for (start, _), (end, _) in zip(request["events"], request["events"][1:])
        if begin < end and start < first
    ]
    if short and not gaps:
        raise RuntimeError("No short-decode overlap with the long request")
    return dict(
        concurrency=concurrency, length=length, seed=seed, input=long["input"],
        long_ttft_s=first-begin, long_input_tok_s=long["input"]/(first-begin),
        short_ttft_median_s=median(r["events"][0][0]-r["start"] for r in short)
        if short else None, short_during_long_gap_p95_s=percentile(gaps, 0.95),
        short_during_long_gap_p99_s=percentile(gaps, 0.99),
        short_during_long_gap_max_s=max(gaps, default=None),
        short_during_long_events=len(gaps), engine_metrics=delta,
        late_ttft_s=short[-1]["events"][0][0]-short[-1]["start"] if late else None,
        span_s=max(r["events"][-1][0] for r in [long, *short])
        - min(r["start"] for r in [long, *short]),
        short_codes=sum(r["codes_correct"] for r in short),
        short_requests=len(short),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Base URL without /v1, such as http://127.0.0.1:8011")
    parser.add_argument("tokenizer")
    parser.add_argument("cases", nargs="+", help="concurrency:length[:seed[:late]]")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    for case in args.cases:
        params = list(map(int, case.split(":")))
        concurrency, length = params[:2]
        seed = params[2] if len(params) >= 3 else 5301
        late = bool(params[3]) if len(params) >= 4 else False
        print(json.dumps(measure(args.url.rstrip("/"), tokenizer, concurrency, length, seed, late)),
              flush=True)


if __name__ == "__main__":
    main()
