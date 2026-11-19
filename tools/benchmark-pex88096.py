#!/usr/bin/env python3
"""Measure isolated streaming requests against the DeepSeek 1017 API."""
import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer


HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CODES = ("78140635", "42063971", "63951728")
METRICS = (
    "vllm:prompt_tokens_cached_total",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def counters(url):
    result = {}
    with HTTP.open(url + "/metrics", timeout=10) as response:
        for line in response.read().decode().splitlines():
            match = re.match(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})? (\S+)", line)
            if match and match[1] in METRICS:
                result[match[1]] = result.get(match[1], 0) + float(match[2])
    return result


def prompt_for(tokenizer, length, seed, retrieval):
    rng = random.Random(seed * 1000003 + length)
    records = [[] for _ in range(4)]
    for row in range(length // 12 + 100):
        records[row % 4].append(
            f"Log {row + 101}: rack {rng.randrange(150, 999)} held "
            f"{rng.randrange(20, 800)} units. An auditor checked the tag.\n"
        )
    sections = tokenizer.encode_batch(["".join(group) for group in records], add_special_tokens=False)
    pieces = [f"Independent audit fixture {rng.getrandbits(64):016x}. Read the records.\n"]
    for i, section in enumerate(sections):
        pieces.append(tokenizer.decode(section.ids[:max(0, (length - 130) // 4)]))
        if i < len(CODES):
            pieces.append(f"\nThe exact authorization number for Project {i + 1} is {CODES[i]}.\n")
    pieces.append("\nGive the authorization numbers for Projects 1, 2 and 3 in that order.")
    if retrieval:
        pieces.append(" Print only the three numbers, in that order.")
    else:
        pieces.append(
            " Put the numbers first. Then write 35 distinct one-sentence steps for auditing this "
            "warehouse. Use a different concrete action and example in each step."
        )
    return "".join(pieces)


def measure(args, tokenizer, length, seed, retrieval):
    prompt = prompt_for(tokenizer, length, seed, retrieval)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": 64 if retrieval else args.output,
        "stream": True,
        "ignore_eos": not retrieval,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    if args.cache_salt:
        body["cache_salt"] = args.cache_salt
    before = counters(args.url)
    request = urllib.request.Request(
        args.url + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    started_unix = time.time()
    started = time.perf_counter()
    first = last = None
    usage = finish = None
    content = []
    with HTTP.open(request, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b"data:") or raw[5:].strip() == b"[DONE]":
                continue
            event = json.loads(raw[5:])
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", ()):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                fragment = delta.get("content") or ""
                if fragment or delta.get("reasoning_content"):
                    last = time.perf_counter()
                    first = last if first is None else first
                content.append(fragment)
    if not usage or first is None or finish not in {"stop", "length"}:
        raise RuntimeError(f"Incomplete stream: usage={usage}, finish={finish}")
    after = counters(args.url)
    diff = {metric: after[metric] - before.get(metric, 0.0) for metric in after}
    answer = "".join(content)
    terms = re.findall(r"\w+|[^\w\s]", answer.lower())
    four_grams = list(zip(terms, terms[1:], terms[2:], terms[3:]))
    repeated_4gram_fraction = (
        1.0 - len(set(four_grams)) / len(four_grams) if four_grams else None
    )
    answers_dir = getattr(args, "answers_dir", None)
    if answers_dir is not None:
        answers_dir.mkdir(parents=True, exist_ok=True)
        (answers_dir / f"{length}-{seed}.txt").write_text(answer, encoding="utf-8")
    ttft = first - started
    engine_prefill = diff.get("vllm:request_prefill_time_seconds_sum")
    engine_decode = diff.get("vllm:request_decode_time_seconds_sum")
    decode_time = last - first
    cache = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cache is None and "vllm:prompt_tokens_cached_total" in diff:
        cache = int(diff["vllm:prompt_tokens_cached_total"])
    rounds = diff.get("vllm:spec_decode_num_drafts_total")
    accepted = diff.get("vllm:spec_decode_num_accepted_tokens_total")
    return {
        "target": length, "seed": seed, "retrieval": retrieval, "start_unix": started_unix,
        "prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
        "cached_tokens": cache, "ttft_s": ttft, "decode_s": decode_time,
        "input_over_ttft": usage["prompt_tokens"] / ttft,
        "engine_prefill_s": engine_prefill if diff.get("vllm:request_prefill_time_seconds_count") == 1 else None,
        "engine_decode_s": engine_decode if diff.get("vllm:request_decode_time_seconds_count") == 1 else None,
        "decode_tok_s": (usage["completion_tokens"] - 1) / decode_time if decode_time and not retrieval else None,
        "rounds": rounds, "proposed": diff.get("vllm:spec_decode_num_draft_tokens_total"), "accepted": accepted,
        "sha": hashlib.sha256(answer.encode()).hexdigest()[:16],
        "repeated_4gram_fraction": repeated_4gram_fraction,
        "codes_correct": re.findall(r"\b\d{8}\b", answer)[:3] == list(CODES),
        "finish": finish,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("length", type=int, nargs="+")
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "DeepSeek-V4.1-Flash-1017")
    parser.add_argument("--output", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1017)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache-salt", default=None, help="Unique per cold run")
    parser.add_argument("--answers-dir", type=Path)
    parser.add_argument("--require-cold", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    correct = True
    for length in args.length:
        for offset in range(args.repeats):
            result = measure(args, tokenizer, length, args.seed + offset, args.retrieve)
            print(json.dumps(result), flush=True)
            if args.require_cold:
                engine_time = result["engine_prefill_s"]
                if result["cached_tokens"] != 0 or engine_time is None or engine_time <= 0:
                    raise RuntimeError("Expected zero cached tokens and one positive engine prefill sample")
            correct &= result["codes_correct"] if args.retrieve else result["output_tokens"] == args.output
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
