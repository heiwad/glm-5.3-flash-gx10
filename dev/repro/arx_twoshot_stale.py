#!/usr/bin/env python3
"""Does a prefill step that goes through arx's two-shot all-reduce read stale data?

Standard library only. Point it at a running vLLM OpenAI server with arx's
two-shot path on (VLLM_ARX_TWO_SHOT_MAX_KB) and no other traffic:

    python arx_twoshot_stale.py --url http://127.0.0.1:8000 --model glm53

For each tail length (65, 89, 95 and 97 tokens by default) it builds a prompt
of --full-chunks whole prefill chunks plus that tail, so the request's last
prefill step is exactly the tail. Those steps are too big for arx's one-shot
path (32 tokens at hidden 4096) and not a multiple of 8 tokens, which is where
the two-shot scatter's blocks read parts of the send buffer that other blocks
copied. It sends the prompt RUNS times, one at a time, each under a fresh
cache_salt, with max_tokens 1 and prompt_logprobs 1, and compares every run's
prompt logprobs with the first.

--chunk must be the number of prompt tokens one step prefills for a lone
request. The default, 2304, is the entrypoint's LONG_PREFILL_TOKEN_THRESHOLD,
which caps each prefill; if you change that or MAX_NUM_BATCHED_TOKENS, pass the
smaller of the two. A wrong --chunk moves the final step off the tail length and
the script measures nothing.

A stale read on rank r corrupts rank r's chunk of the step's rows, and with
causal attention only rows from there on can change. At TP=2 that is the
second half of the final step, so differences that all sit there point at
the scatter; differences earlier in the prompt are something else.
"""
import argparse
import json
import os
import random
import string
import urllib.request


def post(base, path, payload, timeout=1800):
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def finished(base):
    """Requests the server has completed so far."""
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    return sum(float(line.split()[-1]) for line in text.splitlines()
               if line.startswith("vllm:request_success_total"))


def filler_tokens(base, model, n, seed=1):
    """n token ids of pseudo-words from a fixed seed: the same ids on any server with this tokenizer."""
    rng = random.Random(seed)
    ids = []
    while len(ids) < n:
        text = " ".join("".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 9)))
                        for _ in range(3000)) + ".\n"
        ids += post(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]
    return ids[:n]


def prompt_logprobs(base, model, ids):
    """The logprob of each prompt token after the first, and how many other requests finished meanwhile."""
    before = finished(base)
    out = post(base, "/v1/completions", {"model": model, "prompt": ids, "temperature": 0, "max_tokens": 1,
                                         "prompt_logprobs": 1, "cache_salt": os.urandom(8).hex()})
    others = int(finished(base) - before - 1)
    entries = out["choices"][0]["prompt_logprobs"]
    # Each entry holds the prompt token and the top-1, which may be the same one.
    return [(e.get(str(t)) or next(iter(e.values())))["logprob"] for t, e in zip(ids[1:], entries[1:])], others


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunk", type=int, default=2304, help="prompt tokens per prefill step (see above)")
    parser.add_argument("--full-chunks", type=int, default=2, help="whole chunks before the tail")
    parser.add_argument("--tails", type=lambda s: [int(x) for x in s.split(",")], default=[65, 89, 95, 97])
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--tp", type=int, default=2, help="tensor-parallel size: where rank 1's chunk starts")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    start = args.full_chunks * args.chunk  # first position of the final step
    ids = filler_tokens(base, args.model, start + max(args.tails))
    # The first long request after a boot can read differently; spend it here.
    prompt_logprobs(base, args.model, ids[:start // 2 + 1])
    print(f"{args.full_chunks} chunks of {args.chunk} tokens, then the tail; {args.runs} runs each")

    any_diff = False
    for tail in args.tails:
        prompt = ids[:start + tail]
        # Rank 1's chunk of the final step starts partway into this row, so
        # the row itself can already differ.
        second = start + tail // args.tp
        runs = []
        for i in range(args.runs):
            lps, others = prompt_logprobs(base, args.model, prompt)
            runs.append(lps)
            if others:
                print(f"  warning: {others} other requests finished during run {i}")
        differing, outside = 0, 0
        for i in range(1, len(runs)):
            # lps[k] is the logprob of prompt token k + 1.
            pos = [k + 1 for k, (a, b) in enumerate(zip(runs[0], runs[i])) if a != b]
            if not pos:
                continue
            differing += 1
            where = "final step, rank 1 onwards" if pos[0] >= second else (
                "final step, before rank 1's rows" if pos[0] >= start else "before the final step")
            outside += pos[0] < second
            print(f"  tail {tail}: run {i} vs 0: {len(pos)} of {len(runs[0])} differ, first at {pos[0]} "
                  f"({pos[0] - start:+d} from the final step's start): {where}")
        any_diff |= differing > 0
        verdict = ("all identical" if not differing else
                   "every difference in rank 1's rows onwards" if not outside else
                   f"{outside} run(s) differ before rank 1's rows")
        print(f"tail {tail:3d} ({start + tail} tokens): {differing} of {args.runs - 1} runs differ from run 0; "
              f"{verdict}")
    raise SystemExit(1 if any_diff else 0)


if __name__ == "__main__":
    main()
