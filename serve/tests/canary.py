#!/usr/bin/env python3
"""Serve validation: 6 adoption canaries (exact answers) + 3 long-prompt cells.

Host-side script (stdlib only):  python3 canary.py --base http://127.0.0.1:8022 \
    --model qwen3.8-27b-exl3 --prompts canary_prompts.json
"""
import argparse, json, re, sys, urllib.request

CANARIES = [
    ("batt-arith",   "last_integer",          703),
    ("batt-capital", "last_line_equals",      "paris"),
    ("batt-token",   "contains",              "B70-FLASH-OK"),
    ("batt-sumsq",   "last_integer",          14),
    ("batt-logic",   "last_line_first_word",  "no"),
    ("batt-json",    "last_json_equals",      {"status": "ok", "count": 3}),
]
# expected rendered prompt_tokens = file ids + serve-template overhead (+44 measured on the plain
# 4.00bpw route, +51 on SC_5.00bpw_H6; informational only, not scored)
FILE_IDS = {"batt-arith": 22, "batt-capital": 20, "batt-token": 22,
            "batt-sumsq": 28, "batt-logic": 31, "batt-json": 26}

LONG_CELLS = [
    ("\n\nSeparately from the text above: what is 17 + 26? Reply with only the integer.",
     "last_integer", 43),
    ("\n\nNow, going back to the long text above: what is its very first word? Reply with only that word.",
     "last_line_first_word", "measurement"),
    ("\n\nSeparately from the text above: what is 100 - 58? Reply with only the integer.",
     "last_integer", 42),
]


def last_lines(text):
    return [l for l in text.strip().split("\n") if l.strip()]


def score(kind, value, text):
    lines = last_lines(text)
    if kind == "contains":
        return value in text
    if not lines:
        return False
    if kind == "last_integer":
        ints = re.findall(r"-?\d[\d,]*", text)
        return bool(ints) and int(ints[-1].replace(",", "")) == value
    if kind == "last_line_equals":
        return lines[-1].strip().lower() == str(value).lower()
    if kind == "last_line_first_word":
        return lines[-1].strip().split()[0].lower() == str(value).lower()
    if kind == "last_json_equals":
        for l in reversed(lines):
            try:
                return json.loads(l.strip()) == value
            except Exception:
                m = re.search(r"\{.*\}", l)
                if m:
                    try:
                        return json.loads(m.group(0)) == value
                    except Exception:
                        pass
        return False
    raise ValueError(kind)


def chat(base, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": max_tokens}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--gen", type=int, default=512)
    a = ap.parse_args()
    prompts = json.load(open(a.prompts))
    nfail = 0
    for name, kind, val in CANARIES:
        r = chat(a.base, a.model, prompts[name]["text"], a.gen)
        msg = r["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
        pt = (r.get("usage") or {}).get("prompt_tokens")
        ok = score(kind, val, text)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} expect={kind}:{val} "
              f"prompt_tokens={pt} (file {FILE_IDS[name]}+template) finish={r['choices'][0].get('finish_reason')} "
              f"out={text[-90:]!r}", flush=True)
        nfail += 0 if ok else 1
    p8 = prompts["p8192"]["text"]
    for i, (q, kind, val) in enumerate(LONG_CELLS):
        r = chat(a.base, a.model, p8 + q, a.gen)
        msg = r["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
        u = r.get("usage") or {}
        ok = score(kind, val, text)
        pt_ok = (u.get("prompt_tokens") or 0) >= 8000
        coherent = len(text.strip()) > 0 and r["choices"][0].get("finish_reason") in ("stop", "length")
        print(f"[{'PASS' if ok and pt_ok and coherent else 'FAIL'}] long-{i} expect={kind}:{val} "
              f"prompt_tokens={u.get('prompt_tokens')} completion={u.get('completion_tokens')} "
              f"finish={r['choices'][0].get('finish_reason')} out={text[-80:]!r}", flush=True)
        nfail += 0 if (ok and pt_ok and coherent) else 1
    print(f"canary/long summary: {nfail} failures")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
