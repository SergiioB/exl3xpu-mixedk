#!/usr/bin/env python3
"""Multi-turn prefix-cache-on corruption check.

Builds a shared long context, then runs a 3-turn conversation where turn 2+
reuse the prefix cache (enable_prefix_caching=true + mamba align mode on the
hybrid model = the new V2 runner/draft state path). Checks:
  - turn2/turn3 report cached prompt tokens (prefix cache hit) or at minimum
    reuse the same prompt; answers must be coherent and correct.
  - Facts planted in the shared context must survive across turns.
"""
import argparse
import json
import sys
import urllib.request


def post(base, payload, timeout=600):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    # Shared context: a fact sheet ~1500 tokens via repetition of a seed doc,
    # plus three planted facts tested in later turns.
    seed = (
        "Project HALIBUT field notes, entry {i}: the calibration beacon at site "
        "Kestrel-9 broadcasts on 1420.405 MHz with callsign VX-92-TANGO. "
        "Chief engineer Dr. Mira Okonkwo logged drift of 0.003 arcsec/hour. "
        "Backup protocol: if the beacon fails twice, switch to relay PERIDOT-7 "
        "and notify station BRAVO within 40 minutes.\n"
    )
    context = "".join(seed.format(i=i) for i in range(160))
    facts = [
        ("turn2", "What frequency does the Kestrel-9 beacon broadcast on, and what is its callsign?",
         ["1420.405", "VX-92-TANGO"]),
        ("turn3", "If the beacon fails twice, which relay is used and who is notified within how many minutes?",
         ["PERIDOT-7", "BRAVO", "40"]),
        ("turn4", "Who is the chief engineer and what drift did they log?",
         ["Okonkwo", "0.003"]),
    ]

    msgs = [{"role": "user", "content": context + "\nAcknowledge with 'LOGGED' only."}]
    r = post(a.base, {
        "model": a.model, "messages": msgs, "max_tokens": 8,
        "temperature": 0.0, "stream": False,
    })
    ack = r["choices"][0]["message"]["content"].strip()
    u0 = r.get("usage", {})
    print(f"[setup] ack={ack!r} prompt_tokens={u0.get('prompt_tokens')}")
    msgs.append({"role": "assistant", "content": ack})

    fails = []
    for name, q, needles in facts:
        msgs.append({"role": "user", "content": q})
        r = post(a.base, {
            "model": a.model, "messages": msgs, "max_tokens": 96,
            "temperature": 0.0, "stream": False,
        })
        ans = r["choices"][0]["message"]["content"]
        u = r.get("usage", {})
        ct = u.get("prompt_tokens_details", {}) or {}
        cached = ct.get("cached_tokens", -1)
        ok = all(n.lower() in ans.lower() for n in needles)
        print(f"[{name}] prompt={u.get('prompt_tokens')} cached={cached} "
              f"completion={u.get('completion_tokens')} ok={ok}")
        print(f"  ans: {ans[:200]!r}")
        if not ok:
            fails.append((name, [n for n in needles if n.lower() not in ans.lower()]))
        msgs.append({"role": "assistant", "content": ans})

    if fails:
        print("MULTITURN FAIL:", fails)
        sys.exit(1)
    print("MULTITURN PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
