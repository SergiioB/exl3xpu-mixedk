#!/usr/bin/env python3
"""check_tensor_bitexact.py — per-tensor ESIMD-vs-torch bit-exactness check.

(a) quantize(W_transformed, U_upper, 'exl3_mul1', 4, seed,
    backend='esimd') == backend='torch' BIT-EXACT (packed + W_hat) on the
    2 real tensors, harness U convention (U^T U = G^-1, G = rotated damped
    Hessian built exactly as quantize_linear does).
(b) seconds per tensor incl. U-conversion + LDLQ + pack/decode.

torch backend runs on XPU (same device as esimd run inputs) so the LDLQ
feedback state (W_t/hatW_t/prod, all torch ops on Wq.device) is identical
between backends — only the Viterbi itself differs.

Env:
  XTQ_BF16       Qwen3.8-27B BF16 checkpoint dir (model.safetensors.index.json)
  XTQ_HDIR       dir with L2_linear_attn.in_proj_qkv.safetensors and
                 L2_mlp.down_proj.safetensors, each holding key "H": the
                 undamped input Gram [in, in] of that layer-2 linear
  XTQ_VIT_DEVIDX Level Zero index of the card running the ESIMD worker
  XTQ_MARGIN     pass-1 margin (default 16; 0 = full pass)

Usage: XTQ_VIT_DEVIDX=0 python3 check_tensor_bitexact.py [tensor_filter]
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantize as xq  # noqa: E402

BF16 = os.environ["XTQ_BF16"]
HDIR = os.environ["XTQ_HDIR"]
PREFIX = "model.language_model."
SEED = 20261011
SU_SEED = 20261011
PERCDAMP = 0.01
XDEV = torch.device("xpu:0")          # torch side: first visible XPU


def build_inputs(W, H):
    """Reproduce quantize_linear's transform: W_transformed [out,in] fp32
    and harness-convention U (U^T U = G^-1, fp32) of the rotated damped G."""
    k_in, m_out = W.shape[1], W.shape[0]
    dev = torch.device("cpu")
    suh = xq.sign_vec(k_in, SU_SEED, dev).half()
    svh = xq.sign_vec(m_out, SU_SEED + 1, dev).half()
    h = xq.hadamard128(dev, torch.float32)
    W_i = xq.to_inner(W.float().t().contiguous(), suh, svh, h)   # [k,n]
    G = xq.rotate_hessian(H.float(), suh, h)
    dead = G.diagonal() == 0
    if dead.any():
        G[dead, dead] = 1.0
    G.diagonal().add_(PERCDAMP * G.diagonal().mean())
    t0 = time.time()
    Gi = torch.cholesky_inverse(torch.linalg.cholesky(G.double()))  # fp64
    U = torch.linalg.cholesky(Gi, upper=True)  # U^T U = G^{-1}
    err = (U.t() @ U - Gi).abs().max().item()
    print(f"    U built in {time.time()-t0:.0f}s, |U^TU-G^-1|max={err:.2e}",
          flush=True)
    del Gi
    return W_i.t().contiguous().float(), U.float()


def main():
    torch.set_num_threads(int(os.environ.get("XTQ_THREADS", "8")))
    wm_b = json.loads(Path(BF16, "model.safetensors.index.json")
                      .read_text())["weight_map"]
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = [("linear_attn.in_proj_qkv", "L2_linear_attn.in_proj_qkv.safetensors"),
            ("mlp.down_proj", "L2_mlp.down_proj.safetensors")]
    if which != "all":
        todo = [t for t in todo if which in t[0]]

    for base, hfile in todo:
        key = f"{PREFIX}layers.2.{base}"
        with safe_open(Path(BF16, wm_b[f"{key}.weight"]), framework="pt") as f:
            W = f.get_tensor(f"{key}.weight").float()
        with safe_open(Path(HDIR, hfile), framework="pt") as f:
            H = f.get_tensor("H").float()
        print(f"[{base}] {list(W.shape)}", flush=True)
        W_t, U = build_inputs(W, H)
        del H
        nw = W_t.numel()

        # (a) dual run: identical X to both Viterbi impls per LDLQ launch;
        #     per-launch state equality => whole pipeline bit-exact.
        xq._DUAL_STATS.clear()
        xq._DUAL_STATS.update(launches=0, mismatch_states=0)
        t0 = time.time()
        a = xq.quantize(W_t.to(XDEV), U, "exl3_mul1", 4, SEED,
                        backend="dual", margin=int(os.environ.get("XTQ_MARGIN","16")))
        ta = time.time() - t0
        ds = dict(xq._DUAL_STATS)
        print(f"    dual (torch-xpu + esimd per launch): {ta:.0f}s "
              f"({ta/nw*1e6:.2f} s/1M) launches={ds['launches']} "
              f"mismatch_states={ds['mismatch_states']} "
              f"first={ds.get('first')}", flush=True)

        # (b) esimd end-to-end on the same xpu inputs: timing + packed eq
        t0 = time.time()
        b = xq.quantize(W_t.to(XDEV), U, "exl3_mul1", 4, SEED,
                        backend="esimd", margin=int(os.environ.get("XTQ_MARGIN","16")))
        tb = time.time() - t0
        print(f"    esimd:      {tb:.0f}s ({tb/nw*1e6:.2f} s/1M)", flush=True)

        eq_p = torch.equal(a[1].cpu(), b[1].cpu())
        eq_w = torch.equal(a[0].cpu(), b[0].cpu())
        print(f"    BIT-EXACT packed={eq_p} W_hat={eq_w} "
              f"(per-launch mismatches={ds['mismatch_states']})", flush=True)
        print(json.dumps({"tensor": base, "nw": nw, "dual_s": round(ta, 1),
                          "esimd_s": round(tb, 1),
                          "esimd_s_per_1m": round(tb / nw * 1e6, 3),
                          "launches": ds["launches"],
                          "launch_mismatch_states": ds["mismatch_states"],
                          "bitexact": bool(eq_p and eq_w)}), flush=True)
        del W_t, U, a, b, W
        torch.xpu.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
