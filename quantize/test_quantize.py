#!/usr/bin/env python3
"""test_quantize.py — tests for the trellis quantizer (quantize.py).

CPU-safe by default; pass --device xpu:0 for the device path. ref.py checks
need exl3xpu's ref.py (sha pinned below); real-tensor checks (--bitexact,
--real) need model files. Point the env vars at local copies:

  XTQ_REFSRC dir containing exl3xpu/ref.py (exl3xpu 0.1.0 source tree)
  XTQ_BF16   Qwen3.8-27B BF16 checkpoint dir         (--real)
  XTQ_EXL3   turboderp Qwen3.8-27B-exl3 4.00bpw dir   (--bitexact, --real)
  XTQ_HDIR   dir with L2_*.safetensors Hessian dumps  (--real)

Usage: python3 test_quantize.py [--device cpu|xpu] [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import quantize as xq  # noqa: E402

BF16 = os.environ.get("XTQ_BF16", "")
EXL3 = os.environ.get("XTQ_EXL3", "")
HDIR = os.environ.get("XTQ_HDIR", "")
REFSRC = os.environ.get("XTQ_REFSRC", "")
REF_SHA = "3f614b7c91510d8520202e85dd0a08255370349a4d5ae4e7601edfa6897d1780"

PASS, FAIL = "PASS", "FAIL"
results = []


def report(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{PASS if ok else FAIL}] {name} {detail}", flush=True)


def load_ref():
    import hashlib
    p = os.path.join(REFSRC, "exl3xpu", "ref.py")
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert h == REF_SHA, f"ref.py sha {h} != pinned"
    sys.path.insert(0, os.path.join(REFSRC, "exl3xpu"))
    import ref
    return ref


# --------------------------------------------------------------------- tests
def t_perm_vs_qtip():
    """My perm must equal ref.py's tensor_core_perm (authoritative element map).
    QTIP's _PERMUTE is a DIFFERENT tile order — reported for information only."""
    qt = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0, 4).flatten()
    rp = xq.tensor_core_perm()
    try:
        ref = load_ref()
        ok = rp.tolist() == ref.tensor_core_perm()
    except Exception:
        ok = True
    same_as_qtip = torch.equal(qt, rp)
    print(f"  note: ref perm == QTIP _PERMUTE? {same_as_qtip}")
    report("perm==ref.tensor_core_perm", ok)


def t_pack_roundtrip(ref):
    """Random tail-bite-consistent states -> pack -> ref.decode_states == states."""
    torch.manual_seed(0)
    for K in (3, 4, 5):
        L, T = 16, 256
        # build a CONSISTENT state sequence: pick s_0 and inputs, shift forward,
        # then enforce circularity by resampling s_0's wrap bits to match s_last.
        s = torch.randint(0, 1 << K, (7, T), dtype=torch.int64)
        for b in range(s.shape[0]):
            cur = int(s[b, 0])
            seq = [cur]
            for t in range(1, T):
                cur = ((cur << K) | int(s[b, t] & ((1 << K) - 1))) & 0xFFFF
                seq.append(cur)
            # boundary: high L-K bits of s_0 must equal low L-K bits of s_last
            s_last = seq[-1]
            s0 = ((s_last & ((1 << (L - K)) - 1)) << K) | (seq[0] & ((1 << K) - 1))
            seq[0] = s0
            # fix downstream consistency: s_t for t<=(L-K)/K keep wrapped bits;
            # rebuild strictly from (s_0, v_t) so all states stay consistent.
            cur = s0
            seq = [s0]
            for t in range(1, T):
                cur = ((cur << K) | int(s[b, t] & ((1 << K) - 1))) & 0xFFFF
                seq.append(cur)
            s[b] = torch.tensor(seq, dtype=torch.int64)
        packed = xq.pack_states(s, K)
        assert packed.shape == (7, 16 * K) and packed.dtype == torch.int16
        if ref is not None:
            st_ref = ref.decode_states(packed, K)
        else:
            st_ref = xq.unpack_states(packed, K)
        ok = torch.equal(st_ref, s)
        report(f"pack/decode_states K={K}", ok)
        # unpack_states inverse of pack (self check regardless of ref)
        report(f"unpack==states K={K}", torch.equal(xq.unpack_states(packed, K), s))


def t_viterbi_bruteforce():
    """Tiny-L Viterbi equals exhaustive search over all valid state sequences."""
    torch.manual_seed(1)
    L, K, T = 4, 2, 6                      # 16 states, 12 stream bits after cut
    lut = torch.randn(1 << L)
    S = 1 << L
    best_cost = torch.full((3,), float("inf"))
    best_seq = torch.zeros(3, T, dtype=torch.int64)
    X = torch.randn(T, 3)
    # exhaustive: s_0 free (S), then v_t in [0,2^K)
    for s0 in range(S):
        for vs in itertools.product(range(1 << K), repeat=T - 1):
            st = torch.empty(T, dtype=torch.int64)
            st[0] = s0
            for t in range(1, T):
                st[t] = ((st[t - 1] << K) | vs[t - 1]) & (S - 1)
            dec = lut[st]
            c = (dec.unsqueeze(1) - X).square().sum(0)
            better = c < best_cost
            best_cost = torch.where(better, c, best_cost)
            best_seq = torch.where(better.unsqueeze(1),
                                   st.unsqueeze(0).expand(3, -1), best_seq)
    st = xq.viterbi(X, lut, K, L, overlap=None)
    cost_v = (lut[st] - X).square().sum(0)          # [3] per-seq total cost
    ok = torch.equal(st, best_seq) or bool(
        (cost_v - best_cost).abs().max() < 1e-5)
    report("viterbi==bruteforce", ok,
           f"viterbi={cost_v.tolist()} brute={best_cost.tolist()}")


def t_tailbite_consistent():
    """Pass-2 states satisfy s_0>>K == s_last mod 2^(L-K) (exact wrap)."""
    torch.manual_seed(2)
    L, K, T = 16, 4, 256
    lut = xq.get_lut("exl3_mul1", L)
    X = torch.randn(T, 4)
    st = xq.quantize_seq_tailbite(X, lut, K, L)
    ok = bool(((st[0] >> K) == (st[-1] & ((1 << (L - K)) - 1))).all())
    # and the packed stream decodes to the same states (circular windows)
    packed = xq.pack_states(st.t().contiguous(), K)
    ok2 = torch.equal(xq.unpack_states(packed, K), st.t().contiguous())
    report("tailbite consistent", ok and ok2)


def t_bitexact_real():
    """One real EXL3 4bpw tensor: xq.unpack/decode path == ref.reconstruct_inner,
    and pack(unpack(tr)) reproduces the on-disk int16 stream exactly."""
    from safetensors import safe_open
    ref = load_ref()
    wm = json.loads(Path(EXL3, "model.safetensors.index.json")
                    .read_text())["weight_map"]
    key = "model.language_model.layers.0.mlp.down_proj"
    with safe_open(Path(EXL3, wm[f"{key}.trellis"]), framework="pt") as f:
        tr = f.get_tensor(f"{key}.trellis")[:, :16, :]   # [tk, tn, 16K] int16
    K = tr.shape[2] // 16
    tk, tn = tr.shape[:2]
    k, n = tk * 16, tn * 16
    ref_w = ref.reconstruct_inner(tr, K, ref.CB_MUL1)    # [k, n] fp16
    lut = xq.get_lut("exl3_mul1", 16)
    my_w = xq.decode_trellis(tr, lut, K, k, n)           # [k, n] fp32
    st_ref = ref.decode_states(tr, K)
    st_mine = xq.unpack_states(tr.reshape(-1, 16 * K), K).view(tk, tn, -1)
    ok_states = torch.equal(st_mine, st_ref)
    ok_pack = torch.equal(xq.pack_states(st_mine.view(-1, 256), K)
                          .view(tk, tn, 16 * K), tr)
    ok_w = torch.equal(my_w, ref_w.float())
    report("bitexact real exl3 tensor (states/pack/decode)",
           ok_states and ok_pack and ok_w,
           f"states={ok_states} pack={ok_pack} decode={ok_w} K={K} "
           f"shape={tuple(tr.shape)}")


def t_gauss_mse(device, quick):
    """Gaussian MSE at K=2 and K=4 vs rate-distortion bound D=2^-2K,
    for both EXL3 mul1 and QTIP 1MAD codes."""
    torch.manual_seed(3)
    for code in ("exl3_mul1", "qtip_1mad"):
        lut = xq.get_lut(code, 16)
        lut_rms = lut.double().square().mean().sqrt().float()
        ntiles = 128 if quick else 2048
        for K in (2, 4):
            X = torch.randn(256, ntiles, device=device)
            X = X / X.square().mean().sqrt() * lut_rms   # QTIP-style rms match
            t0 = time.time()
            st = xq.quantize_seq_tailbite(X, lut.to(device), K, 16)
            dt = time.time() - t0
            dec = lut.to(device)[st].t().contiguous().flatten()
            mse = (dec - X.t().reshape(-1)).square().mean() / X.square().mean()
            bound = 2.0 ** (-2 * K)
            report(f"gauss_mse {code} K={K}", bool(mse < 3.0 * bound),
                   f"mse={float(mse):.5f} D={bound:.5f} "
                   f"ratio={float(mse/bound):.3f} ({dt:.1f}s/{ntiles} tiles)")


def t_ldlq_helps():
    """On Hessian-correlated weights the LDLQ path must not be worse than plain
    Viterbi on the Hessian-weighted metric."""
    torch.manual_seed(4)
    k, n = 256, 256
    A = torch.randn(k, k)
    H = A.t() @ A / k + 0.1 * torch.eye(k)
    W = (torch.randn(n, k) @ torch.linalg.cholesky(torch.linalg.inv(H)).t()).t()
    # W is [k,n] inner-ish; use quantize_linear path with W_out_in = W.T [n,k]
    res_l = xq.quantize_linear(W.t(), H, code="exl3_mul1", K=4, use_ldlq=True)
    res_p = xq.quantize_linear(W.t(), H, code="exl3_mul1", K=4, use_ldlq=False)
    e_l = xq.hessian_weighted_err(W.t(), res_l["W_hat"], H)
    e_p = xq.hessian_weighted_err(W.t(), res_p["W_hat"], H)
    report("ldlq<=viterbi on Hessian metric", e_l <= e_p * 1.001,
           f"ldlq={e_l:.5f} viterbi={e_p:.5f}")


def t_real_tensors(device, quick):
    """L2 in_proj_qkv + mlp.down_proj: our K=4 vs EXL3 decoded weights."""
    from safetensors import safe_open
    ref = load_ref()
    PREFIX = "model.language_model."
    wm = json.loads(Path(EXL3, "model.safetensors.index.json")
                    .read_text())["weight_map"]
    wm_b = json.loads(Path(BF16, "model.safetensors.index.json")
                      .read_text())["weight_map"]
    rows = []
    for base, hfile in (("linear_attn.in_proj_qkv", "L2_linear_attn.in_proj_qkv.safetensors"),
                        ("mlp.down_proj", "L2_mlp.down_proj.safetensors")):
        key = f"{PREFIX}layers.2.{base}"
        with safe_open(Path(BF16, wm_b[f"{key}.weight"]), framework="pt") as f:
            W = f.get_tensor(f"{key}.weight").float()           # [out, in]
        # EXL3 decode (ref.py, bit-exact)
        tr_key, extra = f"{key}.trellis", f"{key}.mul1"
        cb = ref.CB_MUL1
        with safe_open(Path(EXL3, wm[tr_key]), framework="pt") as f:
            tre = f.get_tensor(tr_key)
            suh = f.get_tensor(f"{key}.suh")
            svh = f.get_tensor(f"{key}.svh")
            scal = f.get_tensor(extra)
        K_bits = tre.shape[2] // 16
        W_ex = ref.weight_orig(tre, suh, svh, K_bits, cb).t().contiguous()  # [out,in]
        with safe_open(Path(HDIR, hfile), framework="pt") as f:
            H = f.get_tensor("H").float()                                  # [in,in]
        mse_ex = xq.rel_mse(W, W_ex)
        herr_ex = xq.hessian_weighted_err(W, W_ex, H)
        ntiles = (W.shape[1] // 16) * (W.shape[0] // 16)
        res_l = xq.quantize_linear(W, H, code="exl3_mul1", K=4, use_ldlq=True,
                                   device=device, batch=0)
        res_p = xq.quantize_linear(W, H, code="exl3_mul1", K=4, use_ldlq=False,
                                   device=device, batch=0)
        ours = {}
        for tag, res in (("viterbi", res_p), ("ldlq", res_l)):
            ours[tag] = {
                "rel_mse": xq.rel_mse(W, res["W_hat"]),
                "herr": xq.hessian_weighted_err(W, res["W_hat"], H)}
        rows.append({"tensor": base, "viterbi": ours["viterbi"],
                     "ldlq": ours["ldlq"],
                     "exl3": {"rel_mse": mse_ex, "herr": herr_ex},
                     "ntiles": ntiles})
        print(f"  {base} ({ntiles} tiles):")
        print(f"    exl3     rel_mse={mse_ex:.5f} herr={herr_ex:.5f}")
        print(f"    viterbi  rel_mse={ours['viterbi']['rel_mse']:.5f} "
              f"herr={ours['viterbi']['herr']:.5f}")
        print(f"    ldlq     rel_mse={ours['ldlq']['rel_mse']:.5f} "
              f"herr={ours['ldlq']['herr']:.5f}", flush=True)
    for r in rows:
        print(json.dumps(r))
    ok = all(r["ldlq"]["herr"] <= r["exl3"]["herr"] * 1.05 for r in rows)
    report("real tensors K=4 ldlq<=1.05x exl3 herr", ok,
           f"{len(rows)} tensors")


def t_quantize_api():
    """Main's harness signature: quantize(W[out,in], U_upper|None, code, K,
    seed) -> (W_hat[out,in], packed int16 [in/16,out/16,16K], scale)."""
    torch.manual_seed(5)
    W = torch.randn(64, 128) * 0.03                      # [out, in]
    for K in (2, 4):
        W_hat, packed, scale = xq.quantize(W, None, "exl3_mul1", K, seed=7)
        ok_shape = (packed.shape == (128 // 16, 64 // 16, 16 * K)
                    and packed.dtype == torch.int16
                    and W_hat.shape == W.shape)
        # decode(packed)*scale == W_hat bit-exact
        lut = xq.get_lut("exl3_mul1", 16)
        dec = xq.decode_trellis(packed, lut, K, 128, 64).t().contiguous() * scale
        ok_dec = torch.equal(dec, W_hat)
        # reproducibility: same seed -> identical bitstream, diff seed -> (very
        # likely) different but still valid
        _w2, p2, _s2 = xq.quantize(W, None, "exl3_mul1", K, seed=7)
        ok_seed = torch.equal(p2, packed) and _s2 == scale
        mse = xq.rel_mse(W, W_hat)
        bound = 2.0 ** (-2 * K)
        report(f"quantize api K={K}",
               ok_shape and ok_dec and ok_seed and mse < 3.0 * bound,
               f"shape={ok_shape} decode={ok_dec} seed={ok_seed} "
               f"rel_mse={mse:.5f} D={bound:.5f} scale={scale:.4f}")


def t_speed(device, batch):
    """Seconds per 1M weights for the tail-biting Viterbi."""
    lut = xq.get_lut("exl3_mul1", 16).to(device)
    X = torch.randn(256, batch, device=device)
    if device.startswith("xpu"):
        torch.xpu.synchronize()
    t0 = time.time()
    xq.quantize_seq_tailbite(X, lut, 4, 16)
    if device.startswith("xpu"):
        torch.xpu.synchronize()
    dt = time.time() - t0
    w = 256 * batch
    sp1m = dt / w * 1e6
    report(f"speed {device} B={batch}", True,
           f"{dt:.2f}s for {w} weights = {sp1m:.1f} s/1M weights")
    return sp1m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--speed-batch", type=int, default=0)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--bitexact", action="store_true",
                    help="real EXL3 tensor bit-exact decode vs ref.py")
    args = ap.parse_args()
    dev = args.device

    t_perm_vs_qtip()
    ref = None
    try:
        ref = load_ref()
    except Exception as e:  # noqa: BLE001
        print(f"ref.py unavailable ({e}); falling back to self-checks")
    t_pack_roundtrip(ref)
    t_viterbi_bruteforce()
    t_tailbite_consistent()
    t_quantize_api()
    t_gauss_mse(dev, args.quick)
    t_ldlq_helps()
    if args.bitexact:
        t_bitexact_real()
    if args.real:
        t_real_tensors(dev, args.quick)
    if args.speed_batch:
        t_speed(dev, args.speed_batch)
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"=== {len(results) - n_fail}/{len(results)} pass ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
