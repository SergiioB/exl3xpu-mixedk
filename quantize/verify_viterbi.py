#!/usr/bin/env python3
"""verify_viterbi.py — generate X + CPU reference states for the SYCL Viterbi,
and compare the kernel's output bit-for-bit.

Usage:
  python3 verify_viterbi.py gen <nseq> <seed> <xbin>            # write X.bin
  python3 verify_viterbi.py check <xbin> <nseq> <code> <margin> <pivot> <kernel_states.bin>
    -> CPU ref states (emit_mode='sq', same formula as the kernel) vs kernel
       states: exact-match fraction per sequence + decoded MSE delta.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import quantize as xq


def gen(nseq, seed, xbin):
    g = np.random.default_rng(seed)
    X = g.standard_normal((nseq, 256), dtype=np.float32)
    X.astype(np.float32).tofile(xbin)
    print(f"wrote {xbin} [{nseq}x256 fp32]")


def check(xbin, nseq, code, margin, pivot, states_bin):
    X = np.fromfile(xbin, dtype=np.float32).reshape(nseq, 256)
    ks = np.fromfile(states_bin, dtype=np.int32)
    meta = {"mul1": ("exl3_mul1", 1, 4), "ih2x2": ("xe_ih2x2", 2, 4)}[code]
    cname, V, K = meta
    nsteps = 256 // V
    ks = ks.reshape(nseq, nsteps)
    lut = xq.get_lut(cname, 16)
    fn, _ = xq.CODES[cname] if cname in xq.CODES else (None, None)

    # CPU reference states, kernel emit formula (f(s)-x)^2, same pivot/margin.
    Xt = torch.from_numpy(X.T.copy()).contiguous()          # [256, nseq]
    st_ref = xq.quantize_seq_tailbite(Xt, lut, K, 16, V=V, pivot=pivot,
                                     margin=margin, emit_mode="sq")
    st_ref = st_ref.t().contiguous().numpy()                # [nseq, nsteps]
    eq_seq = (ks == st_ref).all(axis=1)
    exact_frac = eq_seq.mean()
    n_mismatch = int((ks != st_ref).sum())
    print(f"{code}: exact-state seqs {eq_seq.sum()}/{nseq} "
          f"({exact_frac:.3f}), mismatched states {n_mismatch}/{ks.size}")

    # decoded MSE comparison (kernel states decoded vs ref states decoded)
    if fn is not None:
        dec_k = fn(torch.from_numpy(ks).long()).float().numpy()
        dec_r = fn(torch.from_numpy(st_ref).long()).float().numpy()
    else:
        lut_np = lut.numpy()
        dec_k = lut_np[ks]
        dec_r = lut_np[st_ref]
    mse_k = float(((dec_k.reshape(nseq, -1) - X) ** 2).mean())
    mse_r = float(((dec_r.reshape(nseq, -1) - X) ** 2).mean())
    print(f"{code}: mse kernel={mse_k:.6f} ref={mse_r:.6f} "
          f"delta={mse_k - mse_r:+.2e}")

    # tail-bite consistency of kernel states
    ok = int(((ks[:, 0] >> (K * V)) ==
              (ks[:, -1] & ((1 << (16 - K * V)) - 1))).sum())
    print(f"{code}: circular-consistent seqs {ok}/{nseq}")
    return exact_frac >= 0.99 and abs(mse_k - mse_r) < 1e-4


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        ok = check(sys.argv[2], int(sys.argv[3]), sys.argv[4],
                   int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
        sys.exit(0 if ok else 1)
