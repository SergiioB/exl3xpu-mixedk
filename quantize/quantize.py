#!/usr/bin/env python3
"""quantize.py — EXL3-format trellis quantizer for Intel Arc Pro B70 (Xe2).

Algorithm: QTIP BlockLDLQ + tail-biting Viterbi over a bitshift trellis
(arXiv 2406.11235; github.com/Cornell-RelaxML/qtip lib/algo/ldlq.py,
lib/codebook/bitshift.py — reimplemented, not imported), with the EXL3 format's
incoherence and bitstream layout (ref.py, sha 3f614b7c...):

  W_orig[k,n] = diag(suh) . H128_blocks(k) . W_inner . H128_blocks(n) . diag(svh)

Tile = 16x16 weights = one 256-element trellis-coded sequence. Element t of the
sequence is decoded from the 16-bit circular window ending at bit (t+1)*K of the
tile bitstream and lands at tile position perm[t] (ref.tensor_core_perm).
State recursion: s_t = ((s_{t-1} << K) | v_t) & (2^L - 1), output f(s_t) = lut[s_t].

Tail-biting Viterbi follows QTIP: pass 1 Viterbi on the half-rolled sequence
(free boundary), pass 2 Viterbi with the wrap bits pinned to the pass-1 state
(overlap constraint, exact circular consistency given the chosen boundary bits).

BlockLDLQ: right-to-left sweep over 16-wide Hessian tiles with the block-LDL
factor of the rotated, damped Hessian (identity 16x16 diagonal blocks, intra-tile
coupling absorbed by the joint tile quantizer, cross-tile feedback only).

Weight orientation: public functions take nn.Linear layout W [out, in] and the
input Gram H [in, in] in the ORIGINAL (unrotated) basis. Internally everything
runs in ref.py orientation W_i [k=in, n=out].

Codes come from the local fallback tables below (EXL3 3INST/MCG/MUL1 bit-exact
vs the pinned exl3xpu ref.py arithmetic, and QTIP 1MAD). An optional external
registry module `trellis_codes` (CODES[name] -> (decode_fn, metadata)) is used
when importable; it is not part of this release.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

# ----------------------------------------------------------------- code lookup
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import trellis_codes as _codes_mod
    CODES = _codes_mod.CODES
except Exception:  # noqa: BLE001 - optional module, not shipped
    CODES = {}


def _lut_from_bits(fn, L):
    return fn(torch.arange(1 << L, dtype=torch.int64))


def _local_code_lut(name: str, L: int = 16) -> torch.Tensor:
    """Fallback fp32 decode LUTs. 'exl3*' entries reproduce ref.py bit-for-bit
    (same integer ops, same fp16 rounding); 'qtip_1mad' is QTIP decode_1mad."""
    s = np.arange(1 << L, dtype=np.uint64)

    def f16_from_bits(u):
        return u.astype(np.uint16).view(np.float16)

    if name in ("exl3_3inst", "exl3_mcg"):
        if name == "exl3_3inst":
            x = (s * 89226354 + 64248484) & 0xFFFFFFFF
        else:
            x = (s * 0xCBAC1FED) & 0xFFFFFFFF
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        lo = f16_from_bits(x & 0xFFFF).astype(np.float64)
        hi = f16_from_bits(x >> 16).astype(np.float64)
        return torch.from_numpy((lo + hi).astype(np.float16).astype(np.float32).copy())
    if name == "exl3_mul1":
        x = (s * 0x83DCD12D) & 0xFFFFFFFF
        bsum = ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF)
                + ((x >> 24) & 0xFF))
        h = (1024 + bsum).astype(np.float64)
        inv = float(f16_from_bits(np.array([0x1EEE]))[0])
        bias = float(f16_from_bits(np.array([0xC931]))[0])
        return torch.from_numpy((h * inv + bias).astype(np.float16)
                                .astype(np.float32).copy())
    if name == "qtip_1mad":
        x = (s * 34038481 + 76625530) & 0xFFFFFFFF
        y = ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255)
             + ((x >> 24) & 255)).astype(np.float64)
        return torch.from_numpy(((y - 510.0) / 147.800537109375).astype(np.float32))
    raise KeyError(f"unknown local code {name!r}")


def get_lut(code: str, L: int = 16, device=None) -> torch.Tensor:
    """fp32 [2^L] decode table f(state) for the named code (V=1 codes only)."""
    if code in CODES:
        fn, _meta = CODES[code]
        lut = _lut_from_bits(fn, L).float()
    else:
        lut = _local_code_lut(code, L)
    return lut.to(device) if device is not None else lut


# ------------------------------------------------------------- format constants
def tensor_core_perm() -> torch.Tensor:
    """ref.py tensor_core_perm: element t (sequence order) -> tile position
    (row-major index in the 16x16 [k,n] tile)."""
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1, r2, r3 = r0 + 1, r0 + 8, r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm[t * 8 + 0] = r0 * 16 + c0
        perm[t * 8 + 1] = r1 * 16 + c0
        perm[t * 8 + 2] = r2 * 16 + c0
        perm[t * 8 + 3] = r3 * 16 + c0
        perm[t * 8 + 4] = r0 * 16 + c1
        perm[t * 8 + 5] = r1 * 16 + c1
        perm[t * 8 + 6] = r2 * 16 + c1
        perm[t * 8 + 7] = r3 * 16 + c1
    return torch.tensor(perm, dtype=torch.long)


# -------------------------------------------------------------- Hadamard (EXL3)
_H128 = {}


def hadamard128(device=None, dtype=torch.float32) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _H128:
        h = torch.ones((1, 1), dtype=torch.float64)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H128[key] = (h / math.sqrt(128)).to(dtype=dtype, device=device)
    return _H128[key]


def had_left(w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Block-128 Hadamard on the FIRST (k) dim of [k, n]: H_blk @ w."""
    k, n = w.shape
    assert k % 128 == 0
    return (h @ w.view(k // 128, 128, n)).view(k, n)


def had_right(w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Block-128 Hadamard on the LAST (n) dim of [k, n]: w @ H_blk."""
    k, n = w.shape
    assert n % 128 == 0
    return (w.view(k, n // 128, 128) @ h).view(k, n)


def to_inner(w_orig: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
             h: torch.Tensor) -> torch.Tensor:
    """W_orig [k,n] -> W_inner: B_k diag(1/suh) W diag(1/svh) B_n (H is its own
    inverse: normalized Sylvester, H H = I)."""
    w = w_orig / suh.float().unsqueeze(1)
    w = w / svh.float().unsqueeze(0)
    return had_right(had_left(w, h), h)


def from_inner(w_inner: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
               h: torch.Tensor) -> torch.Tensor:
    """W_inner -> W_orig = diag(suh) B_k W_i B_n diag(svh) (ref.weight_orig)."""
    w = had_left(w_inner, h)
    w = w * suh.float().unsqueeze(1)
    w = had_right(w, h)
    return w * svh.float().unsqueeze(0)


def rotate_hessian(H_x: torch.Tensor, suh: torch.Tensor,
                   h: torch.Tensor) -> torch.Tensor:
    """Inner-basis input Gram: G = B diag(suh) H_x diag(suh) B
    (Gram of the inner-domain effective inputs had(x*suh))."""
    k = H_x.shape[0]
    a = H_x.float() * suh.float().unsqueeze(0) * suh.float().unsqueeze(1)
    # B A B = had_right(had_right(A)^T)^T ... expand: had_right applies B on the
    # last dim. B A B = (had_right(A^T))^T computed as had_right(had_right(A).T).T
    # (B symmetric, A symmetric).
    return had_right(had_right(a.view(k, k), h).t().contiguous(), h)


def sign_vec(n: int, seed: int, device=None) -> torch.Tensor:
    """Deterministic Rademacher sign vector."""
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(device)


# --------------------------------------------------------------------- Viterbi
def _predecessors(L: int, KV: int, device):
    """Predecessor table for the shift-register trellis.

    Next-state s' = ((p << KV) | v) & (2^L-1) implies p % 2^(L-KV) = s' >> KV,
    so the predecessors of any next-state with s' >> KV = j are
    {j + w*2^(L-KV) : w in [0, 2^KV)}. cand[w, j] = j + w*S_PRE indexes
    cost.view(B, 2^KV, S_PRE)[b, w, j]."""
    S_PRE = 1 << (L - KV)
    j = torch.arange(S_PRE, device=device).view(1, -1)
    w = torch.arange(1 << KV, device=device).view(-1, 1)
    return (j + w * S_PRE).contiguous()          # [NC, S_PRE]


_STEP_CACHE = {}


def _step_fn(B, NC, S_PRE, dtype, compiled):
    """One DP step, optionally torch.compile'd (measured 3.45x on one B70:
    32.8 -> 9.5 ms/step at B=8192). Returns (cost_next, argmin_v)."""
    if not compiled:
        return None
    key = (B, NC, S_PRE, str(dtype))
    if key not in _STEP_CACHE:
        def f(cost, xneg_i, LUT, L2):
            best = cost.view(B, NC, S_PRE).min(dim=1)
            e = torch.addmm(L2, xneg_i, LUT)
            return e.view(B, S_PRE, NC).add_(
                best.values.unsqueeze(-1)).view(B, -1), best.indices
        _STEP_CACHE[key] = torch.compile(f)
    return _STEP_CACHE[key]


def viterbi(X: torch.Tensor, lut: torch.Tensor, K: int, L: int, V: int = 1,
            overlap: torch.Tensor = None, dtype: torch.dtype = torch.float32,
            back_from: int | None = None, compiled: bool = False,
            emit_mode: str = "sq"):
    """Trellis Viterbi for X [T, B] targets. Returns states [T//V, B] int64;
    with back_from set, stores predecessor links only for steps > back_from
    and returns states[back_from] [B] (the pathwise state at that step — used
    by the cheap boundary pass, which needs only a margin of lookahead).

    cost[s] = min path cost of a state sequence ending AT s with emit(s)=lut[s].
    Tail-biting: overlap [B] pins s_0>>KV == overlap and s_{T-1} mod 2^(L-KV)
    == overlap (exact circular consistency of the L-K wrap bits)."""
    T, B = X.shape
    assert T % V == 0
    KV = K * V
    S = 1 << L
    NC = 1 << KV
    S_PRE = 1 << (L - KV)
    dev = X.device
    X = X.to(dtype)
    lut_v = lut.to(dev, torch.float32)
    if lut_v.dim() == 1:
        lut_v = lut_v.unsqueeze(0)                 # [V=1, S]
    elif lut_v.dim() == 2 and lut_v.shape[0] != V:
        lut_v = lut_v.t()                          # CODES module gives [S, V]
    assert lut_v.shape == (V, S)

    INF = torch.tensor(float("inf"), dtype=dtype, device=dev)

    # emit via one GEMM per step: err[b,s] = sum_v lut_v[v,s]^2
    # - 2 sum_v X[iV+v, b] * lut_v[v,s] (+ ||x||^2, constant per row, dropped —
    # argmin-invariant).  LUT [V,S], Xneg [B,T] with Xneg = -2 X^T.
    LUT = lut_v.to(dtype).contiguous()              # [V, S]
    L2 = LUT.square().sum(0, keepdim=True)          # [1, S] addmm bias (broadcast)
    Xneg = (-2.0 * X.t()).to(dtype).contiguous()    # [B, T]

    def emit(i, out=None):
        if emit_mode == "sq":
            # explicit (f(s)-x)^2 elementwise — the SYCL kernel's exact formula
            e = (LUT[0] - X[i * V].unsqueeze(-1)).square()
            for v in range(1, V):
                e = e + (LUT[v] - X[i * V + v].unsqueeze(-1)).square()
            return e if out is None else out.copy_(e)
        return torch.addmm(L2, Xneg[:, i * V:(i + 1) * V], LUT, out=out)

    nsteps = T // V
    cost = emit(0)                                  # free boundary over all states
    if overlap is not None:
        allow = (overlap.view(B, 1).to(torch.int64) << KV) + \
            torch.arange(NC, device=dev).view(1, NC)
        mask = torch.full((B, S), INF, dtype=dtype, device=dev)
        mask.scatter_(1, allow, 0.0)
        cost = cost + mask                          # pin s_0 >> KV == overlap

    fs_dtype = torch.uint8 if NC <= 256 else torch.int16
    if back_from is None:
        from_state = torch.empty(nsteps, B, S_PRE, dtype=fs_dtype, device=dev)
    else:
        nkeep = max(0, nsteps - 1 - back_from)
        from_state = torch.empty(nkeep, B, S_PRE, dtype=fs_dtype, device=dev)
    e_buf = None if compiled else torch.empty(B, S, dtype=dtype, device=dev)
    cstep = _step_fn(B, NC, S_PRE, dtype, compiled)
    for i in range(1, nsteps):
        if cstep is not None:
            cost, idx = cstep(cost, Xneg[:, i * V:(i + 1) * V], LUT, L2)
            if back_from is None or i > back_from:
                from_state[i if back_from is None else i - 1 - back_from] = \
                    idx.to(fs_dtype)
            continue
        # candidates: cost[b, j + w*S_PRE] = view [B, NC, S_PRE] -> min over w
        cand = cost.view(B, NC, S_PRE)
        best = cand.min(dim=1)                      # values [B,S_PRE], idx [B,S_PRE]
        if back_from is None or i > back_from:
            from_state[i if back_from is None else i - 1 - back_from] = \
                best.indices.to(fs_dtype)
        e = emit(i, out=e_buf)
        # cost_new[b, s'=j*NC+v] = emit[b,s'] + best.values[b,j]
        e.view(B, S_PRE, NC).add_(best.values.unsqueeze(-1))
        cost = e

    if overlap is not None:
        allow = overlap.view(B, 1).to(torch.int64) + \
            (torch.arange(NC, device=dev).view(1, NC) << (L - KV))
        mask = torch.full((B, S), INF, dtype=dtype, device=dev)
        mask.scatter_(1, allow, 0.0)
        cost = cost + mask                          # pin s_last mod 2^(L-KV)

    if back_from is not None:
        s = cost.argmin(dim=1)
        for i in range(nsteps - 1, back_from, -1):
            j_i = s >> KV
            w = from_state[i - 1 - back_from].gather(
                1, j_i.unsqueeze(1)).squeeze(1).to(torch.int64)
            s = j_i + w * S_PRE
        return s                                    # states[back_from] only

    states = torch.empty(nsteps, B, dtype=torch.int64, device=dev)
    states[-1] = cost.argmin(dim=1)
    for i in range(nsteps - 1, 0, -1):
        j_i = states[i] >> KV                       # predecessor group
        w = from_state[i].gather(1, j_i.unsqueeze(1)).squeeze(1).to(torch.int64)
        states[i - 1] = j_i + w * S_PRE
    return states


def quantize_seq_tailbite(X: torch.Tensor, lut: torch.Tensor, K: int, L: int,
                          V: int = 1, dtype: torch.dtype = torch.float32,
                          pivot: int | None = None, margin: int = 0,
                          compiled: bool = False, emit_mode: str = "sq"):
    """QTIP tail-biting Viterbi: pass 1 on rolled X (free boundary) -> wrap
    bits; pass 2 constrained. X [T,B] -> states [T//V,B] int64.

    pivot: pass-1 boundary position in step units (default T/(2V)). Any pivot in
    [0, T/V) is valid — the wrap bits are pinned to the pass-1 state that lands
    on original step 0 either way.

    margin > 0 runs pass 1 only through step pivot+margin with backtrace links
    kept only after the pivot (survivors coalesce in ~L/KV steps, so a short
    lookahead is enough; full pass when margin is 0 or reaches the end)."""
    T = X.shape[0]
    KV = K * V
    nsteps = T // V
    if pivot is None:
        pivot = nsteps // 2
    assert 0 <= pivot < nsteps
    roll = torch.roll(X, pivot * V, 0)
    if margin and pivot + margin < nsteps:
        s_p = viterbi(roll[:(pivot + margin) * V], lut, K, L, V, overlap=None,
                      dtype=dtype, back_from=pivot, compiled=compiled,
                      emit_mode=emit_mode)
    else:
        st1 = viterbi(roll, lut, K, L, V, overlap=None, dtype=dtype,
                      compiled=compiled, emit_mode=emit_mode)
        s_p = st1[pivot]
    overlap = (s_p >> KV).contiguous()
    return viterbi(X, lut, K, L, V, overlap=overlap, dtype=dtype,
                   compiled=compiled, emit_mode=emit_mode)


# ------------------------------------------------------------ pack (ref.py bit
# layout): circular 256*K-bit stream, v_t at bits [tK, tK+K) MSB-first; u32 words
# MSB-first, stored as int16 pairs low-half-first.
def pack_states(states: torch.Tensor, K: int, V: int = 1) -> torch.Tensor:
    """states int64 [ntiles, 256//V] -> int16 [ntiles, 16*K] (ref.py trellis
    slice). sym_t = low K*V bits of state_t at stream bits [tKV, (t+1)KV).
    Chunked over tiles: the bit-expansion temps stay ~128 MiB."""
    nt, T = states.shape
    KV = K * V
    ch = max(1, (1 << 24) // (T * KV))
    out = torch.empty(nt, 16 * K, dtype=torch.int16, device=states.device)
    ar_kv = torch.arange(KV - 1, -1, -1, device=states.device)
    ar_32 = torch.arange(31, -1, -1, device=states.device)
    for a in range(0, nt, ch):
        b = min(nt, a + ch)
        v = (states[a:b] & ((1 << KV) - 1)).unsqueeze(-1) >> ar_kv
        stream = (v & 1).reshape(b - a, T * KV).to(torch.int64)
        w32 = (stream.view(b - a, -1, 32) * (1 << ar_32)).sum(-1)
        lo = (w32 & 0xFFFF).to(torch.int16)
        hi = (w32 >> 16).to(torch.int16)
        out[a:b] = torch.stack([lo, hi], dim=-1).reshape(b - a, 16 * K)
    return out


def unpack_states(trellis: torch.Tensor, K: int, V: int = 1) -> torch.Tensor:
    """int16 [.., 16*K] -> states int64 [.., 256//V]
    (self-contained ref.decode_states; for V>1 the window ends at (t+1)*K*V).
    Chunked over tiles: bit-window temps stay ~256 MiB."""
    KV = K * V
    nbits = 256 * K
    t = torch.arange(256 // V, device=trellis.device)
    b0 = (t * KV + KV - 16 + nbits) % nbits
    pos = (b0[:, None] + torch.arange(16, device=trellis.device)[None, :]) % nbits
    shifts = torch.arange(31, -1, -1, device=trellis.device)
    wts = (1 << torch.arange(15, -1, -1, device=trellis.device))
    flat = trellis.reshape(-1, 16 * K)
    ch = max(1, (1 << 22) // nbits)
    out = torch.empty(flat.shape[0], 256 // V, dtype=torch.int64,
                      device=trellis.device)
    for a in range(0, flat.shape[0], ch):
        b = min(flat.shape[0], a + ch)
        t16 = flat[a:b].to(torch.int64) & 0xFFFF
        words = t16[..., 0::2] | (t16[..., 1::2] << 16)
        bits = ((words.unsqueeze(-1) >> shifts) & 1).flatten(-2)
        out[a:b] = (bits[..., pos] * wts).sum(-1)
    return out.view(*trellis.shape[:-1], 256 // V)


def decode_trellis(trellis: torch.Tensor, lut: torch.Tensor, K: int,
                   k: int, n: int, V: int = 1) -> torch.Tensor:
    """int16 [k/16, n/16, 16K] -> W_inner fp32 [k, n] (positions via perm)."""
    tk, tn, _ = trellis.shape
    perm = tensor_core_perm().to(trellis.device)
    st = unpack_states(trellis.reshape(tk * tn, 16 * K), K, V)
    vals = lut.to(device=st.device, dtype=torch.float32)[st] \
        .reshape(st.shape[0], -1)                                # [tiles,256]
    tiles = torch.empty_like(vals)
    tiles[:, perm] = vals
    return tiles.view(tk, tn, 16, 16).permute(0, 2, 1, 3).reshape(k, n)


# -------------------------------------------------------------------- block-LDL
def block_LDL(H: torch.Tensor, b: int):
    """QTIP block LDL (lib/utils/math_utils.py port): H -> L with identity bxb
    diagonal blocks and D (block-diagonal SPD). L is consumed."""
    n = H.shape[0]
    assert n % b == 0
    m = n // b
    Lm = torch.linalg.cholesky(H)
    DL = torch.diagonal(Lm.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = DL @ DL.permute(0, 2, 1)
    DLi = torch.linalg.inv(DL)
    Lm = Lm.view(n, m, b)
    for i in range(m):
        Lm[:, i, :] = Lm[:, i, :] @ DLi[i]
    Lm = Lm.reshape(n, n).contiguous()
    Lb = Lm.view(m, b, m, b)
    eye = torch.eye(b, dtype=Lm.dtype, device=Lm.device)
    idx = torch.arange(m, device=Lm.device)
    Lb[idx, :, idx, :] = eye
    return Lm, D


# ------------------------------------------------------------------ LDLQ sweep
# ---------------------------------------------------------- ESIMD backend
# Verified ESIMD tail-biting Viterbi (viterbi_esimd.cpp in this directory),
# bit-exact with the torch emit (f-x)^2 path.  Two transports:
#   "proc" (default when the binary exists): persistent `viterbi_esimd serve`
#     subprocess on a framed stdin/stdout protocol.  The child's env gets
#     ONEAPI_DEVICE_SELECTOR=level_zero:<XTQ_VIT_DEVIDX> *explicitly*, so the
#     kernel can run on a different card than the parent (torch) process
#     — the ctypes path cannot do that.  One worker lifetime
#     amortizes the SYCL JIT compile.  The backend never takes a GPU lock;
#     the launcher owns any lock file.
#   "ctypes": in-process libviterbi_esimd.so — only sees devices the parent's
#     ONEAPI_DEVICE_SELECTOR exposes.  XTQ_VIT_LOCK="auto" (or a path) opts
#     into flock /tmp/b70-gpu<devidx>.lock inside this process.
# Env: XTQ_VIT_TRANSPORT=proc|ctypes|auto (default auto); XTQ_VIT_DEVIDX=N
#   (Level Zero device index for the worker); XTQ_VIT_BIN=worker binary path
#   (default build/viterbi_esimd next to this file);
#   XTQ_VIT_MAXSEQ (launch chunk, 1024); XTQ_VIT_MAXBYTES device cap (6 GiB,
#   also enforced in C++).
_ESIMD_LIB = None
_ESIMD_PROC = None
_DUAL_STATS: dict = {}
_ESIMD_CODES = {"exl3_mul1": "mul1p", "xe_ih2x2": "ih2x2"}
_ESIMD_MAXSEQ = int(os.environ.get("XTQ_VIT_MAXSEQ", "1024"))
_SERVE_MAGIC = 0x58545156
_ESIMD_PRELOAD = (
    "/opt/intel/oneapi/tcm/*/lib/libhwloc.so.*",
    "/opt/intel/oneapi/umf/*/lib/libumf.so.*",
    "/opt/intel/oneapi/compiler/*/lib/libur_loader.so.*",
    "/opt/intel/oneapi/compiler/*/lib/libur_adapter_level_zero.so.*",
)


def _esimd_lib():
    global _ESIMD_LIB
    if _ESIMD_LIB is None:
        import ctypes
        import glob
        for pat in _ESIMD_PRELOAD:
            for so in sorted(glob.glob(pat)):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        libpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "build", "libviterbi_esimd.so")
        lib = ctypes.CDLL(libpath)
        lock = os.environ.get("XTQ_VIT_LOCK", "")
        if lock:
            # opt-in flock so the backend can serialize on /tmp/b70-gpuN.lock
            import fcntl
            if lock == "auto":
                idx = os.environ.get("XTQ_VIT_DEVIDX", "0")
                lock = f"/tmp/b70-gpu{idx}.lock"
            global _ESIMD_LOCKFD
            _ESIMD_LOCKFD = open(lock, "w")
            fcntl.flock(_ESIMD_LOCKFD, fcntl.LOCK_EX)
        lib.xtq_vit_init.restype = ctypes.c_int
        lib.xtq_vit_run.restype = ctypes.c_int
        lib.xtq_vit_run.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_char_p, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_void_p]
        if lib.xtq_vit_init() != 0:
            raise RuntimeError("xtq_vit_init failed (no SYCL GPU?)")
        _ESIMD_LIB = lib
    return _ESIMD_LIB


def _esimd_bin() -> str:
    return os.environ.get("XTQ_VIT_BIN") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "build", "viterbi_esimd")


def _esimd_transport() -> str:
    t = os.environ.get("XTQ_VIT_TRANSPORT", "auto")
    if t != "auto":
        return t
    return "proc" if os.path.exists(_esimd_bin()) else "ctypes"


def _readn(fd: int, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = os.read(fd, n - len(out))
        if not chunk:
            raise RuntimeError("xtq_vit worker closed stdout")
        out += chunk
    return bytes(out)


def _esimd_proc():
    """Persistent `viterbi_esimd serve` child with its own device selector —
    this lets the backend use a different physical card than the parent
    process (torch) is restricted to."""
    global _ESIMD_PROC
    if _ESIMD_PROC is None or _ESIMD_PROC.poll() is not None:
        import subprocess
        import glob
        env = dict(os.environ)
        env["ONEAPI_DEVICE_SELECTOR"] = (
            f"level_zero:{int(os.environ.get('XTQ_VIT_DEVIDX', '0'))}")
        libdirs = []
        latest = "/opt/intel/oneapi/compiler/latest/lib"
        if os.path.isdir(latest):
            libdirs.append(latest)          # must precede older compiler dirs
        for pat in _ESIMD_PRELOAD:
            for so in sorted(glob.glob(pat), reverse=True):
                d = os.path.realpath(os.path.dirname(so))
                if d not in [os.path.realpath(x) for x in libdirs]:
                    libdirs.append(os.path.dirname(so))
        if libdirs:   # oneAPI compiler/lib + transitive SYCL/UR deps
            env["LD_LIBRARY_PATH"] = ":".join(libdirs) + (
                ":" + env["LD_LIBRARY_PATH"]
                if env.get("LD_LIBRARY_PATH") else "")
        p = subprocess.Popen([_esimd_bin(), "serve"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             env=env, bufsize=0)      # stderr inherited
        _ESIMD_PROC = p
        import atexit
        import struct as _st

        def _quit(_p=p):
            try:
                os.write(_p.stdin.fileno(),
                         _st.pack("<IIiiQQ16s", _SERVE_MAGIC, 0, 0, 0, 0, 0,
                                  b"\0" * 16))
            except Exception:
                pass
        atexit.register(_quit)
    return _ESIMD_PROC


def _esimd_submit(x: np.ndarray, kcode: str, margin: int, pivot: int,
                  st: np.ndarray) -> None:
    """One chunk: x [nseq,256] fp32 -> st [nseq,nsteps] int32."""
    if _esimd_transport() == "proc":
        import struct
        p = _esimd_proc()
        hdr = struct.pack("<IIiiQQ16s", _SERVE_MAGIC, 1, margin, pivot,
                          x.shape[0], x.nbytes,
                          kcode.encode().ljust(16, b"\0"))
        mv = memoryview(x)
        sent = os.write(p.stdin.fileno(), hdr)
        assert sent == len(hdr)
        off = 0
        while off < x.nbytes:
            off += os.write(p.stdin.fileno(), mv[off:])
        magic, rc, nbytes = struct.unpack(
            "<IiQ", _readn(p.stdout.fileno(), 16))
        if magic != _SERVE_MAGIC:
            raise RuntimeError(f"xtq_vit serve bad magic {magic:#x}")
        if rc:
            raise RuntimeError(f"xtq_vit serve rc={rc}")
        want = st.size * 4
        if nbytes != want:
            raise RuntimeError(
                f"xtq_vit serve nbytes={nbytes} want={want}")
        st[:] = np.frombuffer(
            _readn(p.stdout.fileno(), nbytes), dtype=np.int32
        ).reshape(st.shape)
        return
    import ctypes
    lib = _esimd_lib()
    rc = lib.xtq_vit_run(
        x.ctypes.data_as(ctypes.c_void_p), x.shape[0], kcode.encode(),
        margin, pivot, st.ctypes.data_as(ctypes.c_void_p))
    if rc:
        raise RuntimeError(f"xtq_vit_run({kcode}) rc={rc}")


def _esimd_tailbite(X: torch.Tensor, code: str, K: int, L: int, V: int,
                    pivot: int | None, margin: int) -> torch.Tensor:
    """ESIMD tail-biting Viterbi. X [T, nseq] fp32 -> states [T//V, nseq]
    int64, bit-exact with quantize_seq_tailbite(margin, pivot)."""
    kcode = _ESIMD_CODES.get(code)
    if kcode is None:
        raise ValueError(f"backend='esimd' has no kernel for code {code!r}; "
                         f"supported: {sorted(_ESIMD_CODES)}")
    T, nseq = X.shape
    assert T == 256, f"esimd kernel expects 256-target sequences, got T={T}"
    nsteps = T // V
    pv = nsteps // 2 if pivot is None else int(pivot)
    x = np.ascontiguousarray(X.detach().t().cpu().numpy(), dtype=np.float32)
    st = np.empty((nseq, nsteps), dtype=np.int32)
    for s0 in range(0, nseq, _ESIMD_MAXSEQ):
        s1 = min(nseq, s0 + _ESIMD_MAXSEQ)
        _esimd_submit(x[s0:s1], kcode, margin, pv, st[s0:s1])
    return torch.from_numpy(st).t().contiguous().long()


def ldlq_viterbi(Wq: torch.Tensor, Lfac: torch.Tensor | None, lut: torch.Tensor,
                 K: int, L: int, buf_cols: int = 128, V: int = 1,
                 batch: int = 0, dtype: torch.dtype = torch.float32,
                 pivot: int | None = None, margin: int = 0,
                 compiled: bool = False, progress=None,
                 code: str | None = None, backend: str = "torch"):
    """BlockLDLQ trellis quantization of W_i [k, n] (unit-scaled inner weights).

    Sweeps 16-row k-tiles right-to-left; each 16x16 tile is one 256-step
    tail-biting Viterbi sequence. Lfac [k,k] is the block-LDL factor of the
    rotated damped Hessian (identity 16-blocks on the diagonal); pass Lfac=None
    or zeros for pure (unweighted) Viterbi. Targets enter the sequence in
    ref-format order: x[t] = tile.flat[perm[t]].

    Returns (hatW [k,n] fp32, states int64 [k/16, n/16, 256]).
    """
    k, n = Wq.shape
    assert k % 16 == 0 and n % 16 == 0
    dev = Wq.device
    lut = lut.to(dev)
    perm = tensor_core_perm().to(dev)
    k_perm, n_perm = perm // 16, perm % 16
    no_ldlq = Lfac is None or (isinstance(Lfac, torch.Tensor) and not Lfac.any())
    if no_ldlq:
        Lfac = torch.zeros(1, 1, dtype=Wq.dtype, device=dev)
    else:
        Lfac = Lfac.to(Wq.device, Wq.dtype)
    tk, tn = k // 16, n // 16

    W_t = Wq.t().contiguous()                       # [n, k]: QTIP's Wr_T layout
    hatW_t = torch.zeros_like(W_t)
    prod = torch.zeros(k, n, dtype=Wq.dtype, device=dev)  # feedback cache [icol, out]
    states = torch.empty(tn, tk, 256 // V, dtype=torch.int64, device=dev)

    buf_tiles = max(1, buf_cols // 16)
    n_steps = 2 * (256 // V)                        # pass1 + pass2 Viterbi steps
    step_i = 0
    for cur in range(tk, 0, -buf_tiles):
        lo = max(0, cur - buf_tiles)
        # rows of W_t / hatW_t indexed by OUTPUT n; the sweep indexes W rows k,
        # which are COLUMNS of W_t. Work on column slices [16i,16i+16) of W_t.T —
        # equivalently row slices of Wq. Keep QTIP's exact arithmetic:
        # WXWX_i = W_T[cols_i] + L[later, i]^T @ err[later] + prod[cols_i]
        for i in range(cur - 1, lo - 1, -1):
            c0, c1 = 16 * i, 16 * (i + 1)
            WXWX = W_t[:, c0:c1].t().contiguous()   # [16, n]
            if not no_ldlq:
                later0 = 16 * (i + 1)
                if later0 < 16 * cur:
                    WXWX = WXWX + Lfac[later0:16 * cur, c0:c1].t() @ \
                        (W_t[:, later0:16 * cur] - hatW_t[:, later0:16 * cur]).t()
                WXWX = WXWX + prod[c0:c1, :]
            # sequence targets: x[t, j_tile] = WXWX[k_perm[t], 16*j + n_perm[t]]
            Xw = WXWX[k_perm].view(256, tn, 16)
            X = Xw.gather(2, n_perm.view(256, 1, 1)
                          .expand(256, tn, 1)).squeeze(2)   # [256, tn]
            if backend == "dual":
                st_t = quantize_seq_tailbite(X, lut, K, L, V, dtype=dtype,
                                             pivot=pivot, margin=margin,
                                             compiled=compiled)
                st_e = _esimd_tailbite(X, code, K, L, V, pivot, margin)
                neq = int((st_t.cpu() != st_e).sum())
                _DUAL_STATS["launches"] += 1
                _DUAL_STATS["mismatch_states"] += neq
                if neq and not _DUAL_STATS.get("first"):
                    d = (st_t.cpu() != st_e).nonzero()
                    _DUAL_STATS["first"] = {"block": i, "count": neq,
                                            "idx": d[:5].tolist()}
                    costs = []
                    for j in d[:, 1].unique()[:3].tolist():
                        lt = lut.to(device=X.device, dtype=torch.float32)
                        c_t = float(((lt[st_t[:, j]] - X[:, j]) ** 2).sum())
                        c_e = float(((lt[st_e[:, j].to(X.device)]
                                      - X[:, j]) ** 2).sum())
                        costs.append({"seq": j, "cost_torch": c_t,
                                      "cost_esimd": c_e})
                    _DUAL_STATS["first"]["costs"] = costs
                    try:
                        j0 = int(d[0, 1])
                        torch.save({"X": X[:, j0].cpu(),
                                    "st_torch": st_t[:, j0].cpu(),
                                    "st_esimd": st_e[:, j0]},
                                   os.environ.get("XTQ_DUMP",
                                                  "/tmp/xtq_dual_mismatch.pt"))
                    except Exception:
                        pass
                states[:, i] = st_t.t()
            elif backend == "esimd":
                st = _esimd_tailbite(X, code, K, L, V, pivot, margin)
                states[:, i] = st.t()
            elif batch and tn > batch:
                for s0 in range(0, tn, batch):
                    st = quantize_seq_tailbite(X[:, s0:s0 + batch], lut, K, L, V,
                                               dtype=dtype, pivot=pivot,
                                               margin=margin, compiled=compiled)
                    states[s0:s0 + batch, i] = st.t()
            else:
                st = quantize_seq_tailbite(X, lut, K, L, V, dtype=dtype,
                                           pivot=pivot, margin=margin,
                                           compiled=compiled)
                states[:, i] = st.t()
            step_i += n_steps
            if progress is not None:
                progress(i, step_i)
            # write decoded tile back
            vals = lut.to(torch.float32)[states[:, i].to(dev)] \
                .reshape(tn, 256)                                   # [tn,256]
            tiles = torch.empty_like(vals)
            tiles[:, perm] = vals                                   # positions
            hatW_t[:, c0:c1] = tiles.view(tn, 16, 16).permute(1, 0, 2) \
                .reshape(16, n).t()
        if not no_ldlq and lo > 0:
            prod += Lfac[16 * lo:16 * cur, :].t() @ \
                (W_t[:, 16 * lo:16 * cur] - hatW_t[:, 16 * lo:16 * cur]).t()
    # states layout: [tn, tk, 256] -> want [tk, tn, 256]
    return hatW_t.t().contiguous(), states.permute(1, 0, 2).contiguous()


# ----------------------------------------------------------------- top level
def quantize_linear(W: torch.Tensor, H_x: torch.Tensor | None, code: str = "exl3_mul1",
                    K: int = 4, L: int = 16, V: int = 0, percdamp: float = 0.01,
                    su_seed: int = 20261011, buf_cols: int = 128,
                    use_ldlq: bool = True, device: str = "cpu",
                    batch: int = 0, dtype: torch.dtype = torch.float32,
                    margin: int = 0, compiled: bool = False, progress=None,
                    backend: str = "torch", channel_scales: bool = False):
    """Quantize one nn.Linear weight.

    W [out, in] fp32/bf16; H_x [in, in] fp32 undamped input Gram (None -> pure
    Viterbi on incoherent weights). Returns a dict with the reconstructed
    original-basis weight, the packed ref-layout trellis, sign/scale vectors
    and diagnostics.

    channel_scales=True applies EXL3 1.4.2 per-channel scaling
    (exl3_lib/quantize.py regularize(), lines 1178-1211): per-output-channel
    RMS folded into sv, output Hadamard, then per-input-channel RMS of the
    transformed weight folded into su, input Hadamard.  suh/svh returned are
    then fp16 sign*magnitude vectors (suh also absorbs Wscale), so
    ref.weight_orig(trellis, suh, svh, K, cb) reproduces W_hat bit-exactly.
    The Hessian rotation stays sign-only, matching EXL3's finalize_capture_H
    (lines 891-898 apply only the su SIGNS + input Hadamard to H)."""
    m_out, k_in = W.shape
    assert k_in % 128 == 0 and m_out % 128 == 0, (W.shape, "need k%128, n%128")
    dev = torch.device(device)
    lut = get_lut(code, L).to(torch.float32)
    if V == 0:                                       # take the code's own V
        V = int(CODES[code][1].get("V", 1)) if code in CODES else 1
    lut_rms = lut.double().square().mean().sqrt().float()

    suh = sign_vec(k_in, su_seed, dev).half()
    svh = sign_vec(m_out, su_seed + 1, dev).half()
    h = hadamard128(dev, torch.float32)

    W32 = W.float().to(dev).t().contiguous()        # [k, n] ref orientation
    if channel_scales:
        su, sv = suh.float(), svh.float()
        r_out = W32.square().mean(dim=0).sqrt()         # per-out-channel RMS [n]
        rm = r_out.mean()
        if rm > 1e-30:
            r_out = r_out / rm
        zero_out = r_out.abs() < 1e-30
        r_out = torch.where(zero_out, torch.full_like(r_out, 0.1), r_out)
        w = W32 / (sv * r_out).unsqueeze(0)             # weight /= sv
        w = had_right(w, h)                             # output Hadamard
        r_in = w.square().mean(dim=1).sqrt()            # per-in-channel RMS [k]
        r_in = torch.where(r_in.abs() < 1e-30,
                           torch.full_like(r_in, 0.1), r_in)
        w = w / (su * r_in).unsqueeze(1)                # weight /= su
        W_i = had_left(w, h)                            # input Hadamard
        svh_store = torch.where(zero_out, torch.zeros_like(r_out), r_out)
    else:
        W_i = to_inner(W32, suh.to(dev), svh.to(dev), h)
    Wscale = W_i.square().mean().sqrt() / lut_rms
    Wq = W_i / Wscale

    Lfac = None
    if use_ldlq and H_x is not None:
        t0 = time.time()
        G = rotate_hessian(H_x.float().to(dev), suh.to(dev), h)
        dead = G.diagonal() == 0
        if dead.any():
            G[dead, dead] = 1.0
        damp = percdamp * G.diagonal().mean()
        G.diagonal().add_(damp)
        try:
            Lfac, _D = block_LDL(G.float().cpu(), 16)
        except RuntimeError:
            # Failure-only path (2026-10-13): successful factorizations are
            # unchanged.  Non-finite G is a real fault -> raise so the job
            # restarts and recomputes; otherwise escalate damping until SPD.
            if not torch.isfinite(G).all():
                raise
            Gd = G.double().cpu()
            for extra in (0.0, 0.1, 1.0):
                if extra:
                    Gd.diagonal().add_(extra * Gd.diagonal().mean())
                try:
                    Lfac, _D = block_LDL(Gd.clone(), 16)
                    break
                except RuntimeError:
                    if extra == 1.0:
                        raise
            print(f"xtq_quant: block_LDL fallback used extra damping {extra} "
                  f"x mean diag", flush=True)
            del Gd
        del G
        Lfac = Lfac.to(dev, Wq.dtype)
        if progress is not None:
            progress("hinv", time.time() - t0)

    hatWq, states = ldlq_viterbi(Wq, Lfac, lut, K, L, buf_cols=buf_cols, V=V,
                                 batch=batch, dtype=dtype, margin=margin,
                                 compiled=compiled, progress=progress,
                                 code=code, backend=backend)

    trellis = pack_states(states.reshape(-1, 256 // V), K, V).view(
        k_in // 16, m_out // 16, 16 * K)
    # bit-exact self-check: unpack(decode) must reproduce the quantized weights
    st_rt = unpack_states(trellis.view(-1, 16 * K), K, V)
    assert torch.equal(st_rt.cpu(), states.reshape(-1, 256 // V).cpu()), \
        "pack/unpack round-trip broke state sequence"
    hatW_d = decode_trellis(trellis, lut.to(dev), K, k_in, m_out, V)
    assert torch.equal(hatW_d, hatWq), "decode != encoder output"
    if channel_scales:
        # Stored EXL3 fields: magnitudes ride in suh/svh; Wscale folds into
        # suh like EXL3's g_scale (quantize.py:1218-1219 su /= g_scale).
        suh = (su * r_in * Wscale).half()
        svh = (sv * svh_store).half()
        # W_hat built in ref.weight_orig order on the STORED fp16 vectors so
        # ref.py reconstructs it bit-exactly (no separate scalar).
        W_hat = from_inner(hatW_d, suh.to(dev), svh.to(dev), h) \
            .t().contiguous()
    else:
        W_hat = from_inner(hatW_d * Wscale, suh.to(dev), svh.to(dev), h) \
            .t().contiguous()

    out = {
        "W_hat": W_hat.cpu(),                        # [out, in] fp32
        "trellis": trellis.cpu(),                    # int16 [k/16, n/16, 16K]
        "suh": suh.cpu(), "svh": svh.cpu(),
        "Wscale": float(Wscale), "lut_rms": float(lut_rms),
        "code": code, "K": K, "L": L, "V": V, "use_ldlq": use_ldlq,
        "channel_scales": channel_scales,
    }
    return out


def hessian_weighted_err(W: torch.Tensor, W_hat: torch.Tensor,
                         H_x: torch.Tensor) -> float:
    """tr(dW H dW^T) / tr(W H W^T) in the original basis, W [out,in]."""
    dW = (W_hat - W).float()
    H = H_x.float()
    num = torch.einsum("ij,jk,ik->", dW, H, dW)
    den = torch.einsum("ij,jk,ik->", W.float(), H, W.float())
    return float(num / den)


def rel_mse(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    d = (W_hat.float() - W.float())
    return float(d.square().sum() / W.float().square().sum())


# ------------------------------------------------------------------ harness API
def quantize(W_transformed: torch.Tensor, U_upper: torch.Tensor | None,
             code: str, K: int, seed: int, L: int = 16,
             dtype: torch.dtype = torch.float32,
             backend: str = "torch", margin: int = 16, batch: int = 0,
             su_mag: torch.Tensor | None = None,
             sv_mag: torch.Tensor | None = None):
    """Harness-facing entry point (signature agreed with Main 2026-10-12).

    W_transformed [out, in] fp32: weight ALREADY in the inner (post-incoherence)
    domain — this function adds only the per-tensor scale and the trellis code.
    in % 16 == 0 and out % 16 == 0 required (no 128-block Hadamard in here).

    U_upper [in, in] fp32 or None: optional GPTQ-convention factor of the
    inner-domain (post-incoherence) input Gram: U^T U = G^{-1}, U upper
    triangular, G the rotated damped Hessian. Internally converted to the
    block-LDL factor the right-to-left sweep consumes: G = U^{-1} U^{-T}
    (one triangular solve + one Cholesky, fp64, once per call). Pass None
    for pure tail-biting Viterbi.

    seed selects the tail-bite pivot (pass-1 boundary position), so identical
    (W, U, code, K, seed) calls are bit-identical and different seeds give
    different — equally valid — bitstreams.

    su_mag, sv_mag: optional fp32 [in]/[out] per-channel magnitudes (EXL3
    regularize semantics): the trellis sees W_transformed/(sv_mag x su_mag),
    the returned W_hat remultiplies them.  To emit EXL3 artifact fields the
    caller writes suh = sign_in*su_mag*scale and svh = sign_out*sv_mag (the
    scalar `scale` folds into suh, cf. quantize.py:1219 su /= g_scale).
    U is NOT rescaled by the mags — EXL3's H rotation is sign-only.

    Returns (W_hat fp32 [out, in], packed_states int16 [in/16, out/16, 16*K],
             scale fp32).  packed -> decode * scale reproduces W_hat bit-exactly.
    """
    m_out, k_in = W_transformed.shape
    assert k_in % 16 == 0 and m_out % 16 == 0, W_transformed.shape
    lut = get_lut(code, L).float()
    V = int(CODES[code][1].get("V", 1)) if code in CODES else 1
    lut_rms = lut.double().square().mean().sqrt().float()

    Wt = W_transformed.float()
    if su_mag is not None or sv_mag is not None:
        su_m = torch.ones(k_in, dtype=Wt.dtype, device=Wt.device) \
            if su_mag is None else su_mag.float().to(Wt.device)
        sv_m = torch.ones(m_out, dtype=Wt.dtype, device=Wt.device) \
            if sv_mag is None else sv_mag.float().to(Wt.device)
        Wt = Wt / (sv_m.unsqueeze(1) * su_m.unsqueeze(0))
    Wq = Wt.t().contiguous()                           # [k=in, n=out]
    scale = Wq.square().mean().sqrt() / lut_rms
    Wq = Wq / scale

    Lfac = None
    if U_upper is not None:
        Uu = U_upper.double()
        eye = torch.eye(Uu.shape[0], dtype=Uu.dtype, device=Uu.device)
        Ui = torch.linalg.solve_triangular(Uu, eye, upper=True)  # U^{-1}
        Lfac, _D = block_LDL(Ui @ Ui.t(), 16)                    # G = U^-1 U^-T
        Lfac = Lfac.float()
    pivot = int(seed) % (256 // V)
    hatWq, states = ldlq_viterbi(Wq, Lfac, lut, K, L, V=V, dtype=dtype,
                                 pivot=pivot, margin=margin, code=code,
                                 backend=backend, batch=batch)

    packed = pack_states(states.reshape(-1, 256 // V), K, V).view(
        k_in // 16, m_out // 16, 16 * K)
    st_rt = unpack_states(packed.view(-1, 16 * K), K, V)
    assert torch.equal(st_rt, states.reshape(-1, 256 // V)), \
        "pack/unpack round-trip broke state sequence"
    W_hat = (hatWq * scale).t().contiguous()             # [out, in]
    hatW_rt = decode_trellis(packed, lut, K, k_in, m_out, V) * scale
    assert torch.equal(hatW_rt.t().contiguous(), W_hat), "decode != encoder"
    if su_mag is not None or sv_mag is not None:
        W_hat = W_hat * (sv_m.unsqueeze(1) * su_m.unsqueeze(0))
    return W_hat, packed, float(scale)


def prepare_incoherence(W: torch.Tensor, H_x: torch.Tensor | None = None,
                        su_seed: int = 20261011,
                        sv_seed: int | None = None,
                        su_sign: torch.Tensor | None = None,
                        sv_sign: torch.Tensor | None = None,
                        channel_scales: bool = True,
                        device: str = "cpu") -> dict:
    """Incoherence transform, split so the harness owns damping + U build.

    Reproduces quantize_linear(channel_scales=...)'s exact transform chain in
    pieces (EXL3 1.4.2 regularize order, quantize.py :1191-1211), because the
    per-channel magnitudes cannot be folded into an already-transformed
    weight: sv_mag applies BEFORE the output Hadamard, su_mag AFTER it and
    BEFORE the input Hadamard.

    W [out, in] fp32 original basis (k_in % 128, m_out % 128 required for the
    block Hadamards).  H_x [in, in] fp32 UNDAMPED Gram or None.

    Signs: caller may pass its own Rademacher vectors via su_sign/sv_sign
    (fp +-1, [in]/[out]); otherwise sign_vec(in, su_seed) and
    sign_vec(out, sv_seed or su_seed+1) — same convention as quantize_linear.

    Returns dict:
      Wt   [out, in] fp32 — trellis-domain weight, magnitudes folded:
             [k,n] w = W/(sv*sv_mag); w = B_n(w); w = w/(su*su_mag); Wt = B_k(w)
           (the per-tensor Wscale is NOT folded: quantize() computes it).
      G    [in, in] fp32 or None — sign-only rotated Gram
             G = B_k diag(su) H diag(su) B_k   (rotate_hessian)
           Magnitudes are deliberately excluded — EXL3 keeps them out of the
           Hessian (finalize_capture_H, quantize.py :879-898).  UNDAMPED:
           caller applies its dead-column + percdamp policy and builds U.
      suh, svh fp32 [in]/[out] — SIGNED magnitudes.  For the EXL3 fields:
             suh_store = (suh * scale).half()   (fold the per-tensor scale into
             suh, cf. quantize.py :1219 su /= g_scale);
             svh_store = svh.half()             (dead out channels already 0).
      W_orig_hat = from_inner(inner_hat_unscaled, suh_store, svh_store)
           == ref.weight_orig(packed, suh_store, svh_store, K, cb) bit-exactly.
      su_mag, sv_mag fp32 — raw positive magnitudes (diagnostics).
      zero_out bool [out] — dead output channels (sv_mag replaced 0.1 in the
           fold, svh stored 0).

    Quantize path:  res = quantize(Wt, U, code, K, seed, backend=..., margin=0)
    with NO su_mag/sv_mag args (folds are already inside Wt); res scale feeds
    suh_store above.
    """
    m_out, k_in = W.shape
    assert k_in % 128 == 0 and m_out % 128 == 0, (W.shape, "need %128 dims")
    dev = torch.device(device)
    h = hadamard128(dev, torch.float32)
    suh_sign = (sign_vec(k_in, su_seed, dev).half() if su_sign is None
                else su_sign.to(dev).half())
    svh_sign = (sign_vec(m_out, (su_seed + 1) if sv_seed is None
                         else sv_seed, dev).half()
                if sv_sign is None else sv_sign.to(dev).half())

    W32 = W.float().to(dev).t().contiguous()            # [k, n]
    if channel_scales:
        su, sv = suh_sign.float(), svh_sign.float()
        sv_mag = W32.square().mean(dim=0).sqrt()        # per-out RMS [n]
        rm = sv_mag.mean()
        if rm > 1e-30:
            sv_mag = sv_mag / rm
        zero_out = sv_mag.abs() < 1e-30
        sv_mag = torch.where(zero_out, torch.full_like(sv_mag, 0.1), sv_mag)
        w = W32 / (sv * sv_mag).unsqueeze(0)            # weight /= sv
        w = had_right(w, h)                             # output Hadamard
        su_mag = w.square().mean(dim=1).sqrt()          # per-in RMS [k]
        su_mag = torch.where(su_mag.abs() < 1e-30,
                             torch.full_like(su_mag, 0.1), su_mag)
        w = w / (su * su_mag).unsqueeze(1)              # weight /= su
        W_i = had_left(w, h)                            # input Hadamard
        suh = su * su_mag
        svh = sv * torch.where(zero_out, torch.zeros_like(sv_mag), sv_mag)
    else:
        W_i = to_inner(W32, suh_sign, svh_sign, h)
        suh, svh = suh_sign.float(), svh_sign.float()
        su_mag = torch.ones(k_in, device=dev)
        sv_mag = torch.ones(m_out, device=dev)
        zero_out = torch.zeros(m_out, dtype=torch.bool, device=dev)

    G = None
    if H_x is not None:
        G = rotate_hessian(H_x.float().to(dev), suh_sign, h).cpu()

    return {"Wt": W_i.t().contiguous().cpu(),            # [out, in] fp32
            "G": G,
            "suh": suh.cpu(), "svh": svh.cpu(),
            "su_mag": su_mag.cpu(), "sv_mag": sv_mag.cpu(),
            "zero_out": zero_out.cpu(),
            "channel_scales": channel_scales}
