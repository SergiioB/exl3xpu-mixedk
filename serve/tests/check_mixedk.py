#!/usr/bin/env python3
"""Mixed-K correctness checks for the patched exl3xpu image on real tensors of
turboderp/Qwen3.8-27B-exl3 branch SC_5.00bpw_H6 (mixed K=4..7).

Run inside the patched image (one GPU visible via ONEAPI_DEVICE_SELECTOR):
  python3 /work/check_mixedk.py --model /model

(a) exl3_reconstruct vs exl3xpu.ref.reconstruct_inner  -> torch.equal expected
(b) exl3xpu_C.linear vs ref.linear_forward (fp32 ref) at M=1,2,4,8,16,64,512 per K,
    plus the W8A8 int8 prefill variant at M=64,512
(c) fused mixed-K qkv_proj (Exl3LinearMethod via the real plugin loaders) vs the
    unfused per-member fp32 reference
"""
import argparse, json, os, sys, time

import torch
from safetensors import safe_open

sys.path.insert(0, "/opt/exl3xpu")
from exl3xpu import ref  # noqa: E402

torch.ops.load_library("/opt/exl3xpu/exl3xpu/_C.so")
E = torch.ops.exl3xpu_C
DEV = "xpu"

# one real SC_5.00bpw_H6 tensor per K (name, K) -- K from the artifact's trellis last dim (16*K)
CASES = [
    ("model.language_model.layers.1.linear_attn.in_proj_qkv", 4),   # [320, 640, 64]  n=10240
    ("model.language_model.layers.31.self_attn.q_proj", 5),         # [320, 768, 80]  n=12288
    ("model.language_model.layers.31.self_attn.o_proj", 6),         # [384, 320, 96]  n=5120
    ("model.language_model.layers.31.self_attn.k_proj", 7),         # [320, 64, 112]  n=1024
]

results = []


def report(check, name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {check} {name} {detail}", flush=True)
    results.append((check, name, ok, detail))


def load_member(idx, model, name):
    f = idx[name + ".trellis"]
    with safe_open(os.path.join(model, f), framework="pt") as sf:
        return (sf.get_tensor(name + ".trellis").to(DEV),
                sf.get_tensor(name + ".suh").to(DEV),
                sf.get_tensor(name + ".svh").to(DEV))


def rel_err(y, yr):
    d = (y.double() - yr.double())
    return (d.norm() / yr.double().norm().clamp_min(1e-30)).item(), d.abs().max().item()


def check_reconstruct(model, idx):
    for name, K in CASES:
        tr, suh, svh = load_member(idx, model, name)
        n = tr.shape[1] * 16
        for n0, n_out in ((0, 512), (n // 2 - 256, 512)):
            w = torch.empty((tr.shape[0] * 16, n_out), dtype=torch.float16, device=DEV)
            t0 = time.time()
            E.exl3_reconstruct(tr, w, n0, K, 2)
            torch.xpu.synchronize()
            dt = time.time() - t0
            wr = ref.reconstruct_inner(tr[:, n0 // 16:(n0 + n_out) // 16], K, 2)
            eq = torch.equal(w, wr)
            ndiff = int((w != wr).sum().item())
            report("reconstruct", f"{name.split('.')[-1]} K={K} n0={n0}", eq,
                   f"torch.equal={eq} ndiff={ndiff} shape={tuple(w.shape)} t={dt*1e3:.1f}ms")


def check_linear(model, idx, int8):
    Ms = [1, 2, 4, 8, 16, 64, 512]
    E.exl3_set_int8(1 if int8 else 0)
    tag = "int8" if int8 else "fp16"
    for name, K in CASES:
        tr, suh, svh = load_member(idx, model, name)
        k, n = tr.shape[0] * 16, tr.shape[1] * 16
        g = torch.Generator(device="cpu").manual_seed(20261012)
        shard = torch.zeros(n // 128, dtype=torch.int32, device=DEV)
        for M in Ms:
            if int8 and M < 64:
                continue                      # int8 path only engages at M>16; cover 64/512
            x = torch.randn((M, k), generator=g).to(DEV).half().contiguous()
            t0 = time.time()
            y = E.linear(x, tr, suh.unsqueeze(0).contiguous(), svh, shard, [0, n],
                         K, 2, 128, 16384)
            torch.xpu.synchronize()
            dt = time.time() - t0
            yr = ref.linear_forward(x.float().cpu(), tr.cpu(), suh.cpu(), svh.cpu(), K, 2)
            rel, mx = rel_err(y.cpu(), yr)
            tol = 0.05 if int8 else 0.02
            report(f"linear-{tag}", f"{name.split('.')[-1]} K={K} M={M}", rel < tol,
                   f"rel2={rel:.2e} maxabs={mx:.3f} tol={tol} t={dt*1e3:.1f}ms")
    E.exl3_set_int8(0)


def check_fused(model, idx):
    """Layer-31 qkv_proj: q K=5, k/v K=7 through the real plugin path."""
    from exl3xpu.vllm_plugin import Exl3Config, Exl3LinearMethod
    cfg = Exl3Config(bits=5.0, head_bits=6, codebook="mul1")
    m = Exl3LinearMethod(cfg, [5, 7, 7])
    layer = torch.nn.Module()
    names = ["q_proj", "k_proj", "v_proj"]
    tens = [load_member(idx, model, f"model.language_model.layers.31.self_attn.{nm}") for nm in names]
    sizes = [t[0].shape[1] * 16 for t in tens]
    k = tens[0][0].shape[0] * 16
    m.create_weights(layer, k, sizes, k, sum(sizes), torch.float16)
    layer = layer.to(DEV)
    for j in range(3):
        layer.trellis.weight_loader(layer.trellis, tens[j][0], j)
        layer.suh.weight_loader(layer.suh, tens[j][1], j)
        layer.svh.weight_loader(layer.svh, tens[j][2], j)
    m.process_weights_after_loading(layer)
    assert layer.exl3_mixed and len(layer.exl3_members) == 3
    assert [mm["K"] for mm in layer.exl3_members] == [5, 7, 7]
    for M in (1, 8):
        g = torch.Generator(device="cpu").manual_seed(777 + M)
        x = torch.randn((M, k), generator=g).to(DEV).half().contiguous()
        y = m.apply(layer, x)
        yr = torch.cat([ref.linear_forward(x.float().cpu(), tens[j][0].cpu(),
                                           tens[j][1].cpu(), tens[j][2].cpu(),
                                           [5, 7, 7][j], 2) for j in range(3)], dim=1)
        rel, mx = rel_err(y.cpu(), yr)
        report("fused-mixed", f"qkv[5,7,7] M={M}", rel < 0.02, f"rel2={rel:.2e} maxabs={mx:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    print(f"device={torch.xpu.get_device_name(0)} build_tag check:")
    ok8 = E.exl3_supported(8, 2)
    print(f"  exl3_supported K2..7 cb2 = {[E.exl3_supported(k, 2) for k in range(2, 8)]}, K8={ok8}")
    check_reconstruct(a.model, idx)
    check_linear(a.model, idx, int8=False)
    check_linear(a.model, idx, int8=True)
    check_fused(a.model, idx)
    bad = [r for r in results if not r[2]]
    print(f"SUMMARY {len(results) - len(bad)}/{len(results)} pass" +
          (f"  FAILURES: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
