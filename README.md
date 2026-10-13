# exllamav3 on Intel Arc Pro B70 (Xe2 / BMG)

This directory adds two things for the Intel Arc Pro B70, and nothing else:

1. **Serving mixed-bit EXL3 models.** Two patches to [0xSero's exl3xpu](https://github.com/sybil-solutions/exl3xpu)
   vLLM XPU plugin let it serve turboderp EXL3 quants whose tensors use different
   bit widths (K = 2..7 per tensor, mul1 codebook), for example
   `turboderp/Qwen3.8-27B-exl3` branch `SC_5.00bpw_H6`. Stock exl3xpu 0.1.0
   serves only K = 4 and K = 6, and every tensor in a fused module must share the same K.
2. **An EXL3 trellis quantizer that runs on the B70.** A tail-biting Viterbi
   encoder written in ESIMD (`quantize/viterbi_esimd.cpp`), a Python quantizer
   (BlockLDLQ, EXL3 incoherence and per-channel scales) that drives it
   (`quantize/quantize.py`), and a writer that puts quantized tensors into an
   EXL3 model directory (`quantize/export_exl3.py`).

Everything here was built and checked on Linux (`xe` kernel driver) in a host with
two Arc Pro B70 cards. Every check used one card. None of it runs on CUDA, and no
other GPU was tested. Nothing outside `b70/` was changed, apart from a pointer at
the top of the repository README.

```
b70/
  NOTICE                      credits and licenses (turboderp, 0xSero, QTIP)
  serve/Dockerfile            patched image on top of the pinned exl3xpu digest
  serve/patches/0001-exl3-ops-mixed-k.patch      csrc/exl3_ops.sycl
  serve/patches/0002-vllm-plugin-mixed-k.patch   exl3xpu/vllm_plugin.py
  serve/run.sh                serve command used for the checks below
  serve/tests/check_mixedk.py kernel checks against exl3xpu ref.py (inside the image)
  serve/tests/canary.py       6 exact-answer canaries + 3 long-prompt (~8.2K-token) cells
  serve/tests/canary_prompts.json
  serve/tests/multiturn.py    multi-turn prefix-cache-on check
  quantize/viterbi_esimd.cpp  ESIMD tail-biting Viterbi (worker binary / shared library)
  quantize/build.sh           JIT build with icpx
  quantize/verify_viterbi.py  kernel output vs CPU reference, bit for bit
  quantize/quantize.py        quantizer (torch reference backend + ESIMD backend)
  quantize/test_quantize.py   CPU unit tests
  quantize/check_tensor_bitexact.py  real-tensor ESIMD vs torch per-launch check
  quantize/export_exl3.py     EXL3 directory writer + self-tests
```

## What works and how it was checked

All numbers are **self-reported screens** from a single host (evidence tier E2:
self-reported, with raw logs kept privately). Nobody else has reproduced them yet.
Tests ran on one Arc Pro B70 with its power cap set to 230 W, except where noted. Measured draw is
listed only where it was recorded.

### Serving (patched exl3xpu, image built from `serve/Dockerfile`)

Models: `turboderp/Qwen3.8-27B-exl3`
- branch `SC_5.00bpw_H6`, commit `dfe2f4fd71bf9a1994051d4d47e6460fa96dd723` (mixed K = 4..7, called "SC5" below)
- branch `4.00bpw`, commit `113cf7ab958054860e43fb7f3063b1af19171095` (uniform K = 4, called "plain 4.0" below)

Serve config: the image's default Qwen3.8-27B route, with MTP 3-token
speculation, `FULL_DECODE_ONLY` XPU graphs, fp8 KV cache and int8 prefill.
`gpu_memory_utilization=0.90`, `max_model_len=65536`, client concurrency C1.

| Check | Result |
|---|---|
| `check_mixedk.py` on real SC5 tensors (K = 4, 5, 6, 7). It checks reconstruct `torch.equal` against `exl3xpu.ref`, `linear` against an fp32 reference at M = 1..512 for the fp16 and int8 paths, and a fused mixed-K qkv module `[5,7,7]` loaded through the real plugin loaders | **46/46 pass** |
| `canary.py`, SC5, greedy decoding, `max_tokens=512` | **6/6 exact answers**, plus **3/3** long cells (8,161–8,162 prompt tokens) |
| `multiturn.py` with prefix cache on, `turboderp/Qwen3.8-27B-exl3` `SC_4.00bpw_H5` (mixed K = 3..6; other card, 180 W cap, `gpu_memory_utilization=0.85`) | **pass**: facts planted in the shared context were recalled correctly on every later turn |
| `exl3_supported(K, cb=2)` in the image build step | true for K = 2..7. False for K = 8 (`Geo<8>` fails the kernel's `V>=8` static assert, so it is not supported) |

**Odd-K small-M reroute (patch 0001).** Verify calls with M ≤ 4 and odd K, or
K = 6 with N ≤ 2048, now go to the GEMV kernel instead of DPAS. Speed screen:
SC5, MTP3, greedy, prompt 512 / output 128 tokens, C1. Metric: client-side
post-first-token decode tok/s, n = 3 after 1 warmup, median. Prefix-cache state
was not checked for this screen. Draw was not recorded (cap 230 W).

| Image | Decode tok/s (3 samples) | Median |
|---|---|---|
| without the reroute (otherwise identical patch) | 56.4 / 57.9 / 56.7 | 56.7 |
| with the reroute (this release) | 68.4 / 70.3 / 70.3 | 70.3 |

On the same screen, plain 4.0 measured 83.7 / 83.7 / 79.6 on the patched image
and 80.1 / 84.0 / 79.6 on the stock image. This uniform-K run was a check for
regressions only, not a speed claim.

**Mixed-K is slower than plain 4.0, not faster.** Paired A/B test: A = plain 4.0
on the stock `ghcr.io/0xsero/exl3xpu` 0.1.0 digest, B = SC5 on this patched image.
Sampling: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, `ignore_eos`,
thinking off. MTP3 on, C1, output 128 tokens (all samples hit the length limit).
n = 5 per arm, ABBA order over 4 server loads, 1 discarded warmup per arm and cell,
unique prompts (prefix-cache hit delta 0). Metric: completion tokens /
(last − first token time), median. Measured average draw was about 230 W in both
arms (hwmon energy counter). Comparison class: `matched_except_quant`; the image
also differs between arms.

| Cell (actual prompt tokens) | A plain 4.0 | B SC5 | B/A |
|---|---|---|---|
| p524 / g128 | 63.56 | 56.62 | 0.89 |
| p8124 / g128 | 64.34 | 56.82 | 0.88 |

SC5 is a quality-oriented quant. This release makes it servable on a B70; it does
not make it faster than plain 4.0.

### Quantizer (ESIMD Viterbi + `quantize.py`)

| Check | Result |
|---|---|
| `verify_viterbi.py`: kernel states vs the CPU tail-biting Viterbi, 64 Gaussian 256-weight sequences, mul1 code, margins 0 / 8 / 16 | **64/64 sequences bit-exact**, identical MSE, all circular-consistent |
| `check_tensor_bitexact.py` (dual mode: torch and ESIMD Viterbi on the same input at every LDLQ launch). Qwen3.8-27B BF16 layer 2, K = 4, LDLQ, margin 16 | `linear_attn.in_proj_qkv` [10240×5120]: **320 launches, 0 mismatched states**. `mlp.down_proj` [5120×17408]: **1088 launches, 0 mismatched states**. Packed trellis and W_hat `torch.equal` |
| Same, margin 0, `in_proj_qkv` | 320 launches, 0 mismatched states, bit-exact |
| Current `quantize.py` with per-channel scales, margin 0, dual mode | layer 0 `linear_attn.out_proj`: 384 launches, 0 mismatches. Layer 2 `in_proj_qkv` with LDLQ: 320 launches, 0 mismatches |
| `ref.weight_orig(trellis, suh, svh)` from exl3xpu on quantizer output (per-channel scales) | equals the quantizer's W_hat bit for bit (CPU synthetic 256×512; real layer-2 `down_proj` after export) |
| `export_exl3.py --selftest-a`: rebuild a turboderp Qwen3.8-27B EXL3 directory from its own 400 extracted modules | output shards **byte-identical** to the template. `--check-keys` on a real export: all 2,426 template keys present, with the same shape and dtype |
| `test_quantize.py --device cpu --quick` with exl3xpu 0.1.0 `ref.py` (pack/decode vs ref.py for K = 3/4/5, Viterbi vs brute force, tail-biting, `quantize()` API, Gaussian MSE vs the rate-distortion bound, LDLQ ≤ plain Viterbi) | **16/16 pass**. The non-`--quick` Gaussian test (2,048 tiles) needs more than 8 GiB of RAM on CPU |

Speed screen, mul1 kernel only (one card, synthetic input, nseq = 1024, one run
per setting, build from just before the final tie-break fix): **1.06 s per 1M
weights at margin 8, 1.09 s at margin 16**. End-to-end `quantize()` per real
tensor on one card (ESIMD backend, margin 16, including U conversion, LDLQ
feedback and pack/decode checks): 2.45 s per 1M weights (`in_proj_qkv`) and
2.69 s per 1M weights (`down_proj`).

The export self-test (`--selftest-a`) and key check were run and recorded during
development, but their console logs were not kept.

## Quickstart: serving

```bash
# 1. Build the patched image. Base pinned by digest; only _C.so and the plugin change.
docker build -t exl3xpu-b70:mixedk b70/serve

# 2. Serve an EXL3 model directory (mixed or uniform K = 2..7).
MODEL_DIR=/path/to/Qwen3.8-27B-exl3-SC_5.00bpw_H6 GPU_INDEX=0 PORT=8022 \
  bash b70/serve/run.sh
#    Weights are mmapped (~20 GB for SC5); run.sh sets --memory=30g.
#    No speculation: append --set vllm.speculative_config=null

# 3. Canary (from the host; stdlib only). Greedy; needs max_tokens >= 512 because SC5 thinks first.
python3 b70/serve/tests/canary.py --base http://127.0.0.1:8022 \
  --model qwen3.8-27b-exl3 --prompts b70/serve/tests/canary_prompts.json
python3 b70/serve/tests/multiturn.py --base http://127.0.0.1:8022 --model qwen3.8-27b-exl3

# 4. Kernel checks inside the image (one card visible; needs the SC_5.00bpw_H6 files).
docker run --rm --device=/dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -v "$PWD/b70/serve/tests:/work:ro" -v "$MODEL_DIR:/model:ro" \
  --entrypoint python3 exl3xpu-b70:mixedk /work/check_mixedk.py --model /model
```

The patches apply to exl3xpu 0.1.0 with `patch -p1` (or `git apply`) from the
package root (`/opt/exl3xpu` in the image).

## Quickstart: quantizing

Requirements: Intel oneAPI DPC++ (`icpx`; checked with 2026.1.1), PyTorch,
`numpy`, `safetensors`. For ref.py-based checks you also need exl3xpu's `ref.py`
(from the exl3xpu 0.1.0 source tree or `/opt/exl3xpu` in the image).

```bash
cd b70/quantize

# 1. Build the ESIMD worker. JIT only: do NOT add -fsycl-targets=spir64_gen.
bash build.sh                      # -> build/viterbi_esimd, build/libviterbi_esimd.so

# 2. Kernel bit-exactness on one card (no model files needed).
python3 verify_viterbi.py gen 64 1 X64.bin
ONEAPI_DEVICE_SELECTOR=level_zero:0 build/viterbi_esimd X64.bin 64 mul1p 0 128 st64.bin
python3 verify_viterbi.py check X64.bin 64 mul1 0 128 st64.bin   # expect "exact-state seqs 64/64"

# 3. CPU unit tests (ref.py checks need XTQ_REFSRC = dir containing exl3xpu/ref.py).
XTQ_REFSRC=/path/to/exl3xpu-0.1.0 python3 test_quantize.py --device cpu --quick

# 4. One real tensor, ESIMD vs torch at every LDLQ launch (needs >= 20 GiB host RAM for down_proj).
XTQ_BF16=/path/to/Qwen3.8-27B XTQ_HDIR=/path/to/hessians XTQ_VIT_DEVIDX=0 \
  python3 check_tensor_bitexact.py in_proj_qkv
```

Per-tensor API (W in `nn.Linear` layout `[out, in]`, H = input Gram `[in, in]`):

```python
import quantize as xq
r = xq.quantize_linear(W, H, code="exl3_mul1", K=4, use_ldlq=True, device="xpu",
                       backend="esimd", margin=0, channel_scales=True)
# r["trellis"] int16 [in/16, out/16, 16K], r["suh"], r["svh"] fp16, r["W_hat"]
```

To build a model, save each tensor as `ROOT/L{nn}/{module}.safetensors` with
keys `trellis`, `suh` and `svh`, then run
`PYTHONPATH=/path/to/exl3xpu-0.1.0/exl3xpu python3 export_exl3.py --tensors ROOT --template <turboderp EXL3 dir with the same K layout> --out OUT --check-keys`.

**This is a per-tensor tool plus a writer, not a one-command model converter.**
The full-model driver (calibration capture, Hessians, layer loop) is not published.
You need your own calibration pass to produce H for each linear layer.

## Limits

- **B70 / Xe2 only.** ESIMD and SYCL code with no CUDA path. Tested only on Linux
  with the `xe` driver, one card per job, in a two-B70 host. Other Arc GPUs were not tested.
- **The AOT build gave wrong results; use the JIT build.** With
  `-fsycl-targets=spir64_gen -device bmg`, the scalar SYCL variant of the kernel
  matched only 61/64 sequences (64/64 under JIT), and the ESIMD kernel's output was
  also wrong. The ESIMD AOT build was not re-tested after later kernel fixes.
  The worker JIT-compiles once per process.
- **Exact floating-point ties.** The bit-exact results need the `min(d*d, FINF) + b`
  emit form in the kernel. Without it, the ESIMD backend fuses `d*d + b` into one
  rounding, and exact-cost ties then resolve differently from torch.
- **No full-model result yet.** A full Qwen3.8-27B quant made by this quantizer is
  still running, so full-model KL or perplexity has not been measured.
- **Quality vs turboderp's quantizer is not better so far:**
  - With turboderp's default calibration mix, per-tensor Hessian-weighted error on
    layers 0–49 (312 tensors) is **in-sample** (the same calibration Gram the
    quantizer was fit on): median 1.027× turboderp's, range 0.93–1.06×. That is
    about 3 % higher error. Full-model quality is not measured yet.
  - An earlier run with a 32K-token calibration set, scored on a 13-layer prefix,
    had final-logit KL vs BF16 of **1.16×** (wiki) and **1.41×** (held-out chat)
    relative to turboderp plain 4.0. On a held-out Gram, 79 of 87 tensors were
    worse than turboderp's.
- **Serving:** K = 8 and non-mul1 codebooks are not supported for mixed-K serving.
  Mixed-K SC5 decode is about 0.88–0.89× plain 4.0 (see the table above).
  Only Qwen3.8-27B checkpoints were tested.
- All numbers are single-host screens (n = 3 or 5) and have not been independently reproduced.

## Credits

- **turboderp** — ExLlamaV3, the EXL3 format, and the Qwen3.8-27B EXL3 quants (MIT).
- **0xSero** — exl3xpu, the ESIMD EXL3 kernels and vLLM XPU plugin that the serve patches modify (MIT).
- **Cornell-RelaxML QTIP** — the BlockLDLQ and tail-biting Viterbi algorithms, reimplemented here.

See [NOTICE](NOTICE) for copyright lines and license text.
