# exl3xpu-mixedk

Patches to 0xSero's exl3xpu so that mixed-bit EXL3 models (for example turboderp's
Qwen3.8-27B `SC_5.00bpw_H6`) can be served with vLLM on one Intel Arc Pro B70, plus an
EXL3 trellis quantizer that runs on the B70.

## Built on

- **[exl3xpu](https://github.com/sybil-solutions/exl3xpu)** by 0xSero (MIT): the vLLM XPU
  plugin and SYCL/ESIMD EXL3 kernels. Our image starts from
  `ghcr.io/0xsero/exl3xpu:0.1.0` pinned by digest. Most of the serving work is theirs.
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by turboderp (MIT): the EXL3
  format, and the [Qwen3.8-27B EXL3 quants](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
  including the self-calibrated `SC_*` branches used here.
- **vLLM**, which runs inside the exl3xpu image.
- **Intel's XPU base container**, which the exl3xpu image is built on.
- **[QTIP](https://github.com/Cornell-RelaxML/qtip)** (Cornell-RelaxML): the BlockLDLQ and
  tail-biting Viterbi algorithms, reimplemented in our quantizer.

## What this adds

- `serve/patches/0001-exl3-ops-mixed-k.patch` (`csrc/exl3_ops.sycl`): builds the decode
  and matmul kernels for K = 2..7 instead of only K = 4 and 6, and sends small odd-K
  verify calls to the GEMV kernel, so SC quants decode at a usable speed.
- `serve/patches/0002-vllm-plugin-mixed-k.patch` (`exl3xpu/vllm_plugin.py`): lets one
  fused module (for example qkv) hold tensors with different K, so mixed-K checkpoints load.
- `quantize/`: an ESIMD tail-biting Viterbi encoder and a Python quantizer and writer,
  so EXL3 tensors can be quantized on the B70 itself.

## What it covers

Everything was run on Linux with the `xe` driver, one B70 per job.

| Model / quant | Status |
|---|---|
| `turboderp/Qwen3.8-27B-exl3` `SC_5.00bpw_H6` (K = 4..7) | tested, works (main target) |
| `turboderp/Qwen3.8-27B-exl3` `SC_4.00bpw_H5` (K = 3..6) | tested, works (multi-turn check) |
| `turboderp/Qwen3.8-27B-exl3` `4.00bpw` (uniform K = 4) | tested, no regression vs stock exl3xpu |
| `turboderp/Qwen3.8-27B-exl3` `SC_5.00bpw_H6_V6` | not loadable yet: text weights are the same as `SC_5.00bpw_H6`, but the quantized vision tower keys fail to load |
| Other EXL3 model architectures | untested. exl3xpu ships its own per-model routes |
| Qwen3.8-Flash-Next | not covered. It is a different engine (served as GGUF with llama.cpp), not EXL3 |
| K = 8 or non-mul1 codebooks | not supported (`Geo<8>` fails a kernel static assert) |
| CUDA or other GPUs | not supported. ESIMD/SYCL only, no CUDA path; other Arc GPUs not tested |

## Quick start

With [b70-launcher](https://github.com/SergiioB/b70-launcher) 0.6.2 or later:

```bash
b70 launch qwen38-27b-sc5
```

Or run the image directly (this is `serve/run.sh` with the public image):

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-exl3-SC_5.00bpw_H6
IMAGE=ghcr.io/sergiiob/exl3xpu-mixedk@sha256:933eda54bb51a2e8a6669a70f5a2d7696fabd12a5f8b7c469df02fdfecc79d3a
docker run -d --name exl3xpu-b70 \
  --device=/dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --network=host --memory=30g \
  -v "$MODEL_DIR:/model:ro" \
  --entrypoint python3 "$IMAGE" \
  /opt/exl3xpu/scripts/serve.py /opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw \
  --gpu 0 --model-path /model --port 8022 \
  --set vllm.gpu_memory_utilization=0.90 --set vllm.max_model_len=65536
```

This serves an OpenAI-compatible API on port 8022 (model name `qwen3.8-27b-exl3`).
Weights are mmapped (about 20 GB for SC5). To turn off MTP speculation, add
`--set vllm.speculative_config=null`.

To build the image yourself: `docker build -t exl3xpu-b70:mixedk serve`, then
`MODEL_DIR=... bash serve/run.sh`. The patches also apply to exl3xpu 0.1.0 with
`patch -p1` from the package root (`/opt/exl3xpu` in the image).

### Images

| Tag | Digest | Notes |
|---|---|---|
| `20261014` | `sha256:933eda54bb51a2e8a6669a70f5a2d7696fabd12a5f8b7c469df02fdfecc79d3a` | same layers as `20261013`, labels point to this repo (`serve/Dockerfile.labels`) |
| `20261013` | `sha256:5e2d1fab3a8ba559b4d3af7fb1fa5eeff324c4f4274847e3079ec2d57c698ba4` | built from `serve/Dockerfile` (fork commit 244f57e) |
| `20261012c` | `sha256:7bb0c5cf...` | built from the tree just before that commit, same behavior |

## Results

All numbers are self-reported from one host and have not been reproduced by anyone else.
One B70, power cap 230 W unless noted.

Serve config for every serving check: the image's Qwen3.8-27B route, MTP 3-token
speculation, `FULL_DECODE_ONLY` XPU graphs, fp8 KV cache, int8 prefill,
`gpu_memory_utilization=0.90`, `max_model_len=65536`, one client at a time (C1).
Models: `SC_5.00bpw_H6` at commit `dfe2f4fd` ("SC5") and `4.00bpw` at commit `113cf7ab` ("plain 4.0").

### Checks

| Check | Result |
|---|---|
| `serve/tests/check_mixedk.py` on real SC5 tensors (K = 4..7): reconstruct vs `exl3xpu.ref` (`torch.equal`), `linear` vs fp32 at M = 1..512 for fp16 and int8 paths, fused mixed-K qkv `[5,7,7]` through the real plugin loaders | 46/46 pass |
| `serve/tests/canary.py`, SC5, greedy, `max_tokens=512` | 6/6 exact answers, 3/3 long prompts (8,161 to 8,162 tokens) |
| `serve/tests/multiturn.py`, prefix cache on, `SC_4.00bpw_H5` (other card, 180 W cap, `gpu_memory_utilization=0.85`) | pass: planted facts recalled on every later turn |

### Speed: SC5 is slower than plain 4.0

Paired A/B. A = plain 4.0 on stock `ghcr.io/0xsero/exl3xpu:0.1.0`, B = SC5 on this image.
Temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, `ignore_eos`, thinking off,
MTP3, C1, 128 output tokens. n = 5 per arm in ABBA order over 4 server loads, one discarded
warmup per arm and cell, unique prompts (no prefix-cache hits). Metric: completion tokens
/ (last token time - first token time), median. Average draw about 230 W in both arms.

| Prompt tokens | A plain 4.0 tok/s | B SC5 tok/s | B/A |
|---|---|---|---|
| 524 | 63.56 | 56.62 | 0.89 |
| 8,124 | 64.34 | 56.82 | 0.88 |

The GEMV reroute in patch 0001 matters: on a separate greedy screen (prompt 512,
output 128, n = 3 after one warmup, median), SC5 decoded at 56.7 tok/s without it and
70.3 tok/s with it.

### Quality: SC5 is closer to BF16 than plain 4.0

KL divergence of the next-token probabilities against the BF16 model (same weights,
native BF16 output heads), scored on one B70. Lower is closer to BF16. The chat text was
not used to calibrate either quant.

| Text | Scored tokens | Plain 4.0 KL | SC5 KL | SC5 / plain | Same top token as BF16 (plain / SC5) |
|---|---|---|---|---|---|
| Wikitext prose | 1,664 | 0.01391 | 0.00582 | 0.42x | 94.65% / 96.57% |
| Held-out chat | 1,531 | 0.00876 | 0.00141 | 0.16x | 96.41% / 98.11% |

SC5 gives up 11 to 12% decode speed for lower error vs BF16.

## Our quantizer

The quantizer in `quantize/` works, but it does not beat turboderp's.

- The ESIMD Viterbi kernel matches the CPU reference bit for bit: 64/64 Gaussian
  sequences (`verify_viterbi.py`), and 0 mismatched states over 320 launches
  (`in_proj_qkv`) and 1,088 launches (`down_proj`) on real Qwen3.8-27B layer-2 tensors
  (`check_tensor_bitexact.py`).
- exl3xpu's `ref.weight_orig` on our output equals our W_hat bit for bit, and
  `export_exl3.py --selftest-a` rebuilds a turboderp EXL3 directory byte for byte.
- `test_quantize.py --device cpu --quick`: 16/16 pass.
- A full Qwen3.8-27B quant made with it scored KL vs BF16 of 0.89x turboderp's on
  wikitext and 1.09x on held-out chat. So: slightly better on one set, worse on the other.
- Speed: about 1.1 s per 1M weights for the kernel alone, 2.5 to 2.7 s per 1M weights
  end to end per tensor on one card.

It is a per-tensor tool plus a writer, not a one-command converter. The full-model driver
(calibration capture, Hessians, layer loop) is not published; you need your own
calibration pass to produce H for each linear layer.

```bash
cd quantize
bash build.sh        # JIT build with icpx. Do not add -fsycl-targets=spir64_gen
python3 verify_viterbi.py gen 64 1 X64.bin
ONEAPI_DEVICE_SELECTOR=level_zero:0 build/viterbi_esimd X64.bin 64 mul1p 0 128 st64.bin
python3 verify_viterbi.py check X64.bin 64 mul1 0 128 st64.bin   # expect "exact-state seqs 64/64"
```

```python
import quantize as xq
r = xq.quantize_linear(W, H, code="exl3_mul1", K=4, use_ldlq=True, device="xpu",
                       backend="esimd", margin=0, channel_scales=True)
# r["trellis"] int16 [in/16, out/16, 16K], r["suh"], r["svh"] fp16, r["W_hat"]
```

Save each tensor as `ROOT/L{nn}/{module}.safetensors` (keys `trellis`, `suh`, `svh`) and run
`PYTHONPATH=/path/to/exl3xpu-0.1.0/exl3xpu python3 export_exl3.py --tensors ROOT --template <turboderp EXL3 dir with the same K layout> --out OUT --check-keys`.

Known limits: the AOT build (`-fsycl-targets=spir64_gen -device bmg`) gave wrong results,
so use the JIT build. Bit-exact ties depend on the `min(d*d, FINF) + b` form in the kernel.
Requirements: oneAPI DPC++ `icpx` (checked with 2026.1.1), PyTorch, numpy, safetensors, and
exl3xpu's `ref.py` for the reference checks (`XTQ_REFSRC`).

## Layout

```
NOTICE                          third-party credits and license text
LICENSE                         MIT
serve/Dockerfile                patched image on top of the pinned exl3xpu 0.1.0 digest
serve/Dockerfile.labels         label-only layer used for the 20261014 tag
serve/patches/0001-exl3-ops-mixed-k.patch      csrc/exl3_ops.sycl
serve/patches/0002-vllm-plugin-mixed-k.patch   exl3xpu/vllm_plugin.py
serve/run.sh                    serve command used for the checks
serve/tests/check_mixedk.py     kernel checks against exl3xpu ref.py (run inside the image)
serve/tests/canary.py           6 exact-answer canaries + 3 long-prompt cells
serve/tests/multiturn.py        multi-turn prefix-cache check
quantize/viterbi_esimd.cpp      ESIMD tail-biting Viterbi
quantize/build.sh               JIT build with icpx
quantize/quantize.py            quantizer (torch and ESIMD backends)
quantize/export_exl3.py         EXL3 directory writer + self-tests
quantize/verify_viterbi.py, check_tensor_bitexact.py, test_quantize.py   checks
```

This code used to live in the `b70/` directory of
[SergiioB/exllamav3](https://github.com/SergiioB/exllamav3/tree/b70) (commit 244f57e);
the history moved here.

## Credits and license

exl3xpu is by 0xSero, ExLlamaV3, EXL3 and the SC quants are by turboderp, and the
quantization algorithms come from QTIP. This repository is MIT licensed (`LICENSE`).
Third-party copyright lines and license text are in `NOTICE`.
