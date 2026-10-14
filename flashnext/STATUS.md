# FlashNext EXL3 port — STATUS (experimental, unvalidated, blocked)

**Experimental, not a validated serving recipe.** Simple native CLI answer checks passed,
but the declared numerical gate failed. No Flash-Next KL, long-context, FP8 KV,
vision or MTP claim is made. SC5's 196K profile belongs to the separate dense 27B
model and does not apply here. Implementation remains unpublished work in progress;
this page records the observed outcome and blockers.

## What this is
Work on the EXL3 artifact of `Qwen/Qwen3.8-Flash-Next` (`turboderp/Qwen3.8-Flash-Next-exl3`,
branch `2.05bpw_h4_ng4`, commit `65c895314393431c09050b2e04e250836b3a6eb4`) on a dual Intel Arc Pro
B70 host, with a byte-level reference built from the upstream repo revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` (pinned ref decoder sha256
`3f614b7c91510d8520202e85dd0a08255370349a4d5ae4e7601edfa6897d1780`; tokenizer sha256
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`; engine: vendored exllamav3 XPU,
torch 2.14.1+xpu, driver 1.14.37020, card ordinal 1 / UUID `…0f00…`).

## Measured, honestly
* **Qualitative inference works** through the engine CLI (`fn_generate.py`): `The capital of
  France?` -> `Paris`, `Germany?` -> `Berlin`, `17+25` -> `42`, repeat-identical, multiturn
  France -> Germany answers `Paris` then `Berlin`. Numerics are **not** validated by these checks.
* **Head-level port gate FAILS** (gate declared before the numbers: argmax equal, top-5 id sets
  equal, cosine >= 0.99, relative Frobenius <= 0.02). Measured on the frozen panel prefix:
  argmax PASS (`248046` = `248046`), **top-5 sets FAIL**, **cosine 0.98681 FAIL**,
  **rfn 0.16250 FAIL**. A six-position sweep of the same row repeats the same picture. These are
  position-wise observations; they do **not** localize a cause.
* **Aggregate layer statistics (weak)**: reference per-layer `mean/std/absmax` vs identity-mapped
  native module statistics agree closely at layer 0 (ratios 0.99952 / 0.99779) and within ~2-19 %
  at layer 2, while **layer 1 deviates first and strongly (0.65878 / 1.75740)** - the region
  containing the PLE. This is a **weak region hypothesis only**: it compares the same
  artifact's native XPU and FP32 reference paths. Differences from the official BF16
  n-gram rows do not identify the cause, and aggregate statistics are not a parity verdict.

## Negative gates (open, explicit)
1. **Not parity-validated** — the head gate above fails. The README lists experimental
   scope only, with no validated capacity or performance claim.
2. **First divergent module NOT named.** The reference raw per-layer capture completed its full
   forward (`forward_seconds 18247.24`, 48/48 layers, `2026-10-14T10:41:23Z`, inside deadline) and
   then died in `mode_parity` with `NameError: name 'save_file' is not defined` (function-local
   safetensors import vs module-scope writer). Serialization failure of the tool chain, **not** a
   numerical failure of model or port. Raw tensors were memory-only and are lost; no core dump and
   no partial file existed. Element-wise localization therefore cannot be produced from that run.
3. **No KL number.** The BF16 arm ended at its fixed deadline (`SIGINT 11:26:05Z`, `SIGTERM
   11:27:05Z`, gone `11:28:05Z`) at 20/48 layers with zero errors, so no BF16 logits exist. The
   artifact/reference-weight KL is **absent, not estimated**. No GGUF or truncated-layer
   substitute was used.
4. **Capture defect flagged**: native `m_49` and `m_50` (reference layers 46/47) are bitwise
   identical - verify capture reuse before trusting either; the tool flags the pair and continues.
5. **GPU work is blocked** by a foreign `vllm serve` job holding GPU1 since 08:17:36Z; nothing of
   ours runs or is queued, and no foreign process is touched.

## Fail-recovery contract now in place (CPU-validated)
* per-layer checkpoints are written to `raw_layers_partial/layer_<i>.safetensors` **during** the
  forward, so a run cut short keeps every completed layer;
* `fn_kl_eval.py selftest-write` exercises the real writer helpers on a throwaway fixture:
  4/4 checkpoints on disk while the capture loop is still running, consolidated round-trip
  bitwise identical, and a simulated crash before the consolidated write still leaves all layers
  on disk (`logs/save_selftest_report.txt`);
* module-scope `save_file`/`load_file` import (the root cause of gate 2 above) is fixed.

## Local reproduction commands (tooling is not included in this public status-only commit)
```bash
# reference raw per-layer capture (batch 1, exact 20 ids, full 48 layers, per-layer checkpoints)
refenv/bin/python3 tools/fn_kl_eval.py parity --arm exl3 \
  --prompt-ids-file panels/parity20_ids.json --out runs/<run>/parity20
# layer comparison, alignment by enumerated identity only (refuses to guess)
refenv/bin/python3 tools/fn_layer_parity.py \
  --native-map runs/<run>/parity20/native_module_map.json \
  --native runs/<run>/parity20/native_raw_modules.safetensors \
  --reference runs/<run>/parity20/raw_layers.safetensors \
  --native-head runs/engine1-20261014T010425Z/native_first_logits.pt \
  --ref-head runs/<run>/parity20/raw_hidden_logits.safetensors \
  --ref-parity-json runs/<run>/parity20/parity.json --out runs/<run>/parity20/layer_parity.json
# divergence attribution (CPU, identity-mapped; flags duplicate capture tensors)
refenv/bin/python3 tools/fn_divergence_attrib.py \
  --native-map runs/<run>/parity20/native_module_map.json \
  --native runs/<run>/parity20/native_raw_modules.safetensors \
  --reference runs/<run>/parity20/raw_layers.safetensors --out runs/<run>/parity20/attribution.json
# serialization contract test (seconds, no model)
refenv/bin/python3 tools/fn_kl_eval.py selftest-write --out /tmp/fn_parity_write_selftest
```
Surviving data for the 2026-10-14 run: `runs/main-20261014T032613Z/parity20/parity.json` (exact
ids, reference head logits, 48-layer aggregate statistics), `native_module_map.json`,
`native_raw_modules.safetensors`, `arm_deadlines.txt`. Large raws (`logits_exl3.safetensors`,
`ref_row0_pos76.pt`, 3.1 GB each) stay in `/data/nvme/work` and are not synced. A sanitized
evidence bundle (5.2 MB) lives at `/home/sergio/B70-DOCS/evidence/flashnext-kl-20261014/`.

## Hand-off state
No owned process is alive, no VRAM is held, no foreign process was touched, nothing is committed
(this repository's HEAD is unchanged; `flashnext/` is untracked WIP). The port is **blocked**:
head gate failing, first divergence unnamed, no KL. Reaching any of those needs an explicit new
budget and a fresh authority, not a repeat of the lost run.
