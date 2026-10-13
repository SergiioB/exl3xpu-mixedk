#!/usr/bin/env python3
"""Assemble a complete EXL3 model directory from trellis-quantized tensors.

Input  (--tensors ROOT): ROOT/L{nn}/{module}.safetensors with keys
    trellis int16 [k/16, n/16, 16*K], suh fp16 [k], svh fp16 [n]
  and ROOT/L{nn}/{module}.json metadata (optional fields: "module" name,
  "K", "Wscale"...).  {module} is the suffix used in template key names,
  e.g. "mlp.down_proj" -> model.language_model.layers.{nn}.mlp.down_proj.*.
Template (--template): turboderp-style EXL3 model dir (index + shards).
Output (--out): complete model dir.  All non-safetensors files are copied
  verbatim; each template safetensors blob for a provided key is replaced
  in place (same header bytes -> byte-identical output when inputs equal
  the template's own tensors); .mul1 scalars keep the template's value
  (the MUL1_MULT codebook constant).  index + quantization_config copied
  verbatim; ARTIFACTS.sha256 regenerated over the output tree.

Proofs: --selftest A writes the template's own tensors back and diffs file
bytes; --selftest B runs ref.weight_orig on a written tensor vs a saved
W_hat; --check-keys verifies every template key exists with same
shape/dtype.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

ST_DTYPES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
             "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
QUANT_FIELDS = ("trellis", "suh", "svh")


def read_st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def tensor_bytes(path, key):
    """Raw bytes of one tensor blob inside a safetensors file."""
    hdr, base = read_st_header(path)
    e = hdr[key]
    lo, hi = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + lo)
        return f.read(hi - lo)


def collect_inputs(tensors_root):
    """-> dict key -> (dtype, shape, raw_bytes)."""
    from safetensors import safe_open
    out = {}
    root = Path(tensors_root)
    for st_path in sorted(root.glob("L*/*.safetensors")):
        ldir = st_path.parent.name
        nn = str(int(ldir[1:]))
        module = st_path.stem
        jpath = st_path.with_suffix(".json")
        meta = {}
        if jpath.exists():
            meta = json.loads(jpath.read_text())
            module = meta.get("module", module)
        prefix = f"model.language_model.layers.{nn}.{module}."
        with safe_open(str(st_path), framework="np") as f:
            have = set(f.keys())
            for field in QUANT_FIELDS:
                if field not in have:
                    raise KeyError(f"{st_path}: missing key {field!r} "
                                   f"(has {sorted(have)})")
                t = f.get_tensor(field)
                np2st = {"int16": "I16", "float16": "F16",
                         "float32": "F32", "int32": "I32",
                         "int64": "I64", "uint8": "U8", "bool": "BOOL"}
                dt = np2st.get(str(t.dtype))
                if dt is None:
                    raise ValueError(f"{st_path}:{field} dtype {t.dtype}")
                out[prefix + field] = (dt, list(t.shape),
                                       t.tobytes(order="C"))
    return out


def export(tensors_root, template, outdir):
    template = Path(template)
    outdir = Path(outdir)
    index = json.loads(
        (template / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    inputs = collect_inputs(tensors_root)
    for k in inputs:
        if k not in weight_map:
            raise KeyError(f"input key {k} not in template index")

    # group replacements by shard file
    per_file = {}
    for k, v in inputs.items():
        per_file.setdefault(weight_map[k], {})[k] = v

    outdir.mkdir(parents=True, exist_ok=True)
    replaced = set()
    for fname in sorted(set(weight_map.values())):
        src = template / fname
        dst = outdir / fname
        hdr, base = read_st_header(src)
        repl = per_file.get(fname, {})
        for k, (dt, shp, raw) in repl.items():
            e = hdr[k]
            if e["dtype"] != dt:
                raise ValueError(f"{k}: dtype {dt} != template {e['dtype']}")
            if e["shape"] != shp:
                raise ValueError(f"{k}: shape {shp} != template {e['shape']}")
            lo, hi = e["data_offsets"]
            if hi - lo != len(raw):
                raise ValueError(f"{k}: nbytes mismatch")
        shutil.copyfile(src, dst)
        if repl:
            with open(dst, "r+b") as f:
                for k, (dt, shp, raw) in repl.items():
                    lo, _hi = hdr[k]["data_offsets"]
                    f.seek(base + lo)
                    f.write(raw)
                    replaced.add(k)
    missing = set(inputs) - replaced
    if missing:
        raise RuntimeError(f"keys not written: {sorted(missing)}")

    # everything else: verbatim copy (configs, tokenizer, index, license...)
    for item in template.iterdir():
        if item.is_file() and item.name not in set(weight_map.values()) \
                and item.name != "ARTIFACTS.sha256":
            shutil.copyfile(item, outdir / item.name)

    # fresh ARTIFACTS.sha256
    lines = []
    for item in sorted(outdir.iterdir()):
        if not item.is_file() or item.name == "ARTIFACTS.sha256":
            continue
        h = hashlib.sha256()
        with open(item, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {item.name}")
    (outdir / "ARTIFACTS.sha256").write_text("\n".join(lines) + "\n")
    return len(replaced)


_ST_DT = {"F16": "float16", "I16": "int16", "F32": "float32",
          "I32": "int32", "BF16": "bfloat16", "I64": "int64",
          "U8": "uint8", "BOOL": "bool"}


def selftest_a(template, workdir, only_keys=None):
    """Extract template tensors into the input layout for a round-trip."""
    from safetensors.torch import save_file
    import torch
    template = Path(template)
    index = json.loads(
        (template / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    root = Path(workdir) / "tensors"
    written = {}
    for key, fname in wm.items():
        if not key.endswith(QUANT_FIELDS):
            continue
        # Only decoder-layer keys map onto the per-tensor input layout
        # L{nn}/{module}.safetensors.  lm_head/mtp.* keys keep the template
        # bytes verbatim (skipped here, never replaced by export()).
        if not key.startswith("model.language_model.layers."):
            continue
        if only_keys is not None and key not in only_keys:
            continue
        rest = key[key.index(".layers.") + 1:].split(".")
        nn, field = rest[1], rest[-1]
        module = ".".join(rest[2:-1])
        ldir = root / f"L{nn}"
        ldir.mkdir(parents=True, exist_ok=True)
        st = ldir / f"{module}.safetensors"
        hdr, _ = read_st_header(template / fname)
        e = hdr[key]
        dt = getattr(torch, _ST_DT[e["dtype"]])
        t = torch.frombuffer(bytearray(tensor_bytes(template / fname, key)),
                             dtype=dt).reshape(e["shape"]).clone()
        written.setdefault(st, {})[field] = t
    for st, d in written.items():
        save_file(d, str(st))
        st.with_suffix(".json").write_text(
            json.dumps({"module": st.stem, "K": 4}))
    return root, len(written)


def selftest_b(template, outdir, what_if):
    """ref.weight_orig(written tensors) == saved W_hat."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import ref as _ref
    from safetensors import safe_open
    index = json.loads(
        (Path(outdir) / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    results = []
    for spec in what_if:
        # spec: "L{nn}/{module}|/path/W_hat.pt" -> compare
        tensor_ref, wpath = spec.split("|")
        ldir, module = tensor_ref.split("/", 1)
        nn = str(int(ldir[1:]))
        base = f"model.language_model.layers.{nn}.{module}"
        with safe_open(str(Path(outdir) / wm[base + ".trellis"]),
                       framework="pt") as f:
            trellis = f.get_tensor(base + ".trellis")
            suh = f.get_tensor(base + ".suh")
            svh = f.get_tensor(base + ".svh")
        w_ref = _ref.weight_orig(trellis, suh, svh, 4, _ref.CB_MUL1)
        W_hat = torch.load(wpath, map_location="cpu", weights_only=False)
        eq = torch.equal(w_ref.t().contiguous(), W_hat.contiguous())
        results.append((tensor_ref, eq,
                        float((w_ref.t() - W_hat).abs().max())))
    return results


def check_keys(template, outdir):
    """Every template key exists in output with same shape/dtype."""
    template, outdir = Path(template), Path(outdir)
    tidx = json.loads(
        (template / "model.safetensors.index.json").read_text())["weight_map"]
    oidx = json.loads(
        (outdir / "model.safetensors.index.json").read_text())["weight_map"]
    bad = []
    for key, fname in tidx.items():
        if key not in oidx:
            bad.append((key, "missing-from-index"))
            continue
        th, _ = read_st_header(template / fname)
        oh, _ = read_st_header(outdir / oidx[key])
        te, oe = th.get(key), oh.get(key)
        if oe is None:
            bad.append((key, "missing-in-shard"))
        elif te["dtype"] != oe["dtype"] or te["shape"] != oe["shape"]:
            bad.append((key, f"{te['dtype']}{te['shape']} vs "
                             f"{oe['dtype']}{oe['shape']}"))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest-a", action="store_true",
                    help="rebuild template from its own tensors; output "
                         "safetensors must be byte-identical")
    ap.add_argument("--selftest-b", nargs="*", default=None,
                    metavar="L{nn}/{module}|W_hat.pt",
                    help="ref.weight_orig == saved W_hat for each spec")
    ap.add_argument("--check-keys", action="store_true")
    ap.add_argument("--tmpdir", default=None,
                    help="scratch dir for selftest-a extraction "
                         "(default: XTQ_TMPDIR or same filesystem as --out)")
    args = ap.parse_args()

    if args.selftest_a:
        import tempfile
        tmp_parent = (args.tmpdir or os.environ.get("XTQ_TMPDIR")
                      or str(Path(args.out).parent))
        Path(tmp_parent).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_parent) as td:
            root, n = selftest_a(args.template, td)
            print(f"selftest-a: extracted {n} module(s) -> {root}")
            export(root, args.template, args.out)
            # byte-compare every shard
            import filecmp
            same = True
            for f in sorted(Path(args.out).glob("*.safetensors")):
                if not filecmp.cmp(f, Path(args.template) / f.name,
                                   shallow=False):
                    print(f"  DIFFERS: {f.name}")
                    same = False
            print(f"selftest-a: shards byte-identical: {same}")
            return
    n = export(args.tensors, args.template, args.out)
    print(f"export: replaced {n} tensors -> {args.out}")
    if args.selftest_b:
        for ref, eq, mx in selftest_b(args.template, args.out,
                                      args.selftest_b):
            print(f"selftest-b {ref}: equal={eq} maxabs={mx:.3e}")
    if args.check_keys:
        bad = check_keys(args.template, args.out)
        print(f"check-keys: {len(bad)} problems")
        for b in bad[:20]:
            print("  ", b)


if __name__ == "__main__":
    main()
