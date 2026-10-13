# exl3xpu + mixed-K EXL3 (K=2..7) for Intel Arc Pro B70.
# Base image is 0xSero's exl3xpu 0.1.0, pinned by digest. This only patches
# csrc/exl3_ops.sycl and exl3xpu/vllm_plugin.py, then rebuilds exl3xpu/_C.so.
# Build context: the b70/serve directory.
FROM ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8

LABEL org.opencontainers.image.source="https://github.com/SergiioB/exllamav3" \
      b70.mixedk.tag="mixedk_20261012c" \
      b70.mixedk.base="ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8" \
      b70.mixedk.scope="mul1 cb=2 K in {2,3,4,5,6,7} gemv/dpas/reconstruct/q8-prefill; per-member-K fused apply; odd-K + tiny-N K6 M<=4 GEMV reroute"

COPY patches/ /tmp/b70-patches/

RUN set -e; \
    cd /opt/exl3xpu; \
    for p in /tmp/b70-patches/*.patch; do patch -p1 --forward < "$p"; done; \
    bash scripts/build_ext.sh > /tmp/build_ext.log 2>&1 || { tail -80 /tmp/build_ext.log; exit 1; }; \
    tail -15 /tmp/build_ext.log; \
    test -f exl3xpu/_C.so; \
    strings exl3xpu/_C.so | grep -c mixedk_20261012c >/dev/null; \
    python3 -c "import torch; torch.ops.load_library('/opt/exl3xpu/exl3xpu/_C.so'); \
s=torch.ops.exl3xpu_C.exl3_supported; \
assert all(s(k,2) for k in (2,3,4,5,6,7)), 'mixed-K dispatch missing'; \
assert not s(8,2), 'K=8 must stay unsupported'; \
print('exl3xpu mixedk_20261012c: mul1 K=2..7 supported + odd-K vec reroute')"
