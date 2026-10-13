#!/usr/bin/env bash
# Build the ESIMD Viterbi worker (JIT only) into b70/quantize/build/.
# Requires the Intel oneAPI DPC++ compiler (icpx). ONEAPI_ROOT defaults to /opt/intel/oneapi.
# Do NOT add -fsycl-targets=spir64_gen (-device bmg): the AOT build miscompiles this kernel.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ONEAPI_ROOT=${ONEAPI_ROOT:-/opt/intel/oneapi}
LIBDIR="$ONEAPI_ROOT/compiler/latest/lib"
mkdir -p "$HERE/build"
# Worker binary (standalone kernel runs + `serve` mode used by quantize.py).
icpx -fsycl -O3 -std=c++17 -DXTQ_VIT_LIB "$HERE/viterbi_esimd.cpp" \
  -o "$HERE/build/viterbi_esimd" -Wl,-rpath,"$LIBDIR"
# In-process library for XTQ_VIT_TRANSPORT=ctypes.
icpx -fsycl -O3 -std=c++17 -DXTQ_VIT_LIB -fPIC -shared "$HERE/viterbi_esimd.cpp" \
  -o "$HERE/build/libviterbi_esimd.so" -Wl,-rpath,"$LIBDIR"
echo "built $HERE/build/viterbi_esimd and $HERE/build/libviterbi_esimd.so"
