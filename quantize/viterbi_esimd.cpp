// viterbi_esimd.cpp — ESIMD tail-biting Viterbi encoder, Xe2/BMG (Arc Pro B70).
//
// Work-group = one 256-weight sequence. Two-phase in-place fp32 cost table in
// the group's global slice; phase-1 mins in registers; one barrier per phase.
// Each work-item handles JPI groups; per group the 16-wide pred gather + min
// and the 16-state emit+scatter are SIMD16 instructions (~8x fewer issue
// slots than scalar SYCL).
//
//   V=1 (mul1):   KV=4, T=256, NC=16, SPRE=4096, ITEMS=512 (JPI=8)
//   V=2 (ih2x2):  KV=8, T=128, NC=256,SPRE=256,  ITEMS=256 (JPI=1)
//
// Build (JIT only; AOT -fsycl-targets=spir64_gen -device bmg miscompiles it):
//   icpx -fsycl -O3 -std=c++17 -DXTQ_VIT_LIB viterbi_esimd.cpp -o viterbi_esimd
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:0 ./viterbi_esimd X.bin nseq code \
//        margin pivot states.bin [iters]
//      ./viterbi_esimd serve     (persistent worker used by quantize.py;
//                                 needs -DXTQ_VIT_LIB)
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

#pragma clang fp contract(off)

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

constexpr uint32_t S = 1u << 16;
constexpr uint32_t MUL1_MULT = 0x83DCD12Du;
constexpr uint32_t INST3_MULT = 89226354u, INST3_ADD = 64248484u;
constexpr uint32_t INST3_KEEP = 0x8FFF8FFFu, INST3_FP = 0x3B603B60u;
constexpr float MUL1_C1 = 0.00676727294921875f, MUL1_C2 = -10.3828125f;
constexpr float FINF = 1e30f;

// fp32 -> fp16(RNE) -> fp32 in integer ops (the JIT folds half round-trips).
static inline float round_f16(float f) {
    uint32_t b = sycl::bit_cast<uint32_t>(f);
    uint32_t ae = b & 0x7FFFFFFFu;
    int e = (int)(ae >> 23) - 127;
    uint32_t m = ae & 0x7FFFFFu;
    int he = e + 15;
    uint32_t h;
    if (he <= 0) {
        if (he < -10) {
            h = 0;
        } else {
            uint32_t mm = m | 0x800000u;
            int sh = 14 - he;
            uint32_t hm = mm >> sh;
            uint32_t rem = mm & ((1u << sh) - 1u);
            uint32_t way = 1u << (sh - 1);
            if (rem > way || (rem == way && (hm & 1u))) ++hm;
            h = hm;
        }
    } else if (he >= 31) {
        h = 0x7C00u;
    } else {
        uint32_t hm = m >> 13;
        uint32_t rem = m & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
            ++hm;
            if (hm == 0x400u) { hm = 0; ++he; }
        }
        h = (he >= 31) ? 0x7C00u : ((uint32_t)he << 10) | hm;
    }
    uint16_t h16 = (uint16_t)(h | ((b >> 16) & 0x8000u));
    return (float)sycl::bit_cast<fp16>(h16);
}
static inline float mul1_lut_entry(uint32_t bs) {
    fp16 hv = sycl::bit_cast<fp16>((uint16_t)(0x6400u + bs));
    return round_f16((float)hv * MUL1_C1 + MUL1_C2);
}


// ---------------------------------------------------------------------------
// V=1 mul1 kernel: ITEMS work-items, JPI groups each.
template <int ITEMS>
struct VitMul1 {
    static constexpr int KV = 4, NC = 16, SPRE = 4096, NSTEPS = 256;
    static constexpr int JPI = SPRE / ITEMS;
    static constexpr int SPI = S / ITEMS;

    const float* X;
    int32_t* states;
    float* cost;
    uint8_t* bp;
    const float* lutf;
    int pivot, margin, early;

    ESIMD_INLINE void emit_blk(uint32_t sbase, float x, float b,
                               simd<float, 16>& e) const {
        simd<uint32_t, 16> sp = sbase + simd<uint32_t, 16>(0, 1);
        simd<uint32_t, 16> h = sp * MUL1_MULT;
        simd<uint32_t, 16> bs =
            dp4a<uint32_t, uint32_t, uint32_t, uint32_t, 16>(
                simd<uint32_t, 16>(0u), h, simd<uint32_t, 16>(0x01010101u));
        simd<uint32_t, 16> fu = gather<uint32_t, 16>(
            reinterpret_cast<const uint32_t*>(lutf), bs << 2u);
        simd<float, 16> f = fu.template bit_cast_view<float>();
        simd<float, 16> d = f - x;
        e = min(d * d, simd<float, 16>(FINF)) + b;   // defeat mad-fusion
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int gid = it.get_group(0);
        int item = it.get_local_id(0);
        const float* xr = X + (size_t)gid * 256;
        float* c = cost + (size_t)gid * S;
        uint8_t* bpbuf = bp + (size_t)gid * NSTEPS * SPRE;
        int32_t* st = states + (size_t)gid * NSTEPS;
        slm_init<ITEMS * 8 + 64>();

        float bregs[JPI];
        int n1 = (margin > 0 && pivot + margin < NSTEPS) ? pivot + margin
                                                       : NSTEPS;
        int overlap;
        // ---------- two passes: pass k=0 free(rolled), k=1 pinned ----------
        for (int pass = 0; pass < 2; ++pass) {
            int rol = pass == 0 ? pivot : 0;
            int ov = pass == 0 ? -1 : overlap;
            int ns = pass == 0 ? n1 : NSTEPS;
            // init step 0
            {
                int u0 = (NSTEPS - rol) % NSTEPS;
                for (int m = 0; m < JPI; ++m) {
                    int jp = item + ITEMS * m;
                    float b = (ov < 0 || jp == ov) ? 0.f : FINF;
                    simd<float, 16> e;
                    emit_blk((uint32_t)(jp * NC), xr[u0], b, e);
                    block_store<float, 16>(c + (size_t)jp * NC, e);
                }
            }
            barrier();
            for (int i = 1; i < ns; ++i) {
                int u = i - rol;
                if (u < 0) u += NSTEPS;
                // phase 1: gather preds, vector-min, store bp byte
                for (int m = 0; m < JPI; ++m) {
                    int j = item + ITEMS * m;
                    simd<uint32_t, 16> off =
                        simd<uint32_t, 16>(0, SPRE * 4);
                    simd<float, 16> v = gather<float, 16>(c + j, off);
                    float bst = hmin<float>(v);
                    simd_mask<16> mk = (v == bst);
                    simd<uint32_t, 16> idx(0, 1);
                    simd<uint32_t, 16> sel = merge(idx, simd<uint32_t, 16>(16u), mk);
                    bregs[m] = bst;
                    bpbuf[(size_t)i * SPRE + j] =
                        (uint8_t)hmin<uint32_t>(sel);
                }
                barrier();
                // phase 2: vector emit + scatter
                bool lastm = (ov >= 0) && (i == ns - 1);
                for (int m = 0; m < JPI; ++m) {
                    int jp = item + ITEMS * m;
                    float b = bregs[m];
                    simd<float, 16> e;
                    emit_blk((uint32_t)(jp * NC), xr[u], b, e);
                    if (lastm) {
                        simd<uint32_t, 16> sp =
                            (uint32_t)(jp * NC) + simd<uint32_t, 16>(0, 1);
                        simd_mask<16> mk = ((sp & (SPRE - 1)) == ov);
                        e = merge(e, simd<float, 16>(FINF), mk);
                    }
                    block_store<float, 16>(c + (size_t)jp * NC, e);
                }
                barrier();
            }
            // argmin + (partial) backtrace by item 0, staged via SLM
            {
                float mv = FINF;
                int mi = 0;
                for (int r = 0; r < SPI; ++r) {
                    int s = item + ITEMS * r;
                    float cc = c[s];
                    if (cc < mv) { mv = cc; mi = s; }
                }
                slm_scalar_store<float>(item * 8, mv);
                slm_scalar_store<int>(item * 8 + 4, mi);
            }
            barrier();
            if (item == 0) {
                float bv = slm_scalar_load<float>(0);
                int bi = slm_scalar_load<int>(4);
                for (int r = 1; r < ITEMS; ++r) {
                    float rv = slm_scalar_load<float>(r * 8);
                    int ri = slm_scalar_load<int>(r * 8 + 4);
                    if (rv < bv || (rv == bv && ri < bi)) { bv = rv; bi = ri; }
                }
                int s = bi;
                int stop = pass == 0 ? pivot : 0;
                for (int i = ns - 1; i > stop; --i) {
                    int j = s >> KV;
                    int w = bpbuf[(size_t)i * SPRE + j];
                    s = j + w * SPRE;
                    if (pass == 1) st[i - 1] = s;
                }
                if (pass == 1) st[NSTEPS - 1] = bi;
                slm_scalar_store<int>(0, s);
            }
            barrier();
            if (pass == 0) overlap = slm_scalar_load<int>(0) >> KV;
        }
    }
};

// ---------------------------------------------------------------------------
// V=1 mul1, transposed cost layout: pos(s) = (s&4095)*16 + (s>>12), so the 16
// preds of group j are a contiguous 64B block_load (phase 1) at the price of
// 16-way strided scatters in phase 2 (one scatter instr per group).
template <int ITEMS>
struct VitMul1T {
    static constexpr int KV = 4, NC = 16, SPRE = 4096, NSTEPS = 256;
    static constexpr int JPI = SPRE / ITEMS;
    static constexpr int SPI = S / ITEMS;

    const float* X;
    int32_t* states;
    float* cost;
    uint8_t* bp;
    const float* lutf;
    int pivot, margin, early;

    // scatter offsets (bytes) for the 16 states of group jp under pos()
    ESIMD_INLINE simd<uint32_t, 16> tpos(uint32_t jp) const {
        // pos(v') = (jp&255)*256 + v'*16 + jp>>8  (in floats)
        return (jp & 255) * 256 + (jp >> 8) + simd<uint32_t, 16>(0, 16);
    }

    ESIMD_INLINE void emit_blk(uint32_t sbase, float x, float b,
                               simd<float, 16>& e) const {
        simd<uint32_t, 16> sp = sbase + simd<uint32_t, 16>(0, 1);
        simd<uint32_t, 16> h = sp * MUL1_MULT;
        simd<uint32_t, 16> bs =
            dp4a<uint32_t, uint32_t, uint32_t, uint32_t, 16>(
                simd<uint32_t, 16>(0u), h, simd<uint32_t, 16>(0x01010101u));
        simd<uint32_t, 16> fu = gather<uint32_t, 16>(
            reinterpret_cast<const uint32_t*>(lutf), bs << 2u);
        simd<float, 16> f = fu.template bit_cast_view<float>();
        simd<float, 16> d = f - x;
        e = min(d * d, simd<float, 16>(FINF)) + b;   // defeat mad-fusion
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int gid = it.get_group(0);
        int item = it.get_local_id(0);
        const float* xr = X + (size_t)gid * 256;
        float* c = cost + (size_t)gid * S;
        uint8_t* bpbuf = bp + (size_t)gid * NSTEPS * SPRE;
        int32_t* st = states + (size_t)gid * NSTEPS;
        slm_init<ITEMS * 8 + 64>();

        float bregs[JPI];
        int n1 = (margin > 0 && pivot + margin < NSTEPS) ? pivot + margin
                                                       : NSTEPS;
        int overlap;
        for (int pass = 0; pass < 2; ++pass) {
            int rol = pass == 0 ? pivot : 0;
            int ov = pass == 0 ? -1 : overlap;
            int ns = pass == 0 ? n1 : NSTEPS;
            {
                int u0 = (NSTEPS - rol) % NSTEPS;
                for (int m = 0; m < JPI; ++m) {
                    int jp = item + ITEMS * m;
                    float b = (ov < 0 || jp == ov) ? 0.f : FINF;
                    simd<float, 16> e;
                    emit_blk((uint32_t)(jp * NC), xr[u0], b, e);
                    scatter<float, 16>(c, tpos(jp) << 2, e);
                }
            }
            barrier();
            for (int i = 1; i < ns; ++i) {
                int u = i - rol;
                if (u < 0) u += NSTEPS;
                for (int m = 0; m < JPI; ++m) {
                    int j = item + ITEMS * m;
                    simd<float, 16> v = block_load<float, 16>(c + j * NC);
                    float bst = hmin<float>(v);
                    simd_mask<16> mk = (v == bst);
                    simd<uint32_t, 16> idx(0, 1);
                    simd<uint32_t, 16> sel = merge(idx, simd<uint32_t, 16>(16u), mk);
                    bregs[m] = bst;
                    bpbuf[(size_t)i * SPRE + j] =
                        (uint8_t)hmin<uint32_t>(sel);
                }
                barrier();
                bool lastm = (ov >= 0) && (i == ns - 1);
                for (int m = 0; m < JPI; ++m) {
                    int jp = item + ITEMS * m;
                    float b = bregs[m];
                    simd<float, 16> e;
                    emit_blk((uint32_t)(jp * NC), xr[u], b, e);
                    if (lastm) {
                        simd<uint32_t, 16> sp =
                            (uint32_t)(jp * NC) + simd<uint32_t, 16>(0, 1);
                        simd_mask<16> mk = ((sp & (SPRE - 1)) == ov);
                        e = merge(e, simd<float, 16>(FINF), mk);
                    }
                    scatter<float, 16>(c, tpos(jp) << 2, e);
                }
                barrier();
            }
            {
                // argmin over transposed table: scan all 65536 entries linearly;
                // linear index q maps to state s = (q&15)*4096 + (q>>4).
                float mv = FINF;
                int mi = 0;
                for (int r = 0; r < SPI; ++r) {
                    int q = item + ITEMS * r;
                    float cc = c[q];
                    if (cc < mv) { mv = cc; mi = q; }
                }
                int s = (mi & 15) * 4096 + (mi >> 4);
                slm_scalar_store<float>(item * 8, mv);
                slm_scalar_store<int>(item * 8 + 4, s);
            }
            barrier();
            if (item == 0) {
                float bv = slm_scalar_load<float>(0);
                int bi = slm_scalar_load<int>(4);
                for (int r = 1; r < ITEMS; ++r) {
                    float rv = slm_scalar_load<float>(r * 8);
                    int ri = slm_scalar_load<int>(r * 8 + 4);
                    if (rv < bv || (rv == bv && ri < bi)) { bv = rv; bi = ri; }
                }
                int s = bi;
                int stop = pass == 0 ? pivot : 0;
                for (int i = ns - 1; i > stop; --i) {
                    int j = s >> KV;
                    int w = bpbuf[(size_t)i * SPRE + j];
                    s = j + w * SPRE;
                    if (pass == 1) st[i - 1] = s;
                }
                if (pass == 1) st[NSTEPS - 1] = bi;
                slm_scalar_store<int>(0, s);
            }
            barrier();
            if (pass == 0) overlap = slm_scalar_load<int>(0) >> KV;
        }
    }
};

// ---------------------------------------------------------------------------
// V=1 mul1, paired variant: item owns CONTIGUOUS groups {item*JPI .. +JPI-1}
// so adjacent group pairs emit as one simd<32> block (halves phase-2 issue).
template <int ITEMS>
struct VitMul1P {
    static constexpr int KV = 4, NC = 16, SPRE = 4096, NSTEPS = 256;
    static constexpr int JPI = SPRE / ITEMS;
    static constexpr int SPI = S / ITEMS;

    const float* X;
    int32_t* states;
    float* cost;
    uint8_t* bp;
    const float* lutf;
    int pivot, margin, early;

    ESIMD_INLINE void emit_blk(uint32_t sbase, float x, float b,
                               simd<float, 16>& e) const {
        simd<uint32_t, 16> sp = sbase + simd<uint32_t, 16>(0, 1);
        simd<uint32_t, 16> h = sp * MUL1_MULT;
        simd<uint32_t, 16> bs =
            dp4a<uint32_t, uint32_t, uint32_t, uint32_t, 16>(
                simd<uint32_t, 16>(0u), h, simd<uint32_t, 16>(0x01010101u));
        simd<uint32_t, 16> fu = gather<uint32_t, 16>(
            reinterpret_cast<const uint32_t*>(lutf), bs << 2u);
        simd<float, 16> f = fu.template bit_cast_view<float>();
        simd<float, 16> d = f - x;
        // min() defeats visa mad-fusion: torch computes fl32(d*d) + b
        // (two roundings); a fused fma rounds once and diverges at ulp level.
        e = min(d * d, simd<float, 16>(FINF)) + b;
    }

    // emit for 32 consecutive states s' = sbase..sbase+31, per-half bests
    ESIMD_INLINE void emit_blk32(uint32_t sbase, float x, float b0, float b1,
                                 simd<float, 32>& e) const {
        simd<uint32_t, 32> sp = sbase + simd<uint32_t, 32>(0, 1);
        simd<uint32_t, 32> h = sp * MUL1_MULT;
        simd<uint32_t, 32> bs =
            dp4a<uint32_t, uint32_t, uint32_t, uint32_t, 32>(
                simd<uint32_t, 32>(0u), h, simd<uint32_t, 32>(0x01010101u));
        simd<uint32_t, 32> fu = gather<uint32_t, 32>(
            reinterpret_cast<const uint32_t*>(lutf), bs << 2u);
        simd<float, 32> f = fu.template bit_cast_view<float>();
        simd<float, 32> d = f - x;
        simd<float, 32> bv;
        bv.template select<16, 1>(0) = simd<float, 16>(b0);
        bv.template select<16, 1>(16) = simd<float, 16>(b1);
        e = min(d * d, simd<float, 32>(FINF)) + bv;  // defeat mad-fusion
    }
    ESIMD_INLINE void emit_blk32m(uint32_t sbase, float x, float b0, float b1,
                                  int ov, simd<float, 32>& e) const {
        emit_blk32(sbase, x, b0, b1, e);
        simd<uint32_t, 32> sp = sbase + simd<uint32_t, 32>(0, 1);
        simd_mask<32> mk = ((sp & (SPRE - 1)) == (uint32_t)ov);
        e = merge(e, simd<float, 32>(FINF), mk);
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int gid = it.get_group(0);
        int item = it.get_local_id(0);
        const float* xr = X + (size_t)gid * 256;
        float* c = cost + (size_t)gid * S;
        uint8_t* bpbuf = bp + (size_t)gid * NSTEPS * SPRE;
        int32_t* st = states + (size_t)gid * NSTEPS;
        slm_init<SPRE * 4>();

        int n1 = (margin > 0 && pivot + margin < NSTEPS) ? pivot + margin
                                                       : NSTEPS;
        int overlap;
        for (int pass = 0; pass < 2; ++pass) {
            int rol = pass == 0 ? pivot : 0;
            int ov = pass == 0 ? -1 : overlap;
            int ns = pass == 0 ? n1 : NSTEPS;
            {
                int u0 = (NSTEPS - rol) % NSTEPS;
                for (int m = 0; m < JPI; ++m) {
                    int jp = item * JPI + m;
                    float b = (ov < 0 || jp == ov) ? 0.f : FINF;
                    simd<float, 16> e;
                    emit_blk((uint32_t)(jp * NC), xr[u0], b, e);
                    block_store<float, 16>(c + (size_t)jp * NC, e);
                }
            }
            barrier();
            for (int i = 1; i < ns; ++i) {
                int u = i - rol;
                if (u < 0) u += NSTEPS;
                for (int m = 0; m < JPI; ++m) {
                    int j = item * JPI + m;
                    simd<uint32_t, 16> off =
                        simd<uint32_t, 16>(0, SPRE * 4);
                    simd<float, 16> v = gather<float, 16>(c + j, off);
                    float bst = hmin<float>(v);
                    simd_mask<16> mk = (v == bst);
                    simd<uint32_t, 16> idx(0, 1);
                    simd<uint32_t, 16> sel = merge(idx, simd<uint32_t, 16>(16u), mk);
                    slm_scalar_store<float>(j * 4, bst);
                    bpbuf[(size_t)i * SPRE + j] =
                        (uint8_t)hmin<uint32_t>(sel);
                }
                barrier();
                bool lastm = (ov >= 0) && (i == ns - 1);
                for (int m = 0; m < JPI; ++m) {
                    int jp = item + ITEMS * m;
                    float b = slm_scalar_load<float>(jp * 4);
                    simd<float, 16> e;
                    emit_blk((uint32_t)(jp * NC), xr[u], b, e);
                    if (lastm) {
                        simd<uint32_t, 16> sp =
                            (uint32_t)(jp * NC) + simd<uint32_t, 16>(0, 1);
                        simd_mask<16> mk = ((sp & (SPRE - 1)) == ov);
                        e = merge(e, simd<float, 16>(FINF), mk);
                    }
                    block_store<float, 16>(c + (size_t)jp * NC, e);
                }
                barrier();
                if (pass == 1 && early >= 0 && i == early) return;
            }
            {
                float mv = FINF;
                int mi = 0;
                for (int r = 0; r < SPI; ++r) {
                    int s = item + ITEMS * r;
                    float cc = c[s];
                    if (cc < mv) { mv = cc; mi = s; }
                }
                slm_scalar_store<float>(item * 8, mv);
                slm_scalar_store<int>(item * 8 + 4, mi);
            }
            barrier();
            if (item == 0) {
                float bv = slm_scalar_load<float>(0);
                int bi = slm_scalar_load<int>(4);
                for (int r = 1; r < ITEMS; ++r) {
                    float rv = slm_scalar_load<float>(r * 8);
                    int ri = slm_scalar_load<int>(r * 8 + 4);
                    if (rv < bv || (rv == bv && ri < bi)) { bv = rv; bi = ri; }
                }
                int s = bi;
                int stop = pass == 0 ? pivot : 0;
                for (int i = ns - 1; i > stop; --i) {
                    int j = s >> KV;
                    int w = bpbuf[(size_t)i * SPRE + j];
                    s = j + w * SPRE;
                    if (pass == 1) st[i - 1] = s;
                }
                if (pass == 1) st[NSTEPS - 1] = bi;
                slm_scalar_store<int>(0, s);
            }
            barrier();
            if (pass == 0) overlap = slm_scalar_load<int>(0) >> KV;
        }
    }
};

// ---------------------------------------------------------------------------
// V=2 ih2x2 kernel: ITEMS=256, JPI=1 (one j per item, 256 preds/256 updates).
template <int ITEMS>
struct VitIh2 {
    static constexpr int KV = 8, NC = 256, SPRE = 256, NSTEPS = 128;
    static constexpr int JPI = SPRE / ITEMS;    // groups per item (4 @ 64)
    static constexpr int SPI = S / ITEMS;

    const float* X;
    int32_t* states;
    float* cost;
    uint8_t* bp;
    const float* lutf;
    int pivot, margin, early;

    // e[v] = (d0^2 + d1^2) + b for 16 states s' = sbase+v
    ESIMD_INLINE void emit_blk(uint32_t sbase, float x0, float x1, float b,
                               simd<float, 16>& e) const {
        simd<uint32_t, 16> sp = sbase + simd<uint32_t, 16>(0, 1);
        simd<uint32_t, 16> xa =
            ((sp * INST3_MULT + INST3_ADD) & INST3_KEEP) ^ INST3_FP;
        simd<uint32_t, 16> xb =
            ((sp * 0x9E3779B1u + 0x85EBCA6Bu) & INST3_KEEP) ^ INST3_FP;
        auto fa = xa.template bit_cast_view<fp16>();
        auto fb = xb.template bit_cast_view<fp16>();
        simd<uint32_t, 16> r;
        r.template bit_cast_view<fp16>() = fa + fb;
        auto fh = r.template bit_cast_view<fp16>();
        simd<fp16, 16> lo = fh.template select<16, 2>(0);
        simd<fp16, 16> hi = fh.template select<16, 2>(1);
        simd<float, 16> d0 = convert<float>(lo) - x0;
        simd<float, 16> d1 = convert<float>(hi) - x1;
        e = min(d0 * d0, simd<float, 16>(FINF)) +
            min(d1 * d1, simd<float, 16>(FINF)) + b;  // defeat mad-fusion
    }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int gid = it.get_group(0);
        int item = it.get_local_id(0);
        const float* xr = X + (size_t)gid * 256;
        float* c = cost + (size_t)gid * S;
        uint8_t* bpbuf = bp + (size_t)gid * NSTEPS * SPRE;
        int32_t* st = states + (size_t)gid * NSTEPS;
        slm_init<ITEMS * 8 + 64>();
        float bregs[JPI];
        int n1 = (margin > 0 && pivot + margin < NSTEPS) ? pivot + margin
                                                       : NSTEPS;
        int overlap;
        for (int pass = 0; pass < 2; ++pass) {
            int rol = pass == 0 ? pivot : 0;
            int ov = pass == 0 ? -1 : overlap;
            int ns = pass == 0 ? n1 : NSTEPS;
            {
                int u0 = (NSTEPS - rol) % NSTEPS;
                for (int m = 0; m < JPI; ++m) {
                    int jp = item + ITEMS * m;
                    float b = (ov < 0 || jp == ov) ? 0.f : FINF;
                    for (int vb = 0; vb < 16; ++vb) {
                        simd<float, 16> e;
                        emit_blk((uint32_t)(jp * NC + vb * 16), xr[2 * u0],
                                 xr[2 * u0 + 1], b, e);
                        block_store<float, 16>(c + (size_t)jp * NC + vb * 16,
                                               e);
                    }
                }
            }
            barrier();
            for (int i = 1; i < ns; ++i) {
                int u = i - rol;
                if (u < 0) u += NSTEPS;
                // phase 1: 256-way gather-min per group (JPI groups/item)
                for (int m = 0; m < JPI; ++m) {
                    int j = item + ITEMS * m;
                    float bst = FINF;
                    int aw = 0;
                    for (int vb = 0; vb < 16; ++vb) {
                        simd<uint32_t, 16> off =
                            (vb * 16 + simd<uint32_t, 16>(0, 1)) * SPRE * 4;
                        simd<float, 16> v = gather<float, 16>(c + j, off);
                        float b2 = hmin<float>(v);
                        simd_mask<16> mk = (v == b2);
                        simd<uint32_t, 16> idx =
                            vb * 16 + simd<uint32_t, 16>(0, 1);
                        simd<uint32_t, 16> sel = merge(idx, simd<uint32_t, 16>(256u), mk);
                        int a2 = hmin<uint32_t>(sel);
                        if (b2 < bst) { bst = b2; aw = a2; }
                    }
                    bregs[m] = bst;
                    bpbuf[(size_t)i * SPRE + j] = (uint8_t)aw;
                }
                barrier();
                bool lastm = (ov >= 0) && (i == ns - 1);
                for (int m = 0; m < JPI; ++m) {
                    int jp = item + ITEMS * m;
                    float b = bregs[m];
                    for (int vb = 0; vb < 16; ++vb) {
                        simd<float, 16> e;
                        emit_blk((uint32_t)(jp * NC + vb * 16), xr[2 * u],
                                 xr[2 * u + 1], b, e);
                        if (lastm) {
                            simd<uint32_t, 16> sp =
                                (uint32_t)(jp * NC + vb * 16) +
                                simd<uint32_t, 16>(0, 1);
                            simd_mask<16> mk = ((sp & (SPRE - 1)) == ov);
                            e = merge(e, simd<float, 16>(FINF), mk);
                        }
                        block_store<float, 16>(c + (size_t)jp * NC + vb * 16, e);
                    }
                }
                barrier();
            }
            {
                float mv = FINF;
                int mi = 0;
                for (int r = 0; r < SPI; ++r) {
                    int s = item + ITEMS * r;
                    float cc = c[s];
                    if (cc < mv) { mv = cc; mi = s; }
                }
                slm_scalar_store<float>(item * 8, mv);
                slm_scalar_store<int>(item * 8 + 4, mi);
            }
            barrier();
            if (item == 0) {
                float bv = slm_scalar_load<float>(0);
                int bi = slm_scalar_load<int>(4);
                for (int r = 1; r < ITEMS; ++r) {
                    float rv = slm_scalar_load<float>(r * 8);
                    int ri = slm_scalar_load<int>(r * 8 + 4);
                    if (rv < bv || (rv == bv && ri < bi)) { bv = rv; bi = ri; }
                }
                int s = bi;
                int stop = pass == 0 ? pivot : 0;
                for (int i = ns - 1; i > stop; --i) {
                    int j = s >> KV;
                    int w = bpbuf[(size_t)i * SPRE + j];
                    s = j + w * SPRE;
                    if (pass == 1) st[i - 1] = s;
                }
                if (pass == 1) st[NSTEPS - 1] = bi;
                slm_scalar_store<int>(0, s);
            }
            barrier();
            if (pass == 0) overlap = slm_scalar_load<int>(0) >> KV;
        }
    }
};

// --------------------------------------------------------------------------
// PCI BDF identity proof for the selected device.
static void print_device_id(const sycl::device& d, const char* tag) {
    fprintf(stderr, "%s: device %s\n", tag,
            d.get_info<sycl::info::device::name>().c_str());
    if (d.has(sycl::aspect::ext_intel_pci_address))
        fprintf(stderr, "%s: pci %s\n", tag,
                d.get_info<sycl::ext::intel::info::device::pci_address>()
                    .c_str());
}

// --------------------------------------------------------------------------
static std::vector<float> read_f32(const char* p, size_t n) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", p); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return v;
}
static void write_i32(const char* p, const int32_t* d, size_t n) {
    FILE* f = fopen(p, "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", p); exit(1); }
    fwrite(d, 4, n, f);
    fclose(f);
}

#ifdef XTQ_VIT_LIB
extern "C" int xtq_vit_init(void);
extern "C" int xtq_vit_run(const float*, int, const char*, int, int,
                           int32_t*);

// Persistent stdin/stdout worker: the backend runs under its own
// ONEAPI_DEVICE_SELECTOR (set on the child's env by the spawner), so it can
// run on a different physical card than the parent process uses.  One
// process lifetime amortizes the JIT compile.  Logs -> stderr; stdout is
// framed replies only.
//   req: {u32 magic=0x58545156, u32 op, i32 margin, i32 pivot, u64 nseq,
//         u64 xbytes, char code[16]} followed by xbytes of fp32 X.
//   op: 0=quit, 1=run.
//   rep: {u32 magic, i32 rc, u64 nbytes} + nbytes of int32 states (rc==0).
#define XTQ_SERVE_MAGIC 0x58545156u
struct XtqReq {
    uint32_t magic, op;
    int32_t margin, pivot;
    uint64_t nseq, xbytes;
    char code[16];
};
struct XtqRep { uint32_t magic; int32_t rc; uint64_t nbytes; };

static int serve_main(void) {
    std::vector<float> xbuf;
    std::vector<int32_t> sbuf;
    for (;;) {
        XtqReq req;
        if (fread(&req, sizeof(req), 1, stdin) != 1) return 0;  // EOF
        req.code[15] = 0;
        if (req.magic != XTQ_SERVE_MAGIC) {
            fprintf(stderr, "xtq_vit serve: bad magic 0x%x\n", req.magic);
            return 1;
        }
        if (req.op == 0) return 0;
        if (req.op != 1 || req.xbytes != req.nseq * 256 * 4 ||
            req.nseq > (1u << 22)) {
            XtqRep rep{XTQ_SERVE_MAGIC, 7, 0};
            fwrite(&rep, sizeof(rep), 1, stdout);
            fflush(stdout);
            continue;
        }
        xbuf.resize((size_t)req.nseq * 256);
        int nsteps = !strcmp(req.code, "ih2x2") ? 128 : 256;
        sbuf.resize((size_t)req.nseq * nsteps);
        if (fread(xbuf.data(), 4, xbuf.size(), stdin) != xbuf.size()) {
            fprintf(stderr, "xtq_vit serve: short X read\n");
            return 1;
        }
        int rc = xtq_vit_run(xbuf.data(), (int)req.nseq, req.code,
                             req.margin, req.pivot, sbuf.data());
        XtqRep rep{XTQ_SERVE_MAGIC, rc, rc ? 0 : (uint64_t)sbuf.size() * 4};
        fwrite(&rep, sizeof(rep), 1, stdout);
        if (!rc) fwrite(sbuf.data(), 4, sbuf.size(), stdout);
        fflush(stdout);
    }
}
#endif  // XTQ_VIT_LIB

int main(int argc, char** argv) {
#ifdef XTQ_VIT_LIB
    if (argc >= 2 && !strcmp(argv[1], "serve")) return serve_main();
#endif
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s X.bin nseq code[mul1|ih2x2] margin pivot "
                "states.bin [bench_iters]\n",
                argv[0]);
        return 1;
    }
    const char* xfile = argv[1];
    int nseq = atoi(argv[2]);
    const char* code = argv[3];
    int margin = atoi(argv[4]);
    int pivot = atoi(argv[5]);
    const char* out = argv[6];
    int iters = argc > 7 ? atoi(argv[7]) : 1;

    std::vector<float> X = read_f32(xfile, (size_t)nseq * 256);
    sycl::queue q(sycl::default_selector_v,
                  sycl::property::queue::in_order{});
    print_device_id(q.get_device(), "device");

    bool v2 = !strcmp(code, "ih2x2");
    bool vt = !strcmp(code, "mul1t") || !strcmp(code, "mul1p");
    int nsteps = v2 ? 128 : 256;
    int spre = v2 ? 256 : 4096;
    int items = v2 ? 256 : 512;
    float* dX = sycl::malloc_device<float>((size_t)nseq * 256, q);
    int32_t* dS = sycl::malloc_device<int32_t>((size_t)nseq * nsteps, q);
    float* dC = sycl::malloc_device<float>((size_t)nseq * S, q);
    uint8_t* dB = sycl::malloc_device<uint8_t>((size_t)nseq * nsteps * spre, q);
    float* dL = sycl::malloc_device<float>(1024, q);
    if (!dX || !dS || !dC || !dB || !dL) { fprintf(stderr, "alloc fail\n"); return 2; }
    q.memcpy(dX, X.data(), (size_t)nseq * 256 * 4).wait();
    // build mul1 emit LUT on host (exact same values)
    {
        std::vector<float> hl(1024);
        for (int bs = 0; bs < 1024; ++bs) hl[bs] = mul1_lut_entry(bs);
        q.memcpy(dL, hl.data(), 4096).wait();
    }

    auto launch = [&]() {
        int items_v1 = getenv("XTQ_ITEMS") ? atoi(getenv("XTQ_ITEMS")) : 64;
        if (!v2 && !vt) {
            if (items_v1 == 64)
                q.submit([&](sycl::handler& h) {
                    VitMul1<64> k{dX, dS, dC, dB, dL, pivot, margin, -1};
                    h.parallel_for(sycl::nd_range<1>(
                        sycl::range<1>((size_t)nseq * 64),
                        sycl::range<1>(64)), k);
                });
            else if (items_v1 == 128)
                q.submit([&](sycl::handler& h) {
                    VitMul1<128> k{dX, dS, dC, dB, dL, pivot, margin, -1};
                    h.parallel_for(sycl::nd_range<1>(
                        sycl::range<1>((size_t)nseq * 128),
                        sycl::range<1>(128)), k);
                });
            else if (items_v1 == 32)
                q.submit([&](sycl::handler& h) {
                    VitMul1<32> k{dX, dS, dC, dB, dL, pivot, margin, -1};
                    h.parallel_for(sycl::nd_range<1>(
                        sycl::range<1>((size_t)nseq * 32),
                        sycl::range<1>(32)), k);
                });
            else
                q.submit([&](sycl::handler& h) {
                    VitMul1<512> k{dX, dS, dC, dB, dL, pivot, margin, -1};
                    h.parallel_for(sycl::nd_range<1>(
                        sycl::range<1>((size_t)nseq * 512),
                        sycl::range<1>(512)), k);
                });
        } else if (!strcmp(code, "mul1p")) {
            int early = getenv("XTQ_EARLY") ? atoi(getenv("XTQ_EARLY")) : -1;
            q.submit([&](sycl::handler& h) {
                VitMul1P<64> k{dX, dS, dC, dB, dL, pivot, margin, early};
                h.parallel_for(sycl::nd_range<1>(
                    sycl::range<1>((size_t)nseq * 64),
                    sycl::range<1>(64)), k);
            });
        } else if (!strcmp(code, "mul1t")) {
            q.submit([&](sycl::handler& h) {
                VitMul1T<64> k{dX, dS, dC, dB, dL, pivot, margin, -1};
                h.parallel_for(sycl::nd_range<1>(
                    sycl::range<1>((size_t)nseq * 64),
                    sycl::range<1>(64)), k);
            });
        } else {
            int it2 = getenv("XTQ_ITEMS") ? atoi(getenv("XTQ_ITEMS")) : 128;
            if (it2 == 256)
                q.submit([&](sycl::handler& h) {
                    VitIh2<256> k{dX, dS, dC, dB, dL, pivot, margin, -1};
                    h.parallel_for(sycl::nd_range<1>(
                        sycl::range<1>((size_t)nseq * 256),
                        sycl::range<1>(256)), k);
                });
            else if (it2 == 64)
                q.submit([&](sycl::handler& h) {
                    VitIh2<64> k{dX, dS, dC, dB, dL, pivot, margin, -1};
                    h.parallel_for(sycl::nd_range<1>(
                        sycl::range<1>((size_t)nseq * 64),
                        sycl::range<1>(64)), k);
                });
            else
                q.submit([&](sycl::handler& h) {
                    VitIh2<128> k{dX, dS, dC, dB, dL, pivot, margin, -1};
                    h.parallel_for(sycl::nd_range<1>(
                        sycl::range<1>((size_t)nseq * 128),
                        sycl::range<1>(128)), k);
                });
        }
    };

    launch();
    q.wait();
    std::vector<int32_t> hostS((size_t)nseq * nsteps);
    q.memcpy(hostS.data(), dS, hostS.size() * 4).wait();
    if (strcmp(out, "-") != 0) write_i32(out, hostS.data(), hostS.size());
    if (const char* cd = getenv("XTQ_DUMP_COST")) {
        std::vector<float> hc((size_t)nseq * S);
        q.memcpy(hc.data(), dC, hc.size() * 4).wait();
        FILE* f = fopen(cd, "wb");
        fwrite(hc.data(), 4, hc.size(), f); fclose(f);
    }
    if (const char* bd = getenv("XTQ_DUMP_BP")) {
        std::vector<uint8_t> hb((size_t)nseq * nsteps * spre);
        q.memcpy(hb.data(), dB, hb.size()).wait();
        FILE* f = fopen(bd, "wb");
        fwrite(hb.data(), 1, hb.size(), f); fclose(f);
    }

    double best_ms = 1e30;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        launch();
        q.wait();
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        if (ms < best_ms) best_ms = ms;
    }
    printf("ESIMD %s nseq=%d margin=%d: %.2f ms -> %.3f s/1M weights\n",
           code, nseq, margin, best_ms,
           best_ms / 1e3 / (nseq * 256) * 1e6);
    sycl::free(dX, q); sycl::free(dS, q); sycl::free(dC, q);
    sycl::free(dB, q); sycl::free(dL, q);
    return 0;
}

// ------------------------------------------------------------------ library
#ifdef XTQ_VIT_LIB
namespace xtqlib {

static sycl::queue* g_q = nullptr;
static float* g_X = nullptr;
static int32_t* g_S = nullptr;
static float* g_C = nullptr;
static uint8_t* g_B = nullptr;
static float* g_L = nullptr;
static size_t g_cap = 0;

static void submit(const char* code, int nseq, int margin, int pivot) {
    bool v2 = !strcmp(code, "ih2x2");
    if (!strcmp(code, "mul1p"))
        g_q->submit([&](sycl::handler& h) {
            VitMul1P<64> k{g_X, g_S, g_C, g_B, g_L, pivot, margin, -1};
            h.parallel_for(sycl::nd_range<1>((size_t)nseq * 64, 64), k);
        });
    else if (!strcmp(code, "mul1t"))
        g_q->submit([&](sycl::handler& h) {
            VitMul1T<64> k{g_X, g_S, g_C, g_B, g_L, pivot, margin, -1};
            h.parallel_for(sycl::nd_range<1>((size_t)nseq * 64, 64), k);
        });
    else if (v2)
        g_q->submit([&](sycl::handler& h) {
            VitIh2<64> k{g_X, g_S, g_C, g_B, g_L, pivot, margin, -1};
            h.parallel_for(sycl::nd_range<1>((size_t)nseq * 64, 64), k);
        });
    else
        g_q->submit([&](sycl::handler& h) {
            VitMul1<64> k{g_X, g_S, g_C, g_B, g_L, pivot, margin, -1};
            h.parallel_for(sycl::nd_range<1>((size_t)nseq * 64, 64), k);
        });
}

}  // namespace xtqlib

// Worst-case device bytes per sequence (mul1: X 1KiB + S 1KiB + C 256KiB +
// B 1MiB).  Hard bound keeps a runaway env under the 8 GiB display-card cap.
static size_t xtq_max_dev_bytes(void) {
    const char* e = getenv("XTQ_VIT_MAXBYTES");
    return e ? strtoull(e, nullptr, 10) : (size_t)6 << 30;
}
static long long xtq_max_nseq(void) {
    const char* e = getenv("XTQ_VIT_MAXSEQ");
    return e ? atoll(e) : 4096;
}

extern "C" int xtq_vit_init(void) {
    using namespace xtqlib;
    if (g_q) return 0;
    try {
        if (const char* e = getenv("XTQ_VIT_DEVIDX")) {
            std::vector<sycl::device> devs;
            for (auto& p : sycl::platform::get_platforms())
                for (auto& d : p.get_devices(sycl::info::device_type::gpu))
                    devs.push_back(d);
            int idx = atoi(e);
            if (idx < 0 || idx >= (int)devs.size()) {
                fprintf(stderr, "xtq_vit: XTQ_VIT_DEVIDX=%d, %zu gpus\n",
                        idx, devs.size());
                return 3;
            }
            g_q = new sycl::queue(devs[idx],
                                  sycl::property::queue::in_order{});
        } else {
            g_q = new sycl::queue(sycl::default_selector_v,
                                  sycl::property::queue::in_order{});
        }
        print_device_id(g_q->get_device(), "xtq_vit");
        fprintf(stderr, "xtq_vit: bounds nseq<=%lld devbytes<=%zu\n",
                xtq_max_nseq(), xtq_max_dev_bytes());
        g_L = sycl::malloc_device<float>(1024, *g_q);
        std::vector<float> hl(1024);
        for (int bs = 0; bs < 1024; ++bs) hl[bs] = mul1_lut_entry(bs);
        g_q->memcpy(g_L, hl.data(), 4096).wait();
        return 0;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "xtq_vit init: %s\n", e.what());
        return 1;
    }
}

extern "C" int xtq_vit_run(const float* X, int nseq, const char* code,
                           int margin, int pivot, int32_t* out) {
    using namespace xtqlib;
    if (!g_q && xtq_vit_init()) return 1;
    int nsteps = !strcmp(code, "ih2x2") ? 128 : 256;
    if (nseq <= 0 || nseq > xtq_max_nseq()) {
        fprintf(stderr, "xtq_vit: nseq %d over XTQ_VIT_MAXSEQ\n", nseq);
        return 6;
    }
    if ((size_t)nseq * 1317888 > xtq_max_dev_bytes()) {
        fprintf(stderr, "xtq_vit: nseq %d over XTQ_VIT_MAXBYTES\n", nseq);
        return 5;
    }
    if ((size_t)nseq > g_cap) {
        if (g_X) {
            sycl::free(g_X, *g_q); sycl::free(g_S, *g_q);
            sycl::free(g_C, *g_q); sycl::free(g_B, *g_q);
        }
        g_cap = (size_t)nseq;
        g_X = sycl::malloc_device<float>(g_cap * 256, *g_q);
        g_S = sycl::malloc_device<int32_t>(g_cap * 256, *g_q);
        g_C = sycl::malloc_device<float>(g_cap * (size_t)S, *g_q);
        g_B = sycl::malloc_device<uint8_t>(g_cap * 256 * 4096, *g_q);
        if (!g_X || !g_S || !g_C || !g_B) { g_cap = 0; return 2; }
    }
    try {
        g_q->memcpy(g_X, X, (size_t)nseq * 256 * 4).wait();
        submit(code, nseq, margin, pivot);
        g_q->wait();
        g_q->memcpy(out, g_S, (size_t)nseq * nsteps * 4).wait();
    } catch (const sycl::exception& e) {
        fprintf(stderr, "xtq_vit run: %s\n", e.what());
        return 4;
    }
    return 0;
}
#endif  // XTQ_VIT_LIB
