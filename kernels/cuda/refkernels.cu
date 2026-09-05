// kernels/cuda/refkernels.cu — device implementations of the reference policy:
// normalize (Ĉ), pack scores + global top-k, allocate (winner per slot,
// abort rules), and apply. All gated by diff_harness against libapply_ref.so.
//
// Compile: nvcc -O2 -fmad=false -arch=native -cubin refkernels.cu
//
// Bit policy MUST mirror crates/apply-ref exactly:
//   promote: bits << 16 · downcast: RNE bit16, canonical qNaN (sign|0x7FC0)
//   ±0 mag: bit-identical copy BY RULE (D1)
//   reductions: sequential by increasing index, fp32, no FMA
//   selection: IEEE total-order key on score bits; tie → lower feature
//   collision: winner by (mag & 0x7FFF) desc, tie feature asc

#include <cstdint>

// ---------- portable correctly-rounded ops ----------
// NVIDIA: explicit __f*_rn intrinsics (plus -fmad=false).
// AMD/HIP: plain IEEE ops under -ffp-contract=off and
//   -fhip-fp32-correctly-rounded-divide-sqrt (the __fdiv_rn/__fsqrt_rn
//   intrinsics are not guaranteed on CDNA). The differential gate is the
//   arbiter either way: if the bits differ, the backend does not enter.
#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define FADD_RN(a, b) ((a) + (b))
  #define FMUL_RN(a, b) ((a) * (b))
  #define FDIV_RN(a, b) ((a) / (b))
  #define FSQRT_RN(x)   sqrtf(x)
#else
  #define FADD_RN(a, b) __fadd_rn((a), (b))
  #define FMUL_RN(a, b) __fmul_rn((a), (b))
  #define FDIV_RN(a, b) __fdiv_rn((a), (b))
  #define FSQRT_RN(x)   __fsqrt_rn(x)
#endif

extern "C" {

// ---------- shared bit policy ----------

__device__ __forceinline__ float bf16_to_fp32(uint16_t b) {
  return __uint_as_float(((uint32_t)b) << 16);
}

__device__ __forceinline__ uint16_t fp32_to_bf16_rne(float x) {
  uint32_t b = __float_as_uint(x);
  if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u)
    return (uint16_t)(((b >> 16) & 0x8000u) | 0x7FC0u);
  uint32_t lsb = (b >> 16) & 1u;
  return (uint16_t)((b + 0x7FFFu + lsb) >> 16);
}

__device__ __forceinline__ uint32_t total_order_key(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// ---------- normalize: Ĉ rows, one thread per row ----------
// Sequential loop over d_slot inside the thread == reference order.

__global__ void normalize_rows(
    const uint16_t* __restrict__ codebook, // [n_features][d_slot] bf16 bits
    float* __restrict__ nhat,              // [n_features][d_slot] fp32 out
    uint32_t n_features, uint32_t d_slot, float eps)
{
  uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_features) return;
  const uint16_t* row = codebook + (size_t)f * d_slot;
  float acc = 0.0f;
  for (uint32_t j = 0; j < d_slot; ++j) {
    float x = bf16_to_fp32(row[j]);
    acc = FADD_RN(acc, FMUL_RN(x, x));
  }
  float denom = FADD_RN(FSQRT_RN(acc), eps);
  float* out = nhat + (size_t)f * d_slot;
  for (uint32_t j = 0; j < d_slot; ++j)
    out[j] = FDIV_RN(bf16_to_fp32(row[j]), denom);
}

// ---------- pack + allocate: one thread per token ----------
// Phase 1 sizes make this tractable per thread: N_f ≤ 32768 scores of
// d_slot ≤ 16 dims; k ≤ 8. Registers hold the top-k arrays.
//
// Emits VerdictRec-compatible records in the canonical order (slot asc,
// winner first, losers feature asc) at out[pos*k .. pos*k+k).

struct DevVerdict {          // mirrors apply-ref pack::VerdictRec (repr C)
  uint32_t pos;
  uint16_t feature;
  uint16_t mag_bf16;
  uint16_t cand;
  uint8_t  slot;
  uint8_t  verdict;
  uint8_t  reason;
  uint8_t  _pad[3];
};

// selection: 0 = global top-k (phase 1), 1 = slot top-k (phase-2 POLICY:
// winner per slot first, then top-k among slot winners; intra-slot losers
// are SILENCE, OVERFLOW is structurally zero).
#define SELECT_GLOBAL_TOPK 0u
#define SELECT_SLOT_TOPK   1u
#define MAX_SLOTS 64u
#define PACK_THREADS 128u

// Packed (key, feature) comparator: 64-bit value whose DESC order equals the
// reference's (key desc, feature asc). Higher key wins; on key ties, lower
// feature gives a HIGHER packed value (0xFFFFFFFF - f). Deterministic and
// order-independent — parallel argmax over packed values selects EXACTLY the
// reference winner. packed==0 is a safe empty sentinel: key==0 would require
// score bits 0xFFFFFFFF (a NaN), and SPEC_AMEND-002 canonicalizes all NaNs
// to 0x7FC00000 before keying.
__device__ __forceinline__ unsigned long long pack_kf(uint32_t key, uint32_t f) {
  return ((unsigned long long)key << 32) | (unsigned long long)(0xFFFFFFFFu - f);
}
__device__ __forceinline__ uint32_t packed_key(unsigned long long p) {
  return (uint32_t)(p >> 32);
}
__device__ __forceinline__ uint32_t packed_f(unsigned long long p) {
  return 0xFFFFFFFFu - (uint32_t)(p & 0xFFFFFFFFu);
}
// total_order_key is a bijection; invert it to recover score bits from key.
__device__ __forceinline__ uint32_t key_to_bits(uint32_t key) {
  return (key & 0x80000000u) ? (key ^ 0x80000000u) : ~key;
}

// One BLOCK per token, PACK_THREADS threads cooperating over the feature
// sweep (the d12 profile showed the serial per-token sweep dominating wall
// at 26x vanilla). Per-feature dot products keep the reference's sequential
// j-order (unchanged bits); selection uses the packed comparator (identical
// winners); emission runs on thread 0 (identical wire). The gate remains
// the arbiter on every silicon.
__global__ void pack_allocate_tok(
    const uint16_t* __restrict__ delta_hat, // [n_tokens][S][d_slot] bf16
    const float* __restrict__ nhat,         // [n_features][d_slot] fp32
    DevVerdict* __restrict__ out,           // [n_tokens][k]
    uint32_t n_tokens, uint32_t s_slots, uint32_t d_slot,
    uint32_t n_features, uint32_t k, float mag_max, uint32_t selection)
{
  uint32_t pos = blockIdx.x;
  if (pos >= n_tokens) return;
  // Bounds (review 2026-09-02): sh_slot is MAX_SLOTS deep and sel_* hold 8;
  // a manifest with s_slots>64 or k>8 would corrupt shared memory silently.
  // The whole grid returns without writing: n_out stays 0 and the host
  // gate rejects the backend (fail-stop, never garbage facts).
  if (s_slots > MAX_SLOTS || k > 8 || k == 0) return;
  uint32_t tid = threadIdx.x;

  __shared__ unsigned long long sh_cand[PACK_THREADS * 8]; // local top-k dump
  __shared__ unsigned long long sh_slot[MAX_SLOTS];        // slot winners

  uint32_t sel_key[8];
  uint32_t sel_f[8];
  uint32_t sel_score[8];
  uint32_t n_sel = 0;

  const uint16_t* tok = delta_hat + (size_t)pos * s_slots * d_slot;

  if (selection == SELECT_SLOT_TOPK) {
    // Pass 1: winner per slot via atomicMax on packed values (max is
    // commutative — thread order cannot change the result).
    for (uint32_t s = tid; s < s_slots; s += PACK_THREADS) sh_slot[s] = 0ull;
    __syncthreads();
    for (uint32_t f = tid; f < n_features; f += PACK_THREADS) {
      uint32_t s = f % s_slots;
      const float* row = nhat + (size_t)f * d_slot;
      const uint16_t* lf = tok + (size_t)s * d_slot;
      float acc = 0.0f;
      for (uint32_t j = 0; j < d_slot; ++j)
        acc = FADD_RN(acc, FMUL_RN(bf16_to_fp32(lf[j]), row[j]));
      uint32_t acc_bits = __float_as_uint(acc);
      if ((acc_bits & 0x7F800000u) == 0x7F800000u && (acc_bits & 0x007FFFFFu) != 0u)
        acc_bits = 0x7FC00000u;      // SPEC_AMEND-002
      atomicMax(&sh_slot[s], pack_kf(total_order_key(acc_bits), f));
    }
    __syncthreads();
    if (tid == 0) {
      // Pass 2: top-k among slot winners — ascending SLOT scan with strict
      // key comparison (reference tiebreak: earlier slot stays on key ties;
      // this is a slot-order tiebreak, NOT a feature tiebreak, so the scan
      // must stay sequential over slots — 64 iterations on one thread).
      for (uint32_t s = 0; s < s_slots; ++s) {
        unsigned long long p = sh_slot[s];
        if (p == 0ull) continue;
        uint32_t key = packed_key(p), f = packed_f(p);
        uint32_t acc_bits = key_to_bits(key);
        if (n_sel < k) {
          uint32_t ins = n_sel;
          for (uint32_t i = 0; i < n_sel; ++i) { if (key > sel_key[i]) { ins = i; break; } }
          for (uint32_t i = n_sel; i > ins; --i) {
            sel_key[i] = sel_key[i-1]; sel_f[i] = sel_f[i-1]; sel_score[i] = sel_score[i-1];
          }
          sel_key[ins] = key; sel_f[ins] = f; sel_score[ins] = acc_bits;
          ++n_sel;
        } else if (key > sel_key[k-1]) {
          uint32_t ins = k - 1;
          for (uint32_t i = 0; i < k; ++i) { if (key > sel_key[i]) { ins = i; break; } }
          for (uint32_t i = k - 1; i > ins; --i) {
            sel_key[i] = sel_key[i-1]; sel_f[i] = sel_f[i-1]; sel_score[i] = sel_score[i-1];
          }
          sel_key[ins] = key; sel_f[ins] = f; sel_score[ins] = acc_bits;
        }
      }
    }
  } else {
    // Global top-k: each thread keeps a local top-k over its strided
    // features (packed comparator), dumps to shared, thread 0 merges.
    unsigned long long loc[8];
    for (uint32_t i = 0; i < k; ++i) loc[i] = 0ull;
    for (uint32_t f = tid; f < n_features; f += PACK_THREADS) {
      uint32_t s = f % s_slots;
      const float* row = nhat + (size_t)f * d_slot;
      const uint16_t* lf = tok + (size_t)s * d_slot;
      float acc = 0.0f;
      for (uint32_t j = 0; j < d_slot; ++j)
        acc = FADD_RN(acc, FMUL_RN(bf16_to_fp32(lf[j]), row[j]));
      // SPEC_AMEND-002: canonical +qNaN before ordering (x86 emits -qNaN for
      // invalid ops, CUDA emits +qNaN; without this the selection diverges).
      uint32_t acc_bits = __float_as_uint(acc);
      if ((acc_bits & 0x7F800000u) == 0x7F800000u && (acc_bits & 0x007FFFFFu) != 0u)
        acc_bits = 0x7FC00000u;
      unsigned long long p = pack_kf(total_order_key(acc_bits), f);
      if (p > loc[k-1]) {
        uint32_t ins = k - 1;
        for (uint32_t i = 0; i < k; ++i) { if (p > loc[i]) { ins = i; break; } }
        for (uint32_t i = k - 1; i > ins; --i) loc[i] = loc[i-1];
        loc[ins] = p;
      }
    }
    for (uint32_t i = 0; i < k; ++i) sh_cand[tid * k + i] = loc[i];
    __syncthreads();
    if (tid == 0) {
      // Merge PACK_THREADS×k locals: k rounds of argmax over shared (packed
      // desc = key desc, feature asc — exactly the reference order).
      uint32_t total = PACK_THREADS * k;
      for (uint32_t r = 0; r < k; ++r) {
        unsigned long long best = 0ull; uint32_t bi = 0xFFFFFFFFu;
        for (uint32_t i = 0; i < total; ++i) {
          if (sh_cand[i] > best) { best = sh_cand[i]; bi = i; }
        }
        if (bi == 0xFFFFFFFFu) break;   // fewer than k candidates (nf < k)
        sh_cand[bi] = 0ull;
        uint32_t key = packed_key(best), f = packed_f(best);
        sel_key[n_sel] = key; sel_f[n_sel] = f; sel_score[n_sel] = key_to_bits(key);
        ++n_sel;
      }
    }
  }

  if (tid != 0) return;   // emission below is reference-sequential, thread 0

  // Materialize candidates (rank = cand), then canonical emission.
  uint16_t c_mag[8]; uint8_t c_slot[8]; uint16_t c_feat[8]; uint8_t used[8];
  for (uint32_t i = 0; i < n_sel; ++i) {
    c_mag[i]  = fp32_to_bf16_rne(__uint_as_float(sel_score[i]));
    c_slot[i] = (uint8_t)(sel_f[i] % s_slots);
    c_feat[i] = (uint16_t)sel_f[i];
    used[i] = 0;
  }

  DevVerdict* o = out + (size_t)pos * k;
  uint32_t emitted = 0;
  // Slots ascending: pick smallest unused slot each pass.
  while (emitted < n_sel) {
    uint32_t s_min = 256;
    for (uint32_t i = 0; i < n_sel; ++i)
      if (!used[i] && c_slot[i] < s_min) s_min = c_slot[i];
    // Winner in slot: abs-bits desc, tie feature asc.
    uint32_t w = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < n_sel; ++i) {
      if (used[i] || c_slot[i] != s_min) continue;
      if (w == 0xFFFFFFFFu) { w = i; continue; }
      uint16_t ka = (uint16_t)(c_mag[i] & 0x7FFF), kw = (uint16_t)(c_mag[w] & 0x7FFF);
      if (ka > kw || (ka == kw && c_feat[i] < c_feat[w])) w = i;
    }
    // Verdict for the winner.
    uint16_t m = c_mag[w];
    uint8_t verdict = 0, reason = 0;                    // COMMIT
    if ((m & 0x7F80u) == 0x7F80u)          { verdict = 2; reason = 1; }  // nonfinite
    else {
      float av = fabsf(bf16_to_fp32(m));
      if (av > mag_max)                    { verdict = 2; reason = 2; }  // mag_overflow
    }
    o[emitted].pos = pos; o[emitted].feature = c_feat[w]; o[emitted].mag_bf16 = m;
    o[emitted].cand = (uint16_t)w; o[emitted].slot = (uint8_t)s_min;
    o[emitted].verdict = verdict; o[emitted].reason = reason; o[emitted]._pad[0]=0; o[emitted]._pad[1]=0; o[emitted]._pad[2]=0;
    used[w] = 1; ++emitted;
    // Losers of this slot: OVERFLOW, feature asc.
    for (;;) {
      uint32_t l = 0xFFFFFFFFu;
      for (uint32_t i = 0; i < n_sel; ++i) {
        if (used[i] || c_slot[i] != s_min) continue;
        if (l == 0xFFFFFFFFu || c_feat[i] < c_feat[l]) l = i;
      }
      if (l == 0xFFFFFFFFu) break;
      o[emitted].pos = pos; o[emitted].feature = c_feat[l]; o[emitted].mag_bf16 = c_mag[l];
      o[emitted].cand = (uint16_t)l; o[emitted].slot = (uint8_t)s_min;
      o[emitted].verdict = 1; o[emitted].reason = 0; o[emitted]._pad[0]=0; o[emitted]._pad[1]=0; o[emitted]._pad[2]=0;   // OVERFLOW
      used[l] = 1; ++emitted;
    }
  }
}

// ---------- apply: one thread per committed binding ----------
// Writes are disjoint by (pos, slot): one winner per port ⇒ no races and
// the parallel result equals the §04 sequential order.

struct DevBinding {
  uint32_t pos;
  uint16_t feature;
  uint16_t mag_bf16;
  uint8_t  slot;
  uint8_t  _pad[3];
};

__global__ void apply_committed(
    uint16_t* __restrict__ h,
    const float* __restrict__ nhat,
    const DevBinding* __restrict__ bs,
    uint32_t n_bindings, uint32_t s_slots, uint32_t d_slot)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_bindings) return;
  DevBinding b = bs[i];
  if ((b.mag_bf16 & 0x7FFFu) == 0u) return;   // D1: no-op by rule
  float mag = bf16_to_fp32(b.mag_bf16);
  const float* row = nhat + (size_t)b.feature * d_slot;
  uint16_t* leaf = h + ((size_t)b.pos * s_slots + b.slot) * d_slot;
  for (uint32_t j = 0; j < d_slot; ++j) {
    float x = bf16_to_fp32(leaf[j]);
    leaf[j] = fp32_to_bf16_rne(FADD_RN(x, FMUL_RN(mag, row[j])));
  }
}

} // extern "C"
