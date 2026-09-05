// kernels/cuda/diff_harness.cu — THE GATE. Compares device kernels against
// libapply_ref.so bit for bit. If one bit differs, the backend does not enter
// the harness (spec §04). Run ON the exact droplet/arch of the citable run.
//
// Build (on droplet):
//   nvcc -O2 -fmad=false -arch=native diff_harness.cu refkernels.cu \
//        -o diff_harness -ldl
// Run:
//   ./diff_harness /opt/hytorch/target/release/libapply_ref.so [n_iters]
//
// Case families (spec plan M3): random leaves salted with -0.0, RNE ties at
// bit 16, denormals, NaN/inf leaves, mag at mag_max ± 1ulp, zero delta_hat
// (step-0 dynamics), and dense random.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define cudaError_t hipError_t
  #define cudaSuccess hipSuccess
  #define cudaGetErrorString hipGetErrorString
  #define cudaMalloc hipMalloc
  #define cudaMemcpy hipMemcpy
  #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define cudaDeviceSynchronize hipDeviceSynchronize
#else
  #include <cuda_runtime.h>
#endif

// ---- mirror of the reference FFI types ----
struct FfiBinding { uint32_t pos; uint16_t feature; uint16_t mag_bf16; uint8_t slot; uint8_t _pad[3]; };
struct VerdictRec { uint32_t pos; uint16_t feature; uint16_t mag_bf16; uint16_t cand;
                    uint8_t slot; uint8_t verdict; uint8_t reason; uint8_t _pad[3]; };

typedef int  (*apply_fn)(uint16_t*, uint32_t, uint8_t, uint32_t, const uint16_t*, uint16_t,
                         const FfiBinding*, size_t, uint64_t*);
// v2: adds the selection policy byte (0=global_topk, 1=slot_topk).
typedef int  (*pack_fn)(const uint16_t*, uint32_t, uint8_t, uint32_t, const uint16_t*,
                        uint16_t, uint16_t, float, uint8_t, VerdictRec*, size_t*);

// ---- device kernel decls (refkernels.cu) ----
struct DevBinding { uint32_t pos; uint16_t feature; uint16_t mag_bf16; uint8_t slot; uint8_t _pad[3]; };
struct DevVerdict { uint32_t pos; uint16_t feature; uint16_t mag_bf16; uint16_t cand;
                    uint8_t slot; uint8_t verdict; uint8_t reason; uint8_t _pad[3]; };
extern "C" __global__ void normalize_rows(const uint16_t*, float*, uint32_t, uint32_t, float);
extern "C" __global__ void pack_allocate_tok(const uint16_t*, const float*, DevVerdict*,
                                             uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                             float, uint32_t);
extern "C" __global__ void apply_committed(uint16_t*, const float*, const DevBinding*,
                                           uint32_t, uint32_t, uint32_t);

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(3); } } while (0)

// xorshift for reproducible adversarial cases
static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t xr() { uint64_t x = rng_state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return rng_state = x; }

static uint16_t salted_bf16(int family) {
  uint64_t r = xr();
  switch (family % 6) {
    case 0: return 0x8000;                              // -0.0 (the D1 landmine)
    case 1: return 0x0000;                              // +0.0
    case 2: return (uint16_t)(r & 0x00FF);              // denormals / tiny
    case 3: { uint16_t v = (uint16_t)(r & 0x7FFF);      // finite-ish random
              return (v & 0x7F80) == 0x7F80 ? (uint16_t)(v & ~0x0080) : v; }
    case 4: return (uint16_t)((r & 1) ? 0x7F80 : 0x7FC1); // inf / NaN
    default: return (uint16_t)(r & 0xFFFF);             // fully random
  }
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <libapply_ref.so> [iters]\n", argv[0]); return 2; }
  long iters = argc > 2 ? atol(argv[2]) : 200000;   // cases per family batch

  void* so = dlopen(argv[1], RTLD_NOW);
  if (!so) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
  apply_fn ref_apply = (apply_fn)dlsym(so, "apply_ref_apply");
  pack_fn  ref_pack  = (pack_fn)dlsym(so, "apply_ref_pack_allocate_v2");
  if (!ref_apply || !ref_pack) { fprintf(stderr, "dlsym failed\n"); return 2; }

  // Phase-1-shaped small problem per iteration (dense coverage beats size):
  const uint32_t NT = 8, S = 8, D = 12; const uint16_t NF = 64; const uint32_t K = 4;
  const float MAG_MAX = 8.0f;
  const size_t HL = NT * S * D, CL = (size_t)NF * D;

  uint16_t *h_host = (uint16_t*)malloc(HL * 2), *h_ref = (uint16_t*)malloc(HL * 2);
  uint16_t *cb = (uint16_t*)malloc(CL * 2);
  VerdictRec *v_ref = (VerdictRec*)malloc(NT * K * sizeof(VerdictRec));
  DevVerdict *v_dev_h = (DevVerdict*)malloc(NT * K * sizeof(DevVerdict));

  uint16_t *d_h, *d_cb; float* d_nhat; DevVerdict* d_v; DevBinding* d_b;
  CK(cudaMalloc(&d_h, HL * 2)); CK(cudaMalloc(&d_cb, CL * 2));
  CK(cudaMalloc(&d_nhat, CL * 4)); CK(cudaMalloc(&d_v, NT * K * sizeof(DevVerdict)));
  CK(cudaMalloc(&d_b, NT * K * sizeof(DevBinding)));
  FfiBinding* fb = (FfiBinding*)malloc(NT * K * sizeof(FfiBinding));
  DevBinding* db = (DevBinding*)malloc(NT * K * sizeof(DevBinding));

  long total = 0, pack_mismatch = 0, apply_mismatch = 0;
  const float EPS = 6.103515625e-05f; // 2^-14

  for (long it = 0; it < iters; ++it) {
    int fam = (int)(it % 7);
    // Both selection policies are gated: alternate per iteration.
    uint8_t selection = (uint8_t)(it & 1);   // 0=global_topk, 1=slot_topk
    // Generate case. Family 6 = all-zero delta_hat (step-0 dynamics).
    for (size_t i = 0; i < HL; ++i) h_host[i] = (fam == 6) ? 0 : salted_bf16(fam + (int)(xr() % 3));
    for (size_t i = 0; i < CL; ++i) cb[i] = salted_bf16((fam + 1) % 6 + (int)(xr() % 2));

    // ---- reference: pack+allocate ----
    size_t n_ref = 0;
    int rc = ref_pack(h_host, NT, (uint8_t)S, D, cb, NF, (uint16_t)K, MAG_MAX, selection,
                      v_ref, &n_ref);
    if (rc != 0) { fprintf(stderr, "ref_pack rc=%d\n", rc); return 3; }

    // ---- device: normalize + pack+allocate ----
    CK(cudaMemcpy(d_h, h_host, HL * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cb, cb, CL * 2, cudaMemcpyHostToDevice));
    normalize_rows<<<(NF + 63) / 64, 64>>>(d_cb, d_nhat, NF, D, EPS);
    pack_allocate_tok<<<NT, 128>>>(d_h, d_nhat, d_v, NT, S, D, NF, K, MAG_MAX,
                                   (uint32_t)selection);  // one block/token
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(v_dev_h, d_v, NT * K * sizeof(DevVerdict), cudaMemcpyDeviceToHost));

    if (memcmp(v_ref, v_dev_h, n_ref * sizeof(VerdictRec)) != 0) {
      ++pack_mismatch;
      if (pack_mismatch <= 3) {
        for (size_t i = 0; i < n_ref; ++i) {
          VerdictRec* a = &v_ref[i]; DevVerdict* b = (DevVerdict*)&v_dev_h[i];
          if (memcmp(a, b, sizeof(VerdictRec)) != 0)
            fprintf(stderr, "PACK DIFF it=%ld i=%zu ref(pos=%u f=%u mag=%04x s=%u v=%u r=%u) dev(pos=%u f=%u mag=%04x s=%u v=%u r=%u)\n",
              it, i, a->pos, a->feature, a->mag_bf16, a->slot, a->verdict, a->reason,
              b->pos, b->feature, b->mag_bf16, b->slot, b->verdict, b->reason);
        }
      }
    }

    // ---- apply: commits only, canonical order (already canonical) ----
    size_t n_commit = 0;
    for (size_t i = 0; i < n_ref; ++i)
      if (v_ref[i].verdict == 0) {
        fb[n_commit] = { v_ref[i].pos, v_ref[i].feature, v_ref[i].mag_bf16, v_ref[i].slot, {0,0,0} };
        db[n_commit] = { v_ref[i].pos, v_ref[i].feature, v_ref[i].mag_bf16, v_ref[i].slot, {0,0,0} };
        ++n_commit;
      }

    // Residual to mutate: reuse h_host as the residual too (bit salad is fine).
    memcpy(h_ref, h_host, HL * 2);
    uint64_t written = 0;
    rc = ref_apply(h_ref, NT, (uint8_t)S, D, cb, NF, fb, n_commit, &written);
    if (rc != 0) { fprintf(stderr, "ref_apply rc=%d it=%ld\n", rc, it); return 3; }

    CK(cudaMemcpy(d_h, h_host, HL * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b, db, n_commit * sizeof(DevBinding), cudaMemcpyHostToDevice));
    if (n_commit)
      apply_committed<<<((uint32_t)n_commit + 63) / 64, 64>>>(d_h, d_nhat, d_b, (uint32_t)n_commit, S, D);
    CK(cudaDeviceSynchronize());
    static uint16_t h_dev[8 * 8 * 12];
    CK(cudaMemcpy(h_dev, d_h, HL * 2, cudaMemcpyDeviceToHost));

    if (memcmp(h_ref, h_dev, HL * 2) != 0) {
      ++apply_mismatch;
      if (apply_mismatch <= 3)
        for (size_t i = 0; i < HL; ++i)
          if (h_ref[i] != h_dev[i])
            { fprintf(stderr, "APPLY DIFF it=%ld idx=%zu ref=%04x dev=%04x\n", it, i, h_ref[i], h_dev[i]); break; }
    }
    ++total;
  }

  printf("{\"cases\":%ld,\"pack_mismatch\":%ld,\"apply_mismatch\":%ld,\"verdict\":\"%s\"}\n",
         total, pack_mismatch, apply_mismatch,
         (pack_mismatch + apply_mismatch) == 0 ? "BIT_IDENTICAL" : "BACKEND_REJECTED");
  return (pack_mismatch + apply_mismatch) == 0 ? 0 : 1;
}
