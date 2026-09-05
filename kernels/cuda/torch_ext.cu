// kernels/cuda/torch_ext.cpp — torch extension bridging the device kernels
// into the trainer. The kernels are the SAME code the differential gate
// authorizes (refkernels.cu); this file only moves tensors.
//
// Exposed ops (all take/return torch tensors on the current CUDA device):
//   normalize_rows(codebook_bits_u16 [NF, D]) -> nhat_f32 [NF, D]
//   pack_allocate(delta_bits_u16 [NT, S, D], nhat [NF, D], k, mag_max)
//       -> verdicts_u8 [NT*k, 16]   (raw VerdictRec bytes, canonical order)
//   apply_committed(h_bits_u16 [NT, S, D] (in-place), nhat, commits_u8 [N,12])
//       -> ()   (commits = raw FfiBinding-on-device bytes)
//
// Build: infra/build-ext.sh on the run droplet (needs torch + nvcc).

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

// Single translation unit: include the gate-authorized kernels directly
// (avoids -rdc=true device linking). DevVerdict/DevBinding come from there.
#include "refkernels.cu"

static_assert(sizeof(DevVerdict) == 16, "DevVerdict must be 16 bytes");
static_assert(sizeof(DevBinding) == 12, "DevBinding must be 12 bytes");

static void check_cuda(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor ext_normalize_rows(torch::Tensor codebook_bits) {
  check_cuda(codebook_bits, "codebook_bits");
  TORCH_CHECK(codebook_bits.dtype() == torch::kUInt16, "codebook_bits must be uint16");
  TORCH_CHECK(codebook_bits.dim() == 2);
  const uint32_t nf = codebook_bits.size(0), d = codebook_bits.size(1);
  auto nhat = torch::empty({(long)nf, (long)d},
      torch::TensorOptions().dtype(torch::kFloat32).device(codebook_bits.device()));
  const float eps = 6.103515625e-05f; // 2^-14
  normalize_rows<<<(nf + 63) / 64, 64>>>(
      reinterpret_cast<const uint16_t*>(codebook_bits.data_ptr()),
      nhat.data_ptr<float>(), nf, d, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return nhat;
}

torch::Tensor ext_pack_allocate(torch::Tensor delta_bits, torch::Tensor nhat,
                                int64_t k, double mag_max, int64_t selection) {
  check_cuda(delta_bits, "delta_bits");
  check_cuda(nhat, "nhat");
  TORCH_CHECK(delta_bits.dtype() == torch::kUInt16);
  TORCH_CHECK(delta_bits.dim() == 3);
  const uint32_t nt = delta_bits.size(0), s = delta_bits.size(1), d = delta_bits.size(2);
  const uint32_t nf = nhat.size(0);
  TORCH_CHECK((uint32_t)nhat.size(1) == d);
  TORCH_CHECK(k >= 1 && k <= 8);
  TORCH_CHECK(selection == 0 || selection == 1, "selection: 0=global, 1=slot");
  TORCH_CHECK(selection == 0 || (uint32_t)k <= s, "slot_topk: k <= S");
  TORCH_CHECK(s <= 64 && k >= 1 && k <= 8, "device kernel bounds: s_slots<=64, 1<=k<=8 (shared-memory layout)");
  auto out = torch::zeros({(long)(nt * k), 16},
      torch::TensorOptions().dtype(torch::kUInt8).device(delta_bits.device()));
  pack_allocate_tok<<<nt, 128>>>(   // one block per token, 128 cooperating threads
      reinterpret_cast<const uint16_t*>(delta_bits.data_ptr()),
      nhat.data_ptr<float>(),
      reinterpret_cast<DevVerdict*>(out.data_ptr()),
      nt, s, d, nf, (uint32_t)k, (float)mag_max, (uint32_t)selection);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void ext_apply_committed(torch::Tensor h_bits, torch::Tensor nhat, torch::Tensor commits) {
  check_cuda(h_bits, "h_bits");
  check_cuda(nhat, "nhat");
  check_cuda(commits, "commits");
  TORCH_CHECK(h_bits.dtype() == torch::kUInt16);
  TORCH_CHECK(commits.dtype() == torch::kUInt8);
  TORCH_CHECK(commits.dim() == 2 && commits.size(1) == 12,
              "commits must be [N, 12] raw DevBinding bytes");
  const uint32_t nt = h_bits.size(0), s = h_bits.size(1), d = h_bits.size(2);
  const uint32_t n = commits.size(0);
  if (n == 0) return;
  apply_committed<<<(n + 63) / 64, 64>>>(
      reinterpret_cast<uint16_t*>(h_bits.data_ptr()),
      nhat.data_ptr<float>(),
      reinterpret_cast<const DevBinding*>(commits.data_ptr()),
      n, s, d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("normalize_rows", &ext_normalize_rows, "device Ĉ (gate-authorized)");
  m.def("pack_allocate", &ext_pack_allocate, "device pack+allocate (gate-authorized)");
  m.def("apply_committed", &ext_apply_committed, "device apply, in-place (gate-authorized)");
}
