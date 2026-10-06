#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>

std::vector<torch::Tensor> cub_topk_bf16_512x11_cuda(torch::Tensor scores, int64_t k);

torch::Tensor selected_up_silu_bf16_cuda(
    torch::Tensor x,
    torch::Tensor topk_vals,
    torch::Tensor idx,
    torch::Tensor w_up);

torch::Tensor selected_down_bf16_h32_k16_cuda(
    torch::Tensor sparse_z,
    torch::Tensor idx,
    torch::Tensor w_down_t);

void optimized_global_after_gate_bf16_out_cuda(
    torch::Tensor x,
    torch::Tensor gate_scores,
    torch::Tensor up_weight,
    torch::Tensor down_weight_t,
    torch::Tensor idx,
    torch::Tensor sparse_z,
    torch::Tensor out,
    int64_t k);

std::vector<torch::Tensor> cub_topk_bf16_512x11(torch::Tensor scores, int64_t k) {
  TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
  TORCH_CHECK(scores.dim() == 2, "scores must be [B, I]");
  TORCH_CHECK(scores.scalar_type() == torch::kBFloat16, "scores must be BF16");
  TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
  TORCH_CHECK(k > 0 && k <= scores.size(1), "invalid k");
  TORCH_CHECK(scores.size(1) <= 5632, "512x11 CUB path supports I <= 5632");
  return cub_topk_bf16_512x11_cuda(scores, k);
}

torch::Tensor selected_up_silu_bf16(
    torch::Tensor x,
    torch::Tensor topk_vals,
    torch::Tensor idx,
    torch::Tensor w_up) {
  TORCH_CHECK(x.is_cuda() && topk_vals.is_cuda() && idx.is_cuda() && w_up.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
  TORCH_CHECK(topk_vals.scalar_type() == torch::kBFloat16, "topk_vals must be BF16");
  TORCH_CHECK(w_up.scalar_type() == torch::kBFloat16, "w_up must be BF16");
  TORCH_CHECK(idx.scalar_type() == torch::kInt64, "idx must be int64");
  TORCH_CHECK(x.is_contiguous() && topk_vals.is_contiguous() && idx.is_contiguous() && w_up.is_contiguous(), "inputs must be contiguous");
  return selected_up_silu_bf16_cuda(x, topk_vals, idx, w_up);
}

torch::Tensor selected_down_bf16_h32_k16(
    torch::Tensor sparse_z,
    torch::Tensor idx,
    torch::Tensor w_down_t) {
  TORCH_CHECK(sparse_z.is_cuda() && idx.is_cuda() && w_down_t.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(sparse_z.scalar_type() == torch::kBFloat16, "sparse_z must be BF16");
  TORCH_CHECK(w_down_t.scalar_type() == torch::kBFloat16, "w_down_t must be BF16");
  TORCH_CHECK(idx.scalar_type() == torch::kInt64, "idx must be int64");
  TORCH_CHECK(sparse_z.is_contiguous() && idx.is_contiguous() && w_down_t.is_contiguous(), "inputs must be contiguous");
  return selected_down_bf16_h32_k16_cuda(sparse_z, idx, w_down_t);
}

void optimized_global_after_gate_bf16_out(
    torch::Tensor x,
    torch::Tensor gate_scores,
    torch::Tensor up_weight,
    torch::Tensor down_weight_t,
    torch::Tensor idx,
    torch::Tensor sparse_z,
    torch::Tensor out,
    int64_t k) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  for (const auto& tensor : {x, gate_scores, up_weight, down_weight_t, sparse_z, out}) {
    TORCH_CHECK(tensor.device() == x.device(), "all tensors must be on the same device");
    TORCH_CHECK(tensor.dim() == 2 && tensor.is_contiguous(), "tensors must be contiguous matrices");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, "tensors must be BF16");
  }
  TORCH_CHECK(idx.device() == x.device() && idx.scalar_type() == torch::kInt32 &&
              idx.dim() == 2 && idx.is_contiguous(), "idx must be a contiguous int32 matrix on the same device");
  TORCH_CHECK(x.size(1) > 0, "hidden size must be positive");
  TORCH_CHECK(gate_scores.size(0) == x.size(0), "gate batch size mismatch");
  TORCH_CHECK(up_weight.size(0) == gate_scores.size(1) && up_weight.size(1) == x.size(1), "up weight shape mismatch");
  TORCH_CHECK(down_weight_t.sizes() == up_weight.sizes(), "down weight shape mismatch");
  TORCH_CHECK(idx.size(0) == x.size(0) && idx.size(1) == k, "idx shape mismatch");
  TORCH_CHECK(sparse_z.sizes() == idx.sizes() && out.sizes() == x.sizes(), "output shape mismatch");
  TORCH_CHECK(k > 0 && k <= gate_scores.size(1), "invalid k");
  TORCH_CHECK(gate_scores.size(1) <= 5632, "after-gate CUB path supports I <= 5632");
  if (x.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  optimized_global_after_gate_bf16_out_cuda(x, gate_scores, up_weight, down_weight_t, idx, sparse_z, out, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "cub_topk_bf16_512x11",
      &cub_topk_bf16_512x11,
      "Fixed-shape BF16 global Top-K via CUB BlockRadixSort, 512 threads x 11 items");
  m.def("selected_up_silu_bf16", &selected_up_silu_bf16, "Warp-level selected Up + SiLU");
  m.def("selected_down_bf16_h32_k16", &selected_down_bf16_h32_k16, "Tiled selected Down, BLOCK_H=32, WARPS_K=16");
}

TORCH_LIBRARY(moc_native, m) {
  m.def("optimized_global_after_gate_bf16_out(Tensor x, Tensor gate_scores, Tensor up_weight, Tensor down_weight_t, Tensor(a!) idx, Tensor(b!) sparse_z, Tensor(c!) out, int k) -> ()");
}

TORCH_LIBRARY_IMPL(moc_native, CUDA, m) {
  m.impl("optimized_global_after_gate_bf16_out", &optimized_global_after_gate_bf16_out);
}

