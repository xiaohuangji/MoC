#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cub/block/block_radix_sort.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

__device__ __forceinline__ float bf16_bits_to_float(uint16_t x) {
  uint32_t y = static_cast<uint32_t>(x) << 16;
  return __uint_as_float(y);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float f) {
  uint32_t x = __float_as_uint(f);
  uint32_t lsb = (x >> 16) & 1U;
  uint32_t bias = 0x7FFFU + lsb;
  return static_cast<uint16_t>((x + bias) >> 16);
}

__device__ __forceinline__ uint32_t bf16_order_key(uint16_t x) {
  const uint32_t sign = static_cast<uint32_t>(x) & 0x8000U;
  return sign ? (static_cast<uint32_t>(x) ^ 0xFFFFU) : (static_cast<uint32_t>(x) ^ 0x8000U);
}

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xFFFFFFFFU, v, offset);
  }
  return v;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void cub_topk_bf16_kernel(
    const uint16_t* __restrict__ scores,
    uint16_t* __restrict__ values,
    int64_t* __restrict__ indices,
    int B,
    int I,
    int K) {
  using BlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockSort::TempStorage temp_storage;

  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const uint16_t* row = scores + static_cast<int64_t>(b) * I;
  uint16_t* out_v = values + static_cast<int64_t>(b) * K;
  int64_t* out_i = indices + static_cast<int64_t>(b) * K;

  float keys[ITEMS_PER_THREAD];
  int idxs[ITEMS_PER_THREAD];

  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int linear = tid * ITEMS_PER_THREAD + item;
    if (linear < I) {
      keys[item] = bf16_bits_to_float(row[linear]);
      idxs[item] = linear;
    } else {
      keys[item] = -3.4028234663852886e38F;
      idxs[item] = 0;
    }
  }

  BlockSort(temp_storage).SortDescending(keys, idxs);
  __syncthreads();

  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int rank = tid * ITEMS_PER_THREAD + item;
    if (rank < K) {
      const int src = idxs[item];
      out_i[rank] = static_cast<int64_t>(src);
      out_v[rank] = row[src];
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void cub_topk_idx_i32_bf16_kernel(
    const uint16_t* __restrict__ scores,
    int32_t* __restrict__ indices,
    int B,
    int I,
    int K) {
  using BlockSort = cub::BlockRadixSort<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockSort::TempStorage temp_storage;

  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const uint16_t* row = scores + static_cast<int64_t>(b) * I;
  int32_t* out_i = indices + static_cast<int64_t>(b) * K;

  uint32_t keys[ITEMS_PER_THREAD];
  int idxs[ITEMS_PER_THREAD];

  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int linear = tid * ITEMS_PER_THREAD + item;
    if (linear < I) {
      keys[item] = bf16_order_key(row[linear]);
      idxs[item] = linear;
    } else {
      keys[item] = 0U;
      idxs[item] = 0;
    }
  }

  BlockSort(temp_storage).SortDescending(keys, idxs, 0, 16);
  __syncthreads();

  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int rank = tid * ITEMS_PER_THREAD + item;
    if (rank < K) {
      out_i[rank] = static_cast<int32_t>(idxs[item]);
    }
  }
}

__global__ void selected_up_silu_warp_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ topk_vals,
    const int64_t* __restrict__ idx,
    const uint16_t* __restrict__ w_up,
    uint16_t* __restrict__ sparse_z,
    int B,
    int H,
    int I,
    int K) {
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp_global = global_thread >> 5;
  const int lane = threadIdx.x & 31;
  const int total_warps = B * K;
  if (warp_global >= total_warps) {
    return;
  }

  const int b = warp_global / K;
  const int k = warp_global - b * K;
  const int row = static_cast<int>(idx[static_cast<int64_t>(b) * K + k]);
  const uint16_t* x_row = x + static_cast<int64_t>(b) * H;
  const uint16_t* w_row = w_up + static_cast<int64_t>(row) * H;

  float acc = 0.0f;
  for (int h = lane; h < H; h += 32) {
    acc += bf16_bits_to_float(x_row[h]) * bf16_bits_to_float(w_row[h]);
  }
  acc = warp_sum(acc);

  if (lane == 0) {
    const float gate = bf16_bits_to_float(topk_vals[static_cast<int64_t>(b) * K + k]);
    const float silu = gate / (1.0f + __expf(-gate));
    sparse_z[static_cast<int64_t>(b) * K + k] = float_to_bf16_bits(acc * silu);
  }
}

__global__ void selected_up_silu_from_scores_i32_warp_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ gate_scores,
    const int32_t* __restrict__ idx,
    const uint16_t* __restrict__ w_up,
    uint16_t* __restrict__ sparse_z,
    int B,
    int H,
    int I,
    int K) {
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp_global = global_thread >> 5;
  const int lane = threadIdx.x & 31;
  const int total_warps = B * K;
  if (warp_global >= total_warps) {
    return;
  }

  const int b = warp_global / K;
  const int k = warp_global - b * K;
  const int row = static_cast<int>(idx[static_cast<int64_t>(b) * K + k]);
  const uint16_t* x_row = x + static_cast<int64_t>(b) * H;
  const uint16_t* gate_row = gate_scores + static_cast<int64_t>(b) * I;
  const uint16_t* w_row = w_up + static_cast<int64_t>(row) * H;

  float acc = 0.0f;
  for (int h = lane; h < H; h += 32) {
    acc += bf16_bits_to_float(x_row[h]) * bf16_bits_to_float(w_row[h]);
  }
  acc = warp_sum(acc);

  if (lane == 0) {
    const float gate = bf16_bits_to_float(gate_row[row]);
    const float silu = gate / (1.0f + __expf(-gate));
    sparse_z[static_cast<int64_t>(b) * K + k] = float_to_bf16_bits(acc * silu);
  }
}

template <int MAX_H>
__global__ void selected_up_silu_from_scores_i32_cached_x_warp_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ gate_scores,
    const int32_t* __restrict__ idx,
    const uint16_t* __restrict__ w_up,
    uint16_t* __restrict__ sparse_z,
    int B,
    int H,
    int I,
    int K) {
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp_global = global_thread >> 5;
  const int lane = threadIdx.x & 31;
  const int total_warps = B * K;
  if (warp_global >= total_warps) {
    return;
  }

  const int b = warp_global / K;
  const int k = warp_global - b * K;
  const uint16_t* x_row = x + static_cast<int64_t>(b) * H;
  __shared__ uint16_t x_cache[MAX_H];
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    x_cache[h] = x_row[h];
  }
  __syncthreads();

  const int row = static_cast<int>(idx[static_cast<int64_t>(b) * K + k]);
  const uint16_t* gate_row = gate_scores + static_cast<int64_t>(b) * I;
  const uint16_t* w_row = w_up + static_cast<int64_t>(row) * H;

  float acc = 0.0f;
  for (int h = lane; h < H; h += 32) {
    acc += bf16_bits_to_float(x_cache[h]) * bf16_bits_to_float(w_row[h]);
  }
  acc = warp_sum(acc);

  if (lane == 0) {
    const float gate = bf16_bits_to_float(gate_row[row]);
    const float silu = gate / (1.0f + __expf(-gate));
    sparse_z[static_cast<int64_t>(b) * K + k] = float_to_bf16_bits(acc * silu);
  }
}

template <int BLOCK_H, int WARPS_K>
__global__ void selected_down_tile_kernel(
    const uint16_t* __restrict__ sparse_z,
    const int64_t* __restrict__ idx,
    const uint16_t* __restrict__ w_down_t,
    uint16_t* __restrict__ out,
    int B,
    int K,
    int H,
    int I) {
  const int b = blockIdx.x;
  const int h0 = blockIdx.y * BLOCK_H;
  const int h_lane = threadIdx.x;
  const int k_lane = threadIdx.y;
  const int h = h0 + h_lane;

  float acc = 0.0f;
  if (h < H) {
    for (int k = k_lane; k < K; k += WARPS_K) {
      const int row = static_cast<int>(idx[static_cast<int64_t>(b) * K + k]);
      const float z = bf16_bits_to_float(sparse_z[static_cast<int64_t>(b) * K + k]);
      const float w = bf16_bits_to_float(w_down_t[static_cast<int64_t>(row) * H + h]);
      acc += z * w;
    }
  }

  __shared__ float partial[WARPS_K][BLOCK_H];
  partial[k_lane][h_lane] = acc;
  __syncthreads();

  if (k_lane == 0 && h < H) {
    float total = 0.0f;
    #pragma unroll
    for (int j = 0; j < WARPS_K; ++j) {
      total += partial[j][h_lane];
    }
    out[static_cast<int64_t>(b) * H + h] = float_to_bf16_bits(total);
  }
}

template <int BLOCK_H, int WARPS_K>
__global__ void selected_down_i32_tile_kernel(
    const uint16_t* __restrict__ sparse_z,
    const int32_t* __restrict__ idx,
    const uint16_t* __restrict__ w_down_t,
    uint16_t* __restrict__ out,
    int B,
    int K,
    int H,
    int I) {
  const int b = blockIdx.x;
  const int h0 = blockIdx.y * BLOCK_H;
  const int h_lane = threadIdx.x;
  const int k_lane = threadIdx.y;
  const int h = h0 + h_lane;

  float acc = 0.0f;
  if (h < H) {
    for (int k = k_lane; k < K; k += WARPS_K) {
      const int row = static_cast<int>(idx[static_cast<int64_t>(b) * K + k]);
      const float z = bf16_bits_to_float(sparse_z[static_cast<int64_t>(b) * K + k]);
      const float w = bf16_bits_to_float(w_down_t[static_cast<int64_t>(row) * H + h]);
      acc += z * w;
    }
  }

  __shared__ float partial[WARPS_K][BLOCK_H];
  partial[k_lane][h_lane] = acc;
  __syncthreads();

  if (k_lane == 0 && h < H) {
    float total = 0.0f;
    #pragma unroll
    for (int j = 0; j < WARPS_K; ++j) {
      total += partial[j][h_lane];
    }
    out[static_cast<int64_t>(b) * H + h] = float_to_bf16_bits(total);
  }
}

template <int BLOCK_H, int WARPS_K, int MAX_K>
__global__ void selected_down_i32_cached_idx_tile_kernel(
    const uint16_t* __restrict__ sparse_z,
    const int32_t* __restrict__ idx,
    const uint16_t* __restrict__ w_down_t,
    uint16_t* __restrict__ out,
    int B,
    int K,
    int H,
    int I) {
  const int b = blockIdx.x;
  const int h0 = blockIdx.y * BLOCK_H;
  const int h_lane = threadIdx.x;
  const int k_lane = threadIdx.y;
  const int h = h0 + h_lane;
  const int linear_thread = k_lane * BLOCK_H + h_lane;

  __shared__ int32_t idx_cache[MAX_K];
  for (int kk = linear_thread; kk < K; kk += BLOCK_H * WARPS_K) {
    idx_cache[kk] = idx[static_cast<int64_t>(b) * K + kk];
  }
  __syncthreads();

  float acc = 0.0f;
  if (h < H) {
    for (int k = k_lane; k < K; k += WARPS_K) {
      const int row = static_cast<int>(idx_cache[k]);
      const float z = bf16_bits_to_float(sparse_z[static_cast<int64_t>(b) * K + k]);
      const float w = bf16_bits_to_float(w_down_t[static_cast<int64_t>(row) * H + h]);
      acc += z * w;
    }
  }

  __shared__ float partial[WARPS_K][BLOCK_H];
  partial[k_lane][h_lane] = acc;
  __syncthreads();

  if (k_lane == 0 && h < H) {
    float total = 0.0f;
    #pragma unroll
    for (int j = 0; j < WARPS_K; ++j) {
      total += partial[j][h_lane];
    }
    out[static_cast<int64_t>(b) * H + h] = float_to_bf16_bits(total);
  }
}

}  // namespace

std::vector<torch::Tensor> cub_topk_bf16_512x11_cuda(torch::Tensor scores, int64_t k) {
  const int B = static_cast<int>(scores.size(0));
  const int I = static_cast<int>(scores.size(1));
  const int K = static_cast<int>(k);

  auto values = torch::empty({scores.size(0), k}, scores.options());
  auto indices = torch::empty(
      {scores.size(0), k},
      torch::TensorOptions().device(scores.device()).dtype(torch::kInt64));

  constexpr int THREADS = 512;
  constexpr int ITEMS = 11;
  auto stream = at::cuda::getCurrentCUDAStream();
  cub_topk_bf16_kernel<THREADS, ITEMS><<<static_cast<unsigned int>(B), THREADS, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(scores.data_ptr<at::BFloat16>()),
      reinterpret_cast<uint16_t*>(values.data_ptr<at::BFloat16>()),
      indices.data_ptr<int64_t>(),
      B,
      I,
      K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {values, indices};
}

torch::Tensor selected_up_silu_bf16_cuda(
    torch::Tensor x,
    torch::Tensor topk_vals,
    torch::Tensor idx,
    torch::Tensor w_up) {
  const int B = static_cast<int>(x.size(0));
  const int H = static_cast<int>(x.size(1));
  const int K = static_cast<int>(idx.size(1));
  const int I = static_cast<int>(w_up.size(0));
  auto out = torch::empty({x.size(0), idx.size(1)}, x.options());

  constexpr int THREADS = 256;
  const int total_warps = B * K;
  const int blocks = (total_warps * 32 + THREADS - 1) / THREADS;
  auto stream = at::cuda::getCurrentCUDAStream();
  selected_up_silu_warp_kernel<<<blocks, THREADS, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint16_t*>(topk_vals.data_ptr<at::BFloat16>()),
      idx.data_ptr<int64_t>(),
      reinterpret_cast<const uint16_t*>(w_up.data_ptr<at::BFloat16>()),
      reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
      B,
      H,
      I,
      K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor selected_down_bf16_h32_k16_cuda(
    torch::Tensor sparse_z,
    torch::Tensor idx,
    torch::Tensor w_down_t) {
  const int B = static_cast<int>(sparse_z.size(0));
  const int K = static_cast<int>(sparse_z.size(1));
  const int I = static_cast<int>(w_down_t.size(0));
  const int H = static_cast<int>(w_down_t.size(1));
  auto out = torch::empty({sparse_z.size(0), w_down_t.size(1)}, sparse_z.options());

  constexpr int BLOCK_H = 32;
  constexpr int WARPS_K = 16;
  const dim3 grid(static_cast<unsigned int>(B), static_cast<unsigned int>((H + BLOCK_H - 1) / BLOCK_H));
  const dim3 block(BLOCK_H, WARPS_K);
  auto stream = at::cuda::getCurrentCUDAStream();
  selected_down_tile_kernel<BLOCK_H, WARPS_K><<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(sparse_z.data_ptr<at::BFloat16>()),
      idx.data_ptr<int64_t>(),
      reinterpret_cast<const uint16_t*>(w_down_t.data_ptr<at::BFloat16>()),
      reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
      B,
      K,
      H,
      I);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor optimized_global_after_gate_bf16_cuda(
    torch::Tensor x,
    torch::Tensor gate_scores,
    torch::Tensor up_weight,
    torch::Tensor down_weight_t,
    int64_t k) {
  const int B = static_cast<int>(x.size(0));
  const int H = static_cast<int>(x.size(1));
  const int I = static_cast<int>(gate_scores.size(1));
  const int K = static_cast<int>(k);

  auto idx = torch::empty(
      {x.size(0), k},
      torch::TensorOptions().device(x.device()).dtype(torch::kInt32));
  auto sparse_z = torch::empty({x.size(0), k}, x.options());
  auto out = torch::empty({x.size(0), down_weight_t.size(1)}, x.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int TOPK_THREADS = 512;
  constexpr int TOPK_ITEMS = 11;
  cub_topk_idx_i32_bf16_kernel<TOPK_THREADS, TOPK_ITEMS><<<static_cast<unsigned int>(B), TOPK_THREADS, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(gate_scores.data_ptr<at::BFloat16>()),
      idx.data_ptr<int32_t>(),
      B,
      I,
      K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int UP_THREADS = 256;
  const int total_warps = B * K;
  const int up_blocks = (total_warps * 32 + UP_THREADS - 1) / UP_THREADS;
  if (H <= 2048 && K % (UP_THREADS / 32) == 0) {
    selected_up_silu_from_scores_i32_cached_x_warp_kernel<2048><<<up_blocks, UP_THREADS, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(gate_scores.data_ptr<at::BFloat16>()),
        idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(up_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(sparse_z.data_ptr<at::BFloat16>()),
        B,
        H,
        I,
        K);
  } else {
    selected_up_silu_from_scores_i32_warp_kernel<<<up_blocks, UP_THREADS, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(gate_scores.data_ptr<at::BFloat16>()),
        idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(up_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(sparse_z.data_ptr<at::BFloat16>()),
        B,
        H,
        I,
        K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int BLOCK_H = 32;
  constexpr int WARPS_K = 16;
  const dim3 down_grid(static_cast<unsigned int>(B), static_cast<unsigned int>((H + BLOCK_H - 1) / BLOCK_H));
  const dim3 down_block(BLOCK_H, WARPS_K);
  if (K <= 1024) {
    selected_down_i32_cached_idx_tile_kernel<BLOCK_H, WARPS_K, 1024><<<down_grid, down_block, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(sparse_z.data_ptr<at::BFloat16>()),
        idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(down_weight_t.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
        B,
        K,
        H,
        I);
  } else {
    selected_down_i32_tile_kernel<BLOCK_H, WARPS_K><<<down_grid, down_block, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(sparse_z.data_ptr<at::BFloat16>()),
        idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(down_weight_t.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
        B,
        K,
        H,
        I);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

