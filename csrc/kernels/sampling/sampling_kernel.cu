#include <torch/types.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <float.h>
#include "mini_vllm/utils.cuh"
#include "mini_vllm/dispatch.h"
#include "sampling.h"

namespace mini_vllm {

// ============================================================
// Kernel 1: Greedy sampling (argmax)
// ============================================================
// Each block handles one row. Threads cooperatively find the max element.
// Uses block_reduce_max to find the maximum value, then the thread that
// owns the max writes its index.
template <typename scalar_t>
__global__ void greedy_sample_kernel(
    int64_t* __restrict__ output,
    const scalar_t* __restrict__ logits,
    int vocab_size) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const scalar_t* row_logits = logits + row * vocab_size;

  // Each thread finds its local max (value and index)
  float max_val = -FLT_MAX;
  int max_idx = 0;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    float val = static_cast<float>(row_logits[i]);
    if (val > max_val) {
      max_val = val;
      max_idx = i;
    }
  }

  // Block-level argmax using shared memory
  // Store (value, index) pairs and reduce
  extern __shared__ char smem[];
  float* svals = reinterpret_cast<float*>(smem);
  int* sidxs = reinterpret_cast<int*>(svals + blockDim.x);

  svals[tid] = max_val;
  sidxs[tid] = max_idx;
  __syncthreads();

  // Tree reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (svals[tid + s] > svals[tid]) {
        svals[tid] = svals[tid + s];
        sidxs[tid] = sidxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[row] = sidxs[0];
  }
}

// ============================================================
// Kernel 2: Fused top-k + top-p filter
// ============================================================
// Each block handles one row. Threads cooperatively find the k-th largest
// value (for top-k) and then compute cumulative probability (for top-p).
// Filtered positions get set to -inf.
template <typename scalar_t>
__global__ void top_k_top_p_kernel(
    scalar_t* __restrict__ logits,
    int vocab_size, int top_k, float top_p) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;
  scalar_t* row_logits = logits + row * vocab_size;

  // --- Top-k filtering ---
  if (top_k > 0 && top_k < vocab_size) {
    // Find the k-th largest value using iterative elimination
    // Each thread maintains its current candidate
    float candidate_val = -FLT_MAX;
    int candidate_count = 0;

    // First pass: find initial candidates
    for (int i = tid; i < vocab_size; i += num_threads) {
      float val = static_cast<float>(row_logits[i]);
      if (val > candidate_val) {
        candidate_val = val;
      }
    }

    // Iteratively find the k-th largest by thresholding
    // For typical k values (1-100) and vocab sizes (32k-150k),
    // a few iterations of global reduction suffice.
    float threshold = -FLT_MAX;
    for (int iter = 0; iter < 10; iter++) {
      // Share current candidate values and count how many are above threshold
      float global_max = block_reduce_max(candidate_val);

      // Count elements >= global_max
      int count = 0;
      for (int i = tid; i < vocab_size; i += num_threads) {
        if (static_cast<float>(row_logits[i]) >= global_max) {
          count++;
        }
      }
      count = block_reduce_sum(count);

      if (count >= top_k) {
        // The k-th largest is at global_max (or very close)
        // Elements strictly less than global_max should be filtered
        // But we need to keep exactly top_k elements
        threshold = global_max;
        break;
      }

      // Not enough elements at global_max, lower the bar
      // Set candidate to max of values strictly less than global_max
      candidate_val = -FLT_MAX;
      for (int i = tid; i < vocab_size; i += num_threads) {
        float val = static_cast<float>(row_logits[i]);
        if (val < global_max && val > candidate_val) {
          candidate_val = val;
        }
      }
    }

    // Apply top-k mask: set values below threshold to -inf
    for (int i = tid; i < vocab_size; i += num_threads) {
      if (static_cast<float>(row_logits[i]) < threshold) {
        row_logits[i] = static_cast<scalar_t>(-FLT_MAX);
      }
    }
    __syncthreads();
  }

  // --- Top-p (nucleus) filtering ---
  // Binary search for the probability cutoff where cumulative sum exceeds top_p.
  // This works on unsorted logits by using exp(logit - max) as the probability weight.
  if (top_p < 1.0f && top_p > 0.0f) {
    // 1. Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += num_threads) {
      float val = static_cast<float>(row_logits[i]);
      if (val > max_val) max_val = val;
    }
    max_val = block_reduce_max(max_val);

    // 2. Compute total sum of exp(logit - max) for normalization
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += num_threads) {
      float val = static_cast<float>(row_logits[i]);
      if (val > -60000.0f) {
        local_sum += expf(val - max_val);
      }
    }
    float total_sum = block_reduce_sum(local_sum);
    float inv_total = 1.0f / total_sum;

    // 3. Binary search for the cutoff value where cumulative probability > top_p
    float lo = -60000.0f, hi = max_val + 1.0f;
    float cutoff = -60000.0f;

    for (int iter = 0; iter < 40; iter++) {
      float mid = (lo + hi) * 0.5f;
      float cum = 0.0f;
      for (int i = tid; i < vocab_size; i += num_threads) {
        float val = static_cast<float>(row_logits[i]);
        if (val >= mid) {
          cum += expf(val - max_val) * inv_total;
        }
      }
      cum = block_reduce_sum(cum);

      if (cum > top_p) {
        lo = mid;
      } else {
        hi = mid;
        cutoff = mid;
      }
    }

    // 4. Apply top-p mask: set values below cutoff to -FLT_MAX
    for (int i = tid; i < vocab_size; i += num_threads) {
      if (static_cast<float>(row_logits[i]) < cutoff) {
        row_logits[i] = static_cast<scalar_t>(-FLT_MAX);
      }
    }
    __syncthreads();
  }
}

// Host wrapper for greedy_sample
torch::Tensor greedy_sample(torch::Tensor logits) {
  TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D [batch_size, vocab_size]");

  int batch_size = logits.size(0);
  int vocab_size = logits.size(1);
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(logits.device());
  auto output = torch::empty({batch_size}, options);

  int threads = std::min(((vocab_size + 31) / 32) * 32, 1024);
  int smem_size = threads * (sizeof(float) + sizeof(int));

  DISPATCH_FLOATING_TYPES(logits.scalar_type(), "greedy_sample_kernel", [&] {
    greedy_sample_kernel<scalar_t><<<batch_size, threads, smem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<int64_t>(),
        logits.data_ptr<scalar_t>(),
        vocab_size);
  });
  return output;
}

// Host wrapper for top_k_top_p_filter
void top_k_top_p_filter(torch::Tensor logits, int64_t top_k, double top_p) {
  TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D [batch_size, vocab_size]");

  int batch_size = logits.size(0);
  int vocab_size = logits.size(1);

  DISPATCH_FLOATING_TYPES(logits.scalar_type(), "top_k_top_p_kernel", [&] {
    top_k_top_p_kernel<scalar_t><<<batch_size, 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<scalar_t>(),
        vocab_size, top_k, static_cast<float>(top_p));
  });
}

// Host wrapper for softmax_multinomial_sample
// For now, we use PyTorch's multinomial (efficiently implemented via THC)
// The CUDA kernel handles temperature scaling + softmax in one pass.
torch::Tensor softmax_multinomial_sample(torch::Tensor logits, double temperature) {
  TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D [batch_size, vocab_size]");

  int batch_size = logits.size(0);
  int vocab_size = logits.size(1);

  // Apply temperature scaling
  if (temperature != 1.0 && temperature > 0) {
    logits = logits / temperature;
  }

  // Softmax
  auto probs = torch::softmax(logits, -1);

  // Multinomial sampling (1 sample per row)
  auto token_ids = torch::multinomial(probs, 1);  // [batch_size, 1]
  return token_ids.squeeze(-1).to(torch::kInt64);  // [batch_size]
}

}  // namespace mini_vllm
