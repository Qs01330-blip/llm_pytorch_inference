#include <torch/extension.h>
#include "kernels/rmsnorm/rmsnorm.h"
#include "kernels/rope/rope.h"
#include "kernels/activation/activation.h"
#include "kernels/sampling/sampling.h"
#include "kernels/embedding/embedding.h"
#include "kernels/softmax/softmax.h"
#include "kernels/mlp/mlp.h"

TORCH_LIBRARY(mini_vllm_ops, m) {
  m.def("rmsnorm_forward(Tensor input, Tensor weight, float eps) -> Tensor");
  m.def("rmsnorm_plus_one_forward(Tensor input, Tensor weight, float eps) -> Tensor");
  m.def("fused_add_rmsnorm_forward(Tensor residual, Tensor input, Tensor weight, float eps) -> (Tensor, Tensor)");
  m.def("fused_add_rmsnorm_plus_one_forward(Tensor residual, Tensor input, Tensor weight, float eps) -> (Tensor, Tensor)");
  m.def("rotary_pos_emb_forward(Tensor input, Tensor cos, Tensor sin) -> Tensor");
  m.def("silu_forward(Tensor input) -> Tensor");
  m.def("gelu_forward(Tensor input) -> Tensor");
  m.def("swiglu_forward(Tensor gate, Tensor up) -> Tensor");
  m.def("geglu_forward(Tensor gate, Tensor up) -> Tensor");
  m.def("greedy_sample(Tensor logits) -> Tensor");
  m.def("top_k_top_p_filter(Tensor logits, int top_k, float top_p) -> ()");
  m.def("softmax_multinomial_sample(Tensor logits, float temperature) -> Tensor");
  m.def("embedding_forward(Tensor input_ids, Tensor weight, float scale) -> Tensor");
  m.def("softmax_forward(Tensor input) -> Tensor");
  m.def("fused_mlp_forward(Tensor gate_up, int intermediate_size, int activation) -> Tensor");
}

TORCH_LIBRARY_IMPL(mini_vllm_ops, CUDA, m) {
  m.impl("rmsnorm_forward", &mini_vllm::rmsnorm_forward);
  m.impl("rmsnorm_plus_one_forward", &mini_vllm::rmsnorm_plus_one_forward);
  m.impl("fused_add_rmsnorm_forward", &mini_vllm::fused_add_rmsnorm_forward);
  m.impl("fused_add_rmsnorm_plus_one_forward", &mini_vllm::fused_add_rmsnorm_plus_one_forward);
  m.impl("rotary_pos_emb_forward", &mini_vllm::rotary_pos_emb_forward);
  m.impl("silu_forward", &mini_vllm::silu_forward);
  m.impl("gelu_forward", &mini_vllm::gelu_forward);
  m.impl("swiglu_forward", &mini_vllm::swiglu_forward);
  m.impl("geglu_forward", &mini_vllm::geglu_forward);
  m.impl("greedy_sample", &mini_vllm::greedy_sample);
  m.impl("top_k_top_p_filter", &mini_vllm::top_k_top_p_filter);
  m.impl("softmax_multinomial_sample", &mini_vllm::softmax_multinomial_sample);
  m.impl("embedding_forward", &mini_vllm::embedding_forward);
  m.impl("softmax_forward", &mini_vllm::softmax_forward);
  m.impl("fused_mlp_forward", &mini_vllm::fused_mlp_forward);
}

// Trigger TORCH_LIBRARY static initialization on Windows
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "mini-vllm CUDA operators";
}
