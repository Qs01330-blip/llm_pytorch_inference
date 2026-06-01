import torch
import pytest

# Reference implementations
def rmsnorm_ref(input, weight, eps=1e-6):
    variance = input.float().pow(2).mean(-1, keepdim=True)
    return (weight * input * torch.rsqrt(variance + eps)).to(input.dtype)

def rmsnorm_plus_one_ref(input, weight, eps=1e-6):
    variance = input.float().pow(2).mean(-1, keepdim=True)
    x = input * torch.rsqrt(variance + eps)
    return ((1.0 + weight.float()) * x).to(input.dtype)

def rmsnorm_gated_ref(input, gate, weight, eps=1e-6):
    x = input.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight.float() * x
    x = x * torch.nn.functional.silu(gate.float())
    return x.to(input.dtype)


SHAPES = [
    (1, 1, 4096),
    (4, 512, 2048),
    (1, 1, 896),
    (2, 128, 1024),
    (8, 1, 3584),
]

DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCudaRmsnorm:
    @pytest.fixture(autouse=True)
    def _load_ops(self):
        from mini_vllm.ops._utils import is_ext_loaded
        import torch
        if not (is_ext_loaded() and torch.cuda.is_available()):
            pytest.skip("CUDA extension not compiled or no GPU")
        from mini_vllm.ops.rmsnorm import rmsnorm, rmsnorm_plus_one, rmsnorm_gated
        self.rmsnorm = rmsnorm
        self.rmsnorm_plus_one = rmsnorm_plus_one
        self.rmsnorm_gated = rmsnorm_gated

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_rmsnorm(self, shape, dtype):
        torch.manual_seed(42)
        input = torch.randn(*shape, dtype=dtype, device="cuda")
        weight = torch.ones(shape[-1], dtype=dtype, device="cuda")

        out = self.rmsnorm(input, weight)
        ref = rmsnorm_ref(input, weight)

        atol = 1e-3 if dtype == torch.float16 else 2e-2
        rtol = 1e-3 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_rmsnorm_plus_one(self, shape, dtype):
        torch.manual_seed(42)
        input = torch.randn(*shape, dtype=dtype, device="cuda")
        weight = torch.zeros(shape[-1], dtype=dtype, device="cuda")

        out = self.rmsnorm_plus_one(input, weight)
        ref = rmsnorm_plus_one_ref(input, weight)

        atol = 1e-3 if dtype == torch.float16 else 2e-2
        rtol = 1e-3 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_rmsnorm_gated(self, shape, dtype):
        torch.manual_seed(42)
        input = torch.randn(*shape, dtype=dtype, device="cuda")
        gate = torch.randn(*shape, dtype=dtype, device="cuda")
        weight = torch.ones(shape[-1], dtype=dtype, device="cuda")

        out = self.rmsnorm_gated(input, gate, weight)
        ref = rmsnorm_gated_ref(input, gate, weight)

        atol = 1e-3 if dtype == torch.float16 else 2e-2
        rtol = 1e-3 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
