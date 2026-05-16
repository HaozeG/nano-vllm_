import os
import torch
import triton
import triton.language as tl
from torch import nn

# NANOVLLM_DISABLE_TRITON_RMSNORM=1 forces @torch.compile fallback for all sizes.
_DISABLE_TRITON_NORM = os.getenv("NANOVLLM_DISABLE_TRITON_RMSNORM", "0") == "1"

# ---------------------------------------------------------------------------
# Triton RMSNorm kernels — one program per row, FP32 accum, BF16 I/O.
# Used for decode-scale tensors (N ≤ _TRITON_NORM_MAX_N) to cut Python
# dispatch and intermediate-tensor overhead vs the @torch.compile path.
# ---------------------------------------------------------------------------

@triton.jit
def _rms_norm_fwd(
    x_ptr, w_ptr, out_ptr,
    stride_row, H, eps,
    BLOCK_H: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    x    = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    var  = tl.sum(x * x, 0) / H
    rrms = tl.rsqrt(var + eps)
    w    = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(out_ptr + row * stride_row + offs, (x * rrms * w).to(tl.bfloat16), mask=mask)


@triton.jit
def _rms_norm_noscale_fwd(
    x_ptr, out_ptr,
    stride_row, H, eps,
    BLOCK_H: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    x    = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    var  = tl.sum(x * x, 0) / H
    rrms = tl.rsqrt(var + eps)
    tl.store(out_ptr + row * stride_row + offs, (x * rrms).to(tl.bfloat16), mask=mask)


# Triton path active for BF16 decode tensors; @torch.compile fallback for large prefill.
_TRITON_NORM_MAX_N = 256


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return self.add_rms_forward(x, residual)
        if (not _DISABLE_TRITON_NORM and x.dtype == torch.bfloat16 and x.ndim == 2
                and x.shape[0] <= _TRITON_NORM_MAX_N):
            x = x.contiguous()
            N, H = x.shape
            out = torch.empty_like(x)
            _rms_norm_fwd[(N,)](
                x, self.weight, out,
                x.stride(0), H, self.eps,
                BLOCK_H=triton.next_power_of_2(H), num_warps=4,
            )
            return out
        return self.rms_forward(x)


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable weight (used for v_norm in Gemma4)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    @torch.compile
    def _compiled(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(orig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not _DISABLE_TRITON_NORM and x.dtype == torch.bfloat16 and x.ndim == 2
                and x.shape[0] <= _TRITON_NORM_MAX_N):
            x = x.contiguous()
            N, H = x.shape
            out = torch.empty_like(x)
            _rms_norm_noscale_fwd[(N,)](
                x, out,
                x.stride(0), H, self.eps,
                BLOCK_H=triton.next_power_of_2(H), num_warps=4,
            )
            return out
        return self._compiled(x)
