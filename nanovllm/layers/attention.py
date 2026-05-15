import os
import time
import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

# Attention profiling — enabled with NANOVLLM_PROFILE_ATTN=1
# Set NANOVLLM_PROFILE_SYNC=1 to add cuda.synchronize() at each timing boundary
_PROFILE_ATTN = os.getenv("NANOVLLM_PROFILE_ATTN", "0") == "1"
_PROFILE_SYNC  = os.getenv("NANOVLLM_PROFILE_SYNC",  "0") == "1"
_attn_log: list[dict] = []


def get_attn_log() -> list[dict]:
    return _attn_log


def clear_attn_log() -> None:
    _attn_log.clear()


def _ts() -> float:
    if _PROFILE_SYNC:
        torch.cuda.synchronize()
    return time.perf_counter()


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _gather_paged(cache: torch.Tensor, block_table: torch.Tensor, ctx_len: int, block_size: int) -> torch.Tensor:
    """Gather [ctx_len, n_kv_heads, head_dim] from paged KV cache."""
    nb = (ctx_len + block_size - 1) // block_size
    return cache[block_table[:nb]].reshape(-1, cache.shape[2], cache.shape[3])[:ctx_len]


def _sdpa_seq(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, num_q_heads: int) -> torch.Tensor:
    """
    Single-sequence SDPA with GQA expansion and causal masking.
    q: [sq, Hq, D], k/v: [sk, Hkv, D] → [sq, Hq, D]
    Causal: q[i] at absolute position (sk-sq+i) attends to k[0..sk-sq+i].
    """
    sq, _, D = q.shape
    sk, nkv, _ = k.shape
    if nkv < num_q_heads:
        k = k.repeat_interleave(num_q_heads // nkv, dim=1)
        v = v.repeat_interleave(num_q_heads // nkv, dim=1)
    q4 = q.permute(1, 0, 2).unsqueeze(0)   # [1, Hq, sq, D]
    k4 = k.permute(1, 0, 2).unsqueeze(0)
    v4 = v.permute(1, 0, 2).unsqueeze(0)
    col = torch.arange(sk, device=q.device)
    row = torch.arange(sq, device=q.device)
    block_mask = col.unsqueeze(0) > (sk - sq + row.unsqueeze(1))  # [sq, sk]
    attn_mask = q.new_zeros(1, 1, sq, sk).masked_fill_(block_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    o = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_mask, scale=scale)
    return o.squeeze(0).permute(1, 0, 2)   # [sq, Hq, D]


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        window_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size if window_size is not None else (-1, -1)
        self._use_sdpa = head_dim > 256   # flash_attn only supports head_dim <= 256
        self.k_cache = self.v_cache = torch.tensor([])

    def _sdpa_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context) -> torch.Tensor:
        """SDPA fallback path for head_dim > 256."""
        k_cache, v_cache = self.k_cache, self.v_cache
        block_size = k_cache.shape[1] if k_cache.numel() else 0
        if context.is_prefill:
            nseq = len(context.cu_seqlens_q) - 1
            parts = []
            for i in range(nseq):
                sq = int(context.cu_seqlens_q[i + 1] - context.cu_seqlens_q[i])
                sk = int(context.cu_seqlens_k[i + 1] - context.cu_seqlens_k[i])
                qs = int(context.cu_seqlens_q[i])
                qi = q[qs:qs + sq]
                if context.block_tables is not None:
                    ki = _gather_paged(k_cache, context.block_tables[i], sk, block_size)
                    vi = _gather_paged(v_cache, context.block_tables[i], sk, block_size)
                else:
                    ks = int(context.cu_seqlens_k[i])
                    ki = k[ks:ks + sk]
                    vi = v[ks:ks + sk]
                parts.append(_sdpa_seq(qi, ki, vi, self.scale, self.num_heads))
            return torch.cat(parts, dim=0)
        else:
            bs = q.shape[0]
            parts = []
            for i in range(bs):
                ctx = int(context.context_lens[i])
                ki = _gather_paged(k_cache, context.block_tables[i], ctx, block_size)
                vi = _gather_paged(v_cache, context.block_tables[i], ctx, block_size)
                parts.append(_sdpa_seq(q[i:i + 1], ki, vi, self.scale, self.num_heads))
            return torch.cat(parts, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if self._use_sdpa:
            if _PROFILE_ATTN:
                t0 = _ts()
            o = self._sdpa_forward(q, k, v, context)
            if _PROFILE_ATTN:
                _attn_log.append({
                    "layer_type": "full",
                    "is_prefill": context.is_prefill,
                    "n_tokens": q.shape[0],
                    "n_ctx": int(context.context_lens.float().mean().item()) if not context.is_prefill else int(context.cu_seqlens_k[-1].item()),
                    "t_ms": (_ts() - t0) * 1000,
                })
            return o
        if _PROFILE_ATTN:
            t0 = _ts()
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables,
                                       window_size=self.window_size)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True,
                                        window_size=self.window_size)
        if _PROFILE_ATTN:
            _attn_log.append({
                "layer_type": "sliding",
                "is_prefill": context.is_prefill,
                "n_tokens": q.shape[0],
                "n_ctx": int(context.context_lens.float().mean().item()) if not context.is_prefill else int(context.max_seqlen_k),
                "t_ms": (_ts() - t0) * 1000,
            })
        return o
