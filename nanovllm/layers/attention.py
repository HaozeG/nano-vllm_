import os
import time
import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

# ---------------------------------------------------------------------------
# Profiling flags
# NANOVLLM_PROFILE_ATTN=1       — simple per-call timing to _attn_log
# NANOVLLM_PROFILE_ATTN_DETAIL=1 — per-subcomponent timing to _attn_detail_log
# NANOVLLM_PROFILE_SYNC=1       — add cuda.synchronize() at each timing boundary
# Do not combine PROFILE_ATTN and PROFILE_ATTN_DETAIL in the same run.
# ---------------------------------------------------------------------------
_PROFILE_ATTN        = os.getenv("NANOVLLM_PROFILE_ATTN",        "0") == "1"
_PROFILE_ATTN_DETAIL = os.getenv("NANOVLLM_PROFILE_ATTN_DETAIL", "0") == "1"
_PROFILE_SYNC        = os.getenv("NANOVLLM_PROFILE_SYNC",        "0") == "1"

# NANOVLLM_DISABLE_VECTOR_GATHER=1 falls back to per-sequence SDPA gather (full-attn decode).
# Requires eager mode (CUDA graphs use static shapes; this path calls ctx_lens.tolist()).
_DISABLE_VECTOR_GATHER = os.getenv("NANOVLLM_DISABLE_VECTOR_GATHER", "0") == "1"

_attn_log: list[dict] = []

# Staging dict for detail profiling: Attention.forward() populates, Gemma4TextAttention.forward() consumes.
# Never reassigned — only cleared/updated so that importers always reference the same object.
_attn_detail_log: list[dict] = []
_attn_detail_pending: dict = {}


def get_attn_log() -> list[dict]:
    return _attn_log


def clear_attn_log() -> None:
    _attn_log.clear()


def get_attn_detail_log() -> list[dict]:
    return _attn_detail_log


def clear_attn_detail_log() -> None:
    _attn_detail_log.clear()


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


@triton.jit
def store_kvcache_kernel_fp8(
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
    tl.store(k_cache_ptr + cache_offsets, key.to(tl.float8e4nv))
    tl.store(v_cache_ptr + cache_offsets, value.to(tl.float8e4nv))


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    kernel = store_kvcache_kernel_fp8 if k_cache.dtype == torch.float8_e4m3fn else store_kvcache_kernel
    kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def _paged_attn_fp8_kernel(
    Q,            # [bs, Hq, D] bfloat16
    K_cache,      # [num_blocks, BLOCK_SIZE, Hkv, D] float8e4nv
    V_cache,      # same
    Out,          # [bs, Hq, D] bfloat16
    ctx_lens,     # [bs] int32
    block_table,  # [bs, max_nb] int32
    scale,        # float32 scalar
    max_nb,       # runtime int
    window,       # runtime int (effective window = window_size[0] + 1)
    Hq:         tl.constexpr,
    Hkv:        tl.constexpr,
    D:          tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // Hq
    head  = pid % Hq
    kv_head = head // (Hq // Hkv)

    q_off = batch * Hq * D + head * D + tl.arange(0, D)
    q = tl.load(Q + q_off).to(tl.float32) * scale  # [D]

    ctx_len    = tl.load(ctx_lens + batch)
    start_tok  = tl.maximum(0, ctx_len - window)
    n_blocks   = tl.cdiv(ctx_len, BLOCK_SIZE)
    start_blk  = start_tok // BLOCK_SIZE

    m   = float(-1e20)
    l   = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    s_idx      = tl.arange(0, BLOCK_SIZE)
    d_idx      = tl.arange(0, D)
    kv_stride  = Hkv * D       # stride per token within a block
    blk_stride = BLOCK_SIZE * Hkv * D  # stride per physical block

    for blk in range(start_blk, n_blocks):
        phys = tl.load(block_table + batch * max_nb + blk)
        base = phys * blk_stride + kv_head * D

        k_off = base + s_idx[:, None] * kv_stride + d_idx[None, :]
        k = tl.load(K_cache + k_off).to(tl.float32)      # [BLOCK_SIZE, D]

        scores = tl.sum(q[None, :] * k, axis=1)          # [BLOCK_SIZE]
        pos    = blk * BLOCK_SIZE + s_idx
        scores = tl.where((pos >= start_tok) & (pos < ctx_len), scores, float(-1e20))

        m_new  = tl.maximum(m, tl.max(scores, axis=0))
        alpha  = tl.exp(m - m_new)
        exp_s  = tl.exp(scores - m_new)                  # [BLOCK_SIZE]

        v_off  = base + s_idx[:, None] * kv_stride + d_idx[None, :]
        v      = tl.load(V_cache + v_off).to(tl.float32) # [BLOCK_SIZE, D]

        acc = acc * alpha + tl.sum(exp_s[:, None] * v, axis=0)
        l   = l   * alpha + tl.sum(exp_s, axis=0)
        m   = m_new

    safe_l = tl.maximum(l, 1e-9)
    tl.store(Out + batch * Hq * D + head * D + d_idx, (acc / safe_l).to(tl.bfloat16))


def paged_attn_fp8_decode(
    q: torch.Tensor,             # [bs, Hq, D] bfloat16
    k_cache: torch.Tensor,       # [num_blocks, block_size, Hkv, D] float8e4nv
    v_cache: torch.Tensor,
    ctx_lens: torch.Tensor,      # [bs] int32
    block_tables: torch.Tensor,  # [bs, max_nb] int32
    scale: float,
    window: int,
) -> torch.Tensor:
    bs, Hq, D = q.shape
    Hkv       = k_cache.shape[2]
    block_size = k_cache.shape[1]
    max_nb    = block_tables.shape[1]
    out = torch.empty(bs, Hq, D, dtype=torch.bfloat16, device=q.device)
    _paged_attn_fp8_kernel[(bs * Hq,)](
        q, k_cache, v_cache, out,
        ctx_lens, block_tables, scale, max_nb, window,
        Hq=Hq, Hkv=Hkv, D=D, BLOCK_SIZE=block_size,
    )
    return out


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
            # Decode: gather KV for all sequences, then SDPA.
            bs = q.shape[0]

            if _DISABLE_VECTOR_GATHER:
                # Sequential per-sequence gather (NANOVLLM_DISABLE_VECTOR_GATHER=1).
                # Requires eager mode — ctx_lens.tolist() is a CPU sync.
                if _PROFILE_ATTN_DETAIL:
                    _tg = _ts()
                ctx_lens_list = context.context_lens.tolist()
                parts = []
                for i in range(bs):
                    ki = _gather_paged(k_cache, context.block_tables[i], ctx_lens_list[i], block_size)
                    vi = _gather_paged(v_cache, context.block_tables[i], ctx_lens_list[i], block_size)
                    parts.append(_sdpa_seq(q[i:i + 1], ki, vi, self.scale, self.num_heads))
                if _PROFILE_ATTN_DETAIL:
                    _attn_detail_pending["kv_gather_or_padding_ms"] = (_ts() - _tg) * 1000
                    _attn_detail_pending["attention_kernel_ms"] = 0.0
                return torch.cat(parts, dim=0)

            ctx_lens = context.context_lens.long()  # [bs]
            # Static upper bound — no CPU-GPU sync, required for CUDA graph compatibility.
            # block_tables.shape[1] == max_num_blocks for CUDA graph captures; equals the
            # per-batch max for eager runs. Padding mask zeros the extra positions.
            max_ctx = context.block_tables.shape[1] * k_cache.shape[1]

            if _PROFILE_ATTN_DETAIL:
                _tg = _ts()

            # Vectorised paged KV gather: k_cache[bt] → [B, max_nb, block_size, Hkv, D].
            # Positions beyond ctx_lens[i] hold stale cache data; the bias mask below
            # sets those positions to −∞ so they have no effect on the output.
            Hkv, Dkv = k_cache.shape[2], k_cache.shape[3]
            max_nb = (max_ctx + block_size - 1) // block_size
            bt = context.block_tables[:, :max_nb]          # [B, max_nb]
            k_pad = k_cache[bt].reshape(bs, max_nb * block_size, Hkv, Dkv)[:, :max_ctx]
            v_pad = v_cache[bt].reshape(bs, max_nb * block_size, Hkv, Dkv)[:, :max_ctx]
            q4 = q.unsqueeze(2)              # [bs, Hq, 1, D]
            k4 = k_pad.permute(0, 2, 1, 3)  # [bs, Hkv, max_ctx, D]
            v4 = v_pad.permute(0, 2, 1, 3)  # [bs, Hkv, max_ctx, D]
            if self.num_kv_heads < self.num_heads:
                k4 = k4.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                v4 = v4.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            # Padding mask: positions ≥ ctx_len[i] are ignored
            pad = torch.arange(max_ctx, device=q.device).unsqueeze(0) >= ctx_lens.unsqueeze(1)
            bias = q.new_zeros(bs, 1, 1, max_ctx).masked_fill_(pad[:, None, None, :], float('-inf'))

            if _PROFILE_ATTN_DETAIL:
                _attn_detail_pending["kv_gather_or_padding_ms"] = (_ts() - _tg) * 1000
                _tk = _ts()

            o = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=bias, scale=self.scale)

            if _PROFILE_ATTN_DETAIL:
                _attn_detail_pending["attention_kernel_ms"] = (_ts() - _tk) * 1000

            return o.squeeze(2)  # [bs, Hq, D]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if _PROFILE_ATTN_DETAIL:
            _attn_detail_pending.clear()
            _tstore = _ts()

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if _PROFILE_ATTN_DETAIL:
            _attn_detail_pending["kv_cache_update_ms"] = (_ts() - _tstore) * 1000

        if self._use_sdpa:
            if _PROFILE_ATTN_DETAIL and context.is_prefill:
                # Prefill SDPA: gather + kernel not separately timed (decode is the bottleneck)
                _attn_detail_pending["kv_gather_or_padding_ms"] = 0.0
                _attn_detail_pending["attention_kernel_ms"] = 0.0
            elif _PROFILE_ATTN and not _PROFILE_ATTN_DETAIL:
                t0 = _ts()
            o = self._sdpa_forward(q, k, v, context)
            if _PROFILE_ATTN and not _PROFILE_ATTN_DETAIL:
                _attn_log.append({
                    "layer_type": "full",
                    "is_prefill": context.is_prefill,
                    "n_tokens": q.shape[0],
                    "n_ctx": int(context.context_lens.float().mean().item()) if not context.is_prefill else int(context.cu_seqlens_k[-1].item()),
                    "t_ms": (_ts() - t0) * 1000,
                })
            return o

        # Flash-attn path (sliding attention, head_dim ≤ 256)
        if _PROFILE_ATTN_DETAIL:
            _attn_detail_pending["kv_gather_or_padding_ms"] = 0.0  # FA2 reads KV cache internally
            _tk = _ts()
        elif _PROFILE_ATTN:
            t0 = _ts()

        if context.is_prefill:
            if context.block_tables is not None and k_cache.dtype == torch.float8_e4m3fn:
                # Prefix-cache + FP8: gather only the prefix blocks from FP8 cache
                # (GPU fancy-index, no CPU sync) and cat with fresh K from projection.
                # Two small .tolist() syncs on cu_seqlens (≤11 ints) replace the
                # expensive per-element .item() that stalled GPU pipelining.
                bs_c = k_cache.shape[1]
                Hc, Dc = k_cache.shape[2], k_cache.shape[3]
                cu_q = context.cu_seqlens_q.tolist()
                cu_k = context.cu_seqlens_k.tolist()
                nseq = len(cu_q) - 1
                all_k, all_v = [], []
                for i in range(nseq):
                    sq = cu_q[i + 1] - cu_q[i]
                    sk = cu_k[i + 1] - cu_k[i]
                    plen = sk - sq
                    fk_i = k[cu_q[i]:cu_q[i] + sq]
                    fv_i = v[cu_q[i]:cu_q[i] + sq]
                    if plen > 0:
                        n_pb = (plen + bs_c - 1) // bs_c
                        pblks = context.block_tables[i, :n_pb]          # GPU slice, no sync
                        pk_i = k_cache[pblks].to(torch.bfloat16).reshape(-1, Hc, Dc)[:plen]
                        pv_i = v_cache[pblks].to(torch.bfloat16).reshape(-1, Hc, Dc)[:plen]
                        all_k.append(torch.cat([pk_i, fk_i]))
                        all_v.append(torch.cat([pv_i, fv_i]))
                    else:
                        all_k.append(fk_i)
                        all_v.append(fv_i)
                k, v = torch.cat(all_k), torch.cat(all_v)
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True,
                                           window_size=self.window_size)
            else:
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables,
                                           window_size=self.window_size)
        else:    # decode
            if k_cache.dtype == torch.float8_e4m3fn:
                # Custom Triton FP8 paged attention — reads half the KV bytes vs BF16
                window = self.window_size[0] + 1  # +1 matches FA2 window_size=(wl,0) convention
                o = paged_attn_fp8_decode(
                    q, k_cache, v_cache,
                    context.context_lens, context.block_tables,
                    self.scale, window,
                )
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True,
                                            window_size=self.window_size)

        if _PROFILE_ATTN_DETAIL:
            _attn_detail_pending["attention_kernel_ms"] = (_ts() - _tk) * 1000
        elif _PROFILE_ATTN:
            _attn_log.append({
                "layer_type": "sliding",
                "is_prefill": context.is_prefill,
                "n_tokens": q.shape[0],
                "n_ctx": int(context.context_lens.float().mean().item()) if not context.is_prefill else int(context.max_seqlen_k),
                "t_ms": (_ts() - t0) * 1000,
            })
        return o
