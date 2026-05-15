"""
Triton grouped GEMM for sparse MoE expert dispatch.

Replaces the Python for-loop over active experts in Gemma4TextExperts.forward.
Two kernel launches per MoE layer (gate+up, then down) instead of ~70 sequential
F.linear calls, eliminating kernel-launch-latency as the decode bottleneck.

Usage (called from Gemma4TextExperts.forward):
    out = moe_experts_forward(x, top_k_idx, top_k_w, gate_up_proj, down_proj)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Core grouped GEMM kernel
# A[total_tok, K] × B[E, N, K]^T  →  C[total_tok, N]
# Grid: (n_active_experts,  ceil(N / BLOCK_N))
# ---------------------------------------------------------------------------

@triton.jit
def _grouped_gemm_kernel(
    a_ptr, stride_am, stride_ak,          # A [total_tok, K]
    b_ptr, stride_be, stride_bn, stride_bk,  # B [E, N, K]
    c_ptr, stride_cm, stride_cn,          # C [total_tok, N]
    starts_ptr,   # [n_active] start token offset per expert group
    counts_ptr,   # [n_active] token count per expert group
    eids_ptr,     # [n_active] expert weight index per group
    total_tok,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)   # active expert group
    pid_n = tl.program_id(1)   # output column tile

    start = tl.load(starts_ptr + pid_e)
    count = tl.load(counts_ptr + pid_e)
    eid   = tl.load(eids_ptr   + pid_e)

    n_off  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # N and K are multiples of BLOCK_N / BLOCK_K — masks always True, kept for generality
    n_mask = n_off < N

    # Loop over tokens for this expert in BLOCK_M chunks
    for m_base in range(0, count, BLOCK_M):
        m_off  = tl.arange(0, BLOCK_M)
        m_mask = (m_base + m_off) < count
        # Clamp index to avoid out-of-bounds address on masked tail rows
        tok = tl.minimum(start + m_base + m_off, total_tok - 1)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K-reduction tiles
        for k_base in range(0, K, BLOCK_K):
            k_off = k_base + tl.arange(0, BLOCK_K)

            # a: [BLOCK_M, BLOCK_K]
            a = tl.load(
                a_ptr + tok[:, None] * stride_am + k_off[None, :] * stride_ak,
                mask=m_mask[:, None],
                other=0.0,
            )

            # b: [BLOCK_N, BLOCK_K]  (B row-major: B[e, n, k])
            b = tl.load(
                b_ptr + eid * stride_be + n_off[:, None] * stride_bn + k_off[None, :] * stride_bk,
                mask=n_mask[:, None],
                other=0.0,
            )

            # acc += a @ b.T  →  [BLOCK_M, BLOCK_K] × [BLOCK_K, BLOCK_N]
            acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

        tl.store(
            c_ptr + tok[:, None] * stride_cm + n_off[None, :] * stride_cn,
            acc.to(c_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


def _grouped_gemm(
    x:         torch.Tensor,   # [total_tok, K]  sorted by expert
    w:         torch.Tensor,   # [E, N, K]
    eids:      torch.Tensor,   # [n_active] int32, expert weight indices
    starts:    torch.Tensor,   # [n_active] int32, start offsets
    counts:    torch.Tensor,   # [n_active] int32, group sizes
    BLOCK_M:   int = 16,
    BLOCK_N:   int = 64,
    BLOCK_K:   int = 64,
) -> torch.Tensor:
    total_tok, K = x.shape
    _, N, Kw     = w.shape
    assert K == Kw, f"K mismatch: x.K={K}  w.K={Kw}"
    n_active = eids.shape[0]

    out = torch.empty(total_tok, N, dtype=x.dtype, device=x.device)
    if n_active == 0 or total_tok == 0:
        return out

    grid = (n_active, triton.cdiv(N, BLOCK_N))
    _grouped_gemm_kernel[grid](
        x,   x.stride(0),   x.stride(1),
        w,   w.stride(0),   w.stride(1),   w.stride(2),
        out, out.stride(0), out.stride(1),
        starts, counts, eids,
        total_tok,
        K, N, BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return out


# ---------------------------------------------------------------------------
# Full MoE forward  (drop-in for Gemma4TextExperts.forward inner logic)
# ---------------------------------------------------------------------------

def moe_experts_forward(
    x:            torch.Tensor,   # [N_tok, H]
    top_k_idx:    torch.Tensor,   # [N_tok, K]
    top_k_w:      torch.Tensor,   # [N_tok, K]
    gate_up_proj: torch.Tensor,   # [E, 2I, H]
    down_proj:    torch.Tensor,   # [E, H, I]
) -> torch.Tensor:
    N_tok, H = x.shape
    K        = top_k_idx.shape[1]
    I        = down_proj.shape[2]
    out      = torch.zeros_like(x)

    # --- Sort token-expert assignments by expert index (same as PyTorch path) ---
    flat_e   = top_k_idx.view(-1)                                     # [N*K]
    flat_w   = top_k_w.view(-1)                                       # [N*K]
    tok_idx  = torch.arange(N_tok, device=x.device).repeat_interleave(K)  # [N*K]

    perm          = flat_e.argsort(stable=True)
    sorted_e      = flat_e[perm]
    sorted_w      = flat_w[perm]
    sorted_tok    = tok_idx[perm]

    # Fixed-width E-slot counts — CUDA-graph compatible.
    # scatter_add_ is safe; bincount may internally sync for output-size detection.
    E      = gate_up_proj.shape[0]
    counts = torch.zeros(E, dtype=torch.int32, device=x.device)
    counts.scatter_add_(0, flat_e, torch.ones(flat_e.shape[0], dtype=torch.int32, device=x.device))
    starts = torch.zeros(E, dtype=torch.int32, device=x.device)
    starts[1:] = counts[:-1].cumsum(0).to(torch.int32)
    eids   = torch.arange(E, dtype=torch.int32, device=x.device)

    total_tok = N_tok * K

    # --- Gate + up projection (single kernel launch) ---
    x_sorted = x[sorted_tok]                          # [total_tok, H]  gather input
    gu = _grouped_gemm(x_sorted, gate_up_proj, eids, starts, counts)  # [total_tok, 2I]

    # --- GeLU activation (PyTorch, fused op) ---
    act = torch.nn.functional.gelu(gu[..., :I], approximate="tanh") * gu[..., I:]  # [total_tok, I]

    # --- Down projection (single kernel launch) ---
    down_out = _grouped_gemm(act, down_proj, eids, starts, counts)    # [total_tok, H]

    # --- Weighted scatter-add back to output tensor ---
    out.index_add_(0, sorted_tok, (down_out * sorted_w[:, None]).to(out.dtype))

    return out
