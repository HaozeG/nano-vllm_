import os
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovllm.layers.activation import GeluAndMul
from nanovllm.layers.attention import (Attention, _PROFILE_ATTN_DETAIL,
                                        _attn_detail_log, _attn_detail_pending)
from nanovllm.layers.layernorm import RMSNorm, RMSNormNoScale
from nanovllm.layers.rotary_embedding import get_rope, get_partial_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.moe_kernels import moe_experts_forward as _moe_triton
from nanovllm.utils.context import get_context as _get_context

# ---------------------------------------------------------------------------
# Profiling flags — enabled with env vars; zero overhead when disabled.
# NANOVLLM_PROFILE_MOE=1   — MoE-only timing (router + expert GEMM)
# NANOVLLM_PROFILE_LAYER=1 — full decoder-layer component timing
# NANOVLLM_PROFILE_SYNC=1  — add cuda.synchronize() at each boundary
# Do not combine PROFILE_MOE and PROFILE_LAYER in the same run.
# ---------------------------------------------------------------------------
_PROFILE_MOE   = os.getenv("NANOVLLM_PROFILE_MOE",   "0") == "1"
_PROFILE_LAYER = os.getenv("NANOVLLM_PROFILE_LAYER",  "0") == "1"
_PROFILE_SYNC  = os.getenv("NANOVLLM_PROFILE_SYNC",  "0") == "1"

_moe_log:   list[dict] = []
_layer_log: list[dict] = []


def get_moe_log() -> list[dict]:
    return _moe_log


def clear_moe_log() -> None:
    _moe_log.clear()


def get_layer_profile_log() -> list[dict]:
    return _layer_log


def clear_layer_profile_log() -> None:
    _layer_log.clear()


def _ts() -> float:
    if _PROFILE_SYNC:
        torch.cuda.synchronize()
    return perf_counter()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Gemma4TextRouter(nn.Module):

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        E = config.num_experts
        self.top_k = config.top_k_experts
        self.eps = config.rms_norm_eps
        self.scalar = H ** -0.5
        self.proj = nn.Linear(H, E, bias=False)
        self.scale = nn.Parameter(torch.ones(H))
        self.per_expert_scale = nn.Parameter(torch.ones(E))

    @torch.compile
    def _norm_and_scale(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.scale * self.scalar

    def forward(self, x: torch.Tensor):
        xf = self._norm_and_scale(x)
        logits = self.proj(xf)
        probs = F.softmax(logits, dim=-1)
        top_k_w, top_k_idx = probs.topk(self.top_k, dim=-1)
        top_k_w = top_k_w / top_k_w.sum(dim=-1, keepdim=True)
        top_k_w = top_k_w * self.per_expert_scale[top_k_idx]
        return top_k_w, top_k_idx


# ---------------------------------------------------------------------------
# Experts
# ---------------------------------------------------------------------------

class Gemma4TextExperts(nn.Module):

    def __init__(self, config):
        super().__init__()
        E = config.num_experts
        I = config.moe_intermediate_size
        H = config.hidden_size
        self.I = I
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * I, H))
        self.down_proj = nn.Parameter(torch.empty(E, H, I))

    # Tokens above this threshold use the sorted-dispatch PyTorch path to
    # avoid the large x_gathered temporary ([total_tok, H]) on prefill.
    _TRITON_TOK_LIMIT = 2048

    def forward(self, x: torch.Tensor, top_k_idx: torch.Tensor, top_k_w: torch.Tensor):
        # x: [N, H]  top_k_idx/top_k_w: [N, K]
        N, K = top_k_idx.shape[0], top_k_idx.shape[1]
        I = self.I

        if _PROFILE_MOE:
            t0 = _ts()

        # --- Triton grouped GEMM path (decode & small prefill) ---
        if N * K <= self._TRITON_TOK_LIMIT:
            out = _moe_triton(x, top_k_idx, top_k_w, self.gate_up_proj, self.down_proj)
            if _PROFILE_MOE:
                t2 = _ts()
                stats = {"n_tokens": N, "n_assignments": N * K,
                         "t_dispatch_ms": 0.0, "t_gemm_ms": (t2 - t0) * 1000,
                         "t_total_ms": (t2 - t0) * 1000}
                if _moe_log:
                    _moe_log[-1].update(stats)
                else:
                    _moe_log.append(stats)
                _moe_log[-1].setdefault("n_active", -1)
            return out

        # --- Sorted-dispatch PyTorch path (large prefill) ---
        out = torch.zeros_like(x)

        flat_experts = top_k_idx.view(-1)
        flat_weights = top_k_w.view(-1)
        token_idx    = torch.arange(N, device=x.device).repeat_interleave(K)

        perm           = flat_experts.argsort(stable=True)
        sorted_experts = flat_experts[perm]
        sorted_weights = flat_weights[perm]
        sorted_tok_idx = token_idx[perm]

        unique_e, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        expert_ids  = unique_e.tolist()
        group_sizes = counts.tolist()

        if _PROFILE_MOE:
            t1 = _ts()

        offset = 0
        for e, cnt in zip(expert_ids, group_sizes):
            tok  = sorted_tok_idx[offset:offset + cnt]
            w    = sorted_weights[offset:offset + cnt, None]
            x_e  = x[tok]
            gu   = F.linear(x_e, self.gate_up_proj[e])
            act  = F.gelu(gu[..., :I], approximate="tanh") * gu[..., I:]
            out_e = F.linear(act, self.down_proj[e])
            out.index_add_(0, tok, (out_e * w).to(out.dtype))
            offset += cnt

        if _PROFILE_MOE:
            t2 = _ts()
            stats = {"n_tokens": N, "n_assignments": N * K,
                     "n_active": len(expert_ids),
                     "t_dispatch_ms": (t1 - t0) * 1000,
                     "t_gemm_ms": (t2 - t1) * 1000,
                     "t_total_ms": (t2 - t0) * 1000}
            if _moe_log:
                _moe_log[-1].update(stats)
            else:
                _moe_log.append(stats)

        return out


# ---------------------------------------------------------------------------
# Shared expert MLP
# ---------------------------------------------------------------------------

class Gemma4TextMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Gemma4TextAttention(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self._layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = (self.layer_type == "sliding_attention")
        self.num_heads = config.num_attention_heads
        H = config.hidden_size
        eps = config.rms_norm_eps

        if self.is_sliding:
            self.head_dim = config.head_dim
            self.num_kv_heads = config.num_key_value_heads
            window = (config.sliding_window, 0)
        else:
            self.head_dim = config.global_head_dim
            self.num_kv_heads = config.num_global_key_value_heads or config.num_key_value_heads
            window = None

        # attention_k_eq_v: for full-attention layers, V = K (no v_proj weight)
        self.kv_eq = config.attention_k_eq_v and not self.is_sliding

        self.q_proj = nn.Linear(H, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(H, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = None if self.kv_eq else nn.Linear(H, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, H, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=eps)

        # RoPE — sliding uses full rotation; full-attention uses partial proportional RoPE
        rp = config.rope_parameters[self.layer_type]
        theta = rp["rope_theta"]
        max_pos = config.max_position_embeddings
        if self.is_sliding:
            self.rotary_emb = get_rope(self.head_dim, self.head_dim, max_pos, theta)
        else:
            partial = rp.get("partial_rotary_factor", 1.0)
            rotary_dim = int(self.head_dim * partial)
            self.rotary_emb = get_partial_rope(self.head_dim, rotary_dim, max_pos, theta)

        self.attn = Attention(
            self.num_heads, self.head_dim,
            scale=1.0,  # Gemma4 uses scale=1.0; q_norm/k_norm compensate for head_dim
            num_kv_heads=self.num_kv_heads,
            window_size=window,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if not _PROFILE_ATTN_DETAIL:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states) if not self.kv_eq else k
            N = q.shape[0]
            q = q.view(N, self.num_heads, self.head_dim)
            k = k.view(N, self.num_kv_heads, self.head_dim)
            v = v.view(N, self.num_kv_heads, self.head_dim)
            # Reshape to 2D for norm to avoid torch.compile rank-mismatch recompilations
            q = self.q_norm(q.reshape(N * self.num_heads, self.head_dim)).view(N, self.num_heads, self.head_dim)
            k = self.k_norm(k.reshape(N * self.num_kv_heads, self.head_dim)).view(N, self.num_kv_heads, self.head_dim)
            v = self.v_norm(v.reshape(N * self.num_kv_heads, self.head_dim)).view(N, self.num_kv_heads, self.head_dim)
            q, k = self.rotary_emb(positions, q, k)
            o = self.attn(q, k, v)
            return self.o_proj(o.flatten(1, -1))

        # Profiling path — same computation with per-subcomponent timing.
        _ctx = _get_context()
        _rec: dict = {
            "layer_idx": self._layer_idx,
            "layer_type": self.layer_type,
            "is_prefill": _ctx.is_prefill,
            "n_tokens": hidden_states.shape[0],
            "batch_size": (hidden_states.shape[0] if _ctx.is_prefill
                           else int(len(_ctx.context_lens))),
            "context_len": (int(_ctx.cu_seqlens_k[-1].item()) if _ctx.is_prefill
                            else int(_ctx.context_lens.float().mean().item())),
        }
        _tA = _ts()

        _t = _ts(); q = self.q_proj(hidden_states);               _rec["q_proj_ms"] = (_ts() - _t) * 1000
        _t = _ts(); k = self.k_proj(hidden_states);               _rec["k_proj_ms"] = (_ts() - _t) * 1000
        if self.kv_eq:
            _rec["v_proj_ms"] = 0.0
            v = k
        else:
            _t = _ts(); v = self.v_proj(hidden_states);           _rec["v_proj_ms"] = (_ts() - _t) * 1000

        N = q.shape[0]
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        _t = _ts()
        q = self.q_norm(q.reshape(N * self.num_heads, self.head_dim)).view(N, self.num_heads, self.head_dim)
        _rec["q_norm_ms"] = (_ts() - _t) * 1000
        _t = _ts()
        k = self.k_norm(k.reshape(N * self.num_kv_heads, self.head_dim)).view(N, self.num_kv_heads, self.head_dim)
        _rec["k_norm_ms"] = (_ts() - _t) * 1000
        _t = _ts()
        v = self.v_norm(v.reshape(N * self.num_kv_heads, self.head_dim)).view(N, self.num_kv_heads, self.head_dim)
        _rec["v_norm_ms"] = (_ts() - _t) * 1000

        _t = _ts(); q, k = self.rotary_emb(positions, q, k);     _rec["rotary_ms"] = (_ts() - _t) * 1000

        o = self.attn(q, k, v)  # populates _attn_detail_pending

        _rec["kv_cache_update_ms"]      = _attn_detail_pending.get("kv_cache_update_ms", 0.0)
        _rec["kv_gather_or_padding_ms"] = _attn_detail_pending.get("kv_gather_or_padding_ms", 0.0)
        _rec["attention_kernel_ms"]     = _attn_detail_pending.get("attention_kernel_ms", 0.0)

        _t = _ts(); result = self.o_proj(o.flatten(1, -1));       _rec["o_proj_ms"] = (_ts() - _t) * 1000

        _rec["total_ms"] = (_ts() - _tA) * 1000
        _attn_detail_log.append(_rec)
        return result


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class Gemma4TextDecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self._layer_idx = layer_idx
        H = config.hidden_size
        eps = config.rms_norm_eps

        self.self_attn = Gemma4TextAttention(config, layer_idx)
        if getattr(config, 'use_double_wide_mlp', False):
            n_single = config.num_hidden_layers - getattr(config, 'num_kv_shared_layers', 0)
            intermediate_size = config.intermediate_size * 2 if layer_idx >= n_single else config.intermediate_size
        else:
            intermediate_size = config.intermediate_size
        self.mlp = Gemma4TextMLP(H, intermediate_size)

        self.input_layernorm = RMSNorm(H, eps=eps)
        self.post_attention_layernorm = RMSNorm(H, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(H, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(H, eps=eps)

        # layer_scalar: per-layer multiplicative scaling (loaded as buffer, stored as param here)
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

        self.enable_moe = config.enable_moe_block
        if self.enable_moe:
            self.router = Gemma4TextRouter(config)
            self.experts = Gemma4TextExperts(config)
            self.post_feedforward_layernorm_1 = RMSNorm(H, eps=eps)
            self.post_feedforward_layernorm_2 = RMSNorm(H, eps=eps)
            self.pre_feedforward_layernorm_2 = RMSNorm(H, eps=eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # ----------------------------------------------------------------
        # Fast path — no profiling overhead.
        # ----------------------------------------------------------------
        if not _PROFILE_LAYER:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(positions, hidden_states)
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            if self.enable_moe:
                hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
                x_flat = residual.reshape(-1, residual.shape[-1])
                if _PROFILE_MOE:
                    _tr0 = _ts()
                top_k_w, top_k_idx = self.router(x_flat)
                if _PROFILE_MOE:
                    _moe_log.append({"t_router_ms": (_ts() - _tr0) * 1000})
                x_moe = self.pre_feedforward_layernorm_2(x_flat)
                moe_out = self.experts(x_moe, top_k_idx, top_k_w)
                moe_out = self.post_feedforward_layernorm_2(moe_out.reshape(residual.shape))
                hidden_states = hidden_states_1 + moe_out
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states * self.layer_scalar

        # ----------------------------------------------------------------
        # Profiling path (NANOVLLM_PROFILE_LAYER=1).
        # Computes the same result; adds fine-grained per-component timing.
        # ----------------------------------------------------------------
        _ctx = _get_context()
        _rec: dict = {
            "layer_idx": self._layer_idx,
            "layer_type": self.self_attn.layer_type,
            "is_prefill": _ctx.is_prefill,
            "n_tokens": hidden_states.shape[0],
        }
        _tL = _ts()

        residual = hidden_states
        _t = _ts(); hidden_states = self.input_layernorm(hidden_states)
        _rec["t_input_norm_ms"] = (_ts() - _t) * 1000

        _t = _ts(); hidden_states = self.self_attn(positions, hidden_states)
        _rec["t_self_attn_ms"] = (_ts() - _t) * 1000

        _t = _ts(); hidden_states = self.post_attention_layernorm(hidden_states)
        _rec["t_post_attn_norm_ms"] = (_ts() - _t) * 1000

        _t = _ts(); hidden_states = residual + hidden_states
        _rec["t_attn_residual_ms"] = (_ts() - _t) * 1000

        residual = hidden_states
        _t = _ts(); hidden_states = self.pre_feedforward_layernorm(hidden_states)
        _rec["t_pre_ff_norm_ms"] = (_ts() - _t) * 1000

        # Shared MLP: time each sub-op
        _t = _ts(); _g = self.mlp.gate_proj(hidden_states)
        _rec["t_mlp_gate_ms"] = (_ts() - _t) * 1000
        _t = _ts(); _u = self.mlp.up_proj(hidden_states)
        _rec["t_mlp_up_ms"] = (_ts() - _t) * 1000
        _t = _ts(); _a = F.gelu(_g, approximate="tanh") * _u
        _rec["t_mlp_act_ms"] = (_ts() - _t) * 1000
        _t = _ts(); hidden_states = self.mlp.down_proj(_a)
        _rec["t_mlp_down_ms"] = (_ts() - _t) * 1000
        _rec["t_mlp_ms"] = (_rec["t_mlp_gate_ms"] + _rec["t_mlp_up_ms"]
                            + _rec["t_mlp_act_ms"] + _rec["t_mlp_down_ms"])

        if self.enable_moe:
            _t = _ts(); hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            _rec["t_post_ff_norm1_ms"] = (_ts() - _t) * 1000

            x_flat = residual.reshape(-1, residual.shape[-1])

            # Router sub-components
            _t = _ts(); _xf = self.router._norm_and_scale(x_flat)
            _rec["t_router_norm_ms"] = (_ts() - _t) * 1000
            _t = _ts(); _logits = self.router.proj(_xf)
            _rec["t_router_linear_ms"] = (_ts() - _t) * 1000
            _t = _ts()
            _probs = F.softmax(_logits, dim=-1)
            top_k_w, top_k_idx = _probs.topk(self.router.top_k, dim=-1)
            top_k_w = top_k_w / top_k_w.sum(dim=-1, keepdim=True)
            top_k_w = top_k_w * self.router.per_expert_scale[top_k_idx]
            _rec["t_router_topk_ms"] = (_ts() - _t) * 1000
            _rec["t_router_ms"] = (_rec["t_router_norm_ms"] + _rec["t_router_linear_ms"]
                                   + _rec["t_router_topk_ms"])

            _t = _ts(); x_moe = self.pre_feedforward_layernorm_2(x_flat)
            _rec["t_pre_ff_norm2_ms"] = (_ts() - _t) * 1000

            _t = _ts(); moe_out = self.experts(x_moe, top_k_idx, top_k_w)
            _rec["t_experts_ms"] = (_ts() - _t) * 1000

            _t = _ts()
            moe_out = self.post_feedforward_layernorm_2(moe_out.reshape(residual.shape))
            _rec["t_post_ff_norm2_ms"] = (_ts() - _t) * 1000

            hidden_states = hidden_states_1 + moe_out

        _t = _ts(); hidden_states = self.post_feedforward_layernorm(hidden_states)
        _rec["t_post_ff_norm_final_ms"] = (_ts() - _t) * 1000

        _t = _ts(); hidden_states = residual + hidden_states
        _rec["t_final_residual_ms"] = (_ts() - _t) * 1000

        _t = _ts(); hidden_states = hidden_states * self.layer_scalar
        _rec["t_layer_scalar_ms"] = (_ts() - _t) * 1000

        _rec["t_total_ms"] = (_ts() - _tL) * 1000
        _layer_log.append(_rec)
        return hidden_states


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------

class Gemma4TextModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size ** 0.5
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# CausalLM wrapper — weight key hierarchy matches "model.language_model.*"
# ---------------------------------------------------------------------------

class _Gemma4Outer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_model = Gemma4TextModel(config)


class Gemma4ForCausalLM(nn.Module):
    # Keys in the safetensors file follow "model.language_model.*"
    # Our attribute path self.model.language_model.* matches that after get_parameter().
    # Vision / audio weights ("model.vision_tower.*", "model.embed_vision.*") are skipped.
    skip_weight_prefixes = ("model.vision_tower.", "model.embed_vision.", "model.audio_tower.")

    def __init__(self, hf_config):
        super().__init__()
        tc = hf_config.text_config
        self.final_logit_softcapping = getattr(tc, "final_logit_softcapping", None)
        self.model = _Gemma4Outer(tc)
        self.lm_head = ParallelLMHead(tc.vocab_size, tc.hidden_size)
        if tc.tie_word_embeddings:
            # Share the same Parameter so the loader's in-place copy propagates to lm_head.
            # .data= would copy initial garbage values and break after weight loading.
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model.language_model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if logits is not None and self.final_logit_softcapping:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits
