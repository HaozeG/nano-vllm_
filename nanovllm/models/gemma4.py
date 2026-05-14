import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovllm.layers.activation import GeluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm, RMSNormNoScale
from nanovllm.layers.rotary_embedding import get_rope, get_partial_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


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

    def forward(self, x: torch.Tensor):
        # RMSNorm without learnable scale, then apply scale param
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        xf = xf.to(x.dtype) * self.scale * self.scalar
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

    def forward(self, x: torch.Tensor, top_k_idx: torch.Tensor, top_k_w: torch.Tensor):
        # x: [N, H]  top_k_idx/top_k_w: [N, K]
        N = x.shape[0]
        out = torch.zeros_like(x)
        I = self.I
        active = top_k_idx.view(-1).unique()
        for e_item in active:
            e = e_item.item()
            mask = (top_k_idx == e)        # [N, K]
            tok_idx, k_pos = mask.nonzero(as_tuple=True)
            if tok_idx.numel() == 0:
                continue
            x_e = x[tok_idx]                                           # [T, H]
            gu = F.linear(x_e, self.gate_up_proj[e])                  # [T, 2I]
            act = F.gelu(gu[..., :I], approximate="tanh") * gu[..., I:]  # [T, I]
            out_e = F.linear(act, self.down_proj[e])                   # [T, H]
            out_e = out_e * top_k_w[tok_idx, k_pos, None]
            out.index_add_(0, tok_idx, out_e.to(out.dtype))
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
            scale=self.head_dim ** -0.5,
            num_kv_heads=self.num_kv_heads,
            window_size=window,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class Gemma4TextDecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
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
        # Attention sub-layer: norm → attn → post-norm → residual add
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Feedforward sub-layer: residual → pre-norm → shared MLP
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            x_flat = residual.reshape(-1, residual.shape[-1])
            top_k_w, top_k_idx = self.router(x_flat)
            x_moe = self.pre_feedforward_layernorm_2(x_flat)
            moe_out = self.experts(x_moe, top_k_idx, top_k_w)
            moe_out = self.post_feedforward_layernorm_2(moe_out.reshape(residual.shape))
            hidden_states = hidden_states_1 + moe_out

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------

class Gemma4TextModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
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
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model.language_model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if logits is not None and self.final_logit_softcapping:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits
