from typing import Tuple
import einops
import torch
from torch import nn
import torch.nn.functional as F
import math
#try:
#    from flash_attn_interface import flash_attn_func  # type: ignore[import]
#except ImportError:
#    # Fallback to FlashAttention 2
#    from flash_attn import flash_attn_func  # type: ignore[import]
from torch.nn.functional import scaled_dot_product_attention

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)

class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

class GatedAttentionQwenStyle(nn.Module):
    """
    Qwen3-style Gated Attention: gate scores from query projection.
    
    Inner loop gating: controls attention output quality
    
    Two modes:
    - headwise: one gate per attention head (efficient)
    - elementwise: one gate per element (fine-grained)
    """
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, 
                 causal=False, gate_mode='headwise'):
        super().__init__()
        
        assert gate_mode in ['headwise', 'elementwise', 'none'], \
            "gate_mode must be 'headwise', 'elementwise', or 'none'"

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads  # ⭐ Fix: 统一变量名
        self.causal = causal
        self.gate_mode = gate_mode

        # QKV projection with gate
        if gate_mode == 'headwise':
            q_out_dim = self.num_heads * self.head_dim + self.num_heads
        elif gate_mode == 'elementwise':
            q_out_dim = self.num_heads * self.head_dim * 2
        else:  # 'none'
            q_out_dim = self.num_heads * self.head_dim
        
        self.qkv_proj = CastedLinear(
            self.hidden_size, 
            q_out_dim + 2 * self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V (with extra dimension for gate if needed)
        qkv = self.qkv_proj(hidden_states)
        
        # Split Q, Gate, K, V
        if self.gate_mode == 'none':
            query_states = qkv[..., :self.hidden_size]
            gate_score = None
            # ⭐ Fix: 使用 kv_dim 而不是 hidden_size，支持 GQA
            kv_dim = self.num_key_value_heads * self.head_dim
            key_states = qkv[..., self.hidden_size : self.hidden_size + kv_dim]
            value_states = qkv[..., self.hidden_size + kv_dim :]
        else:
            gate_size = self.num_heads if self.gate_mode == 'headwise' else self.hidden_size
            
            query_states = qkv[..., :self.hidden_size]
            gate_score = qkv[..., self.hidden_size : self.hidden_size + gate_size]
            
            # ⭐ Fix: 使用 kv_dim 进行切片
            kv_dim = self.num_key_value_heads * self.head_dim
            start_k = self.hidden_size + gate_size
            key_states = qkv[..., start_k : start_k + kv_dim]
            value_states = qkv[..., start_k + kv_dim : start_k + 2*kv_dim]
        
        # Reshape to [B, S, H, D] BEFORE transpose for easier RoPE
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim) # ⭐ Fix: 使用 num_key_value_heads
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply rotary embeddings (Standard way: apply on [B, S, H, D])
        if cos_sin is not None:
            cos, sin = cos_sin
            # No need to manually reshape cos/sin if apply_rotary_pos_emb follows standard implementation
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Transpose for Attention: [B, H, S, D]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Repeat K/V if needed (for GQA)
        if self.num_key_value_heads != self.num_heads: # ⭐ Fix: 使用 num_key_value_heads
            key_states = torch.repeat_interleave(key_states, self.num_heads // self.num_key_value_heads, dim=1)
            value_states = torch.repeat_interleave(value_states, self.num_heads // self.num_key_value_heads, dim=1)
        
        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if not self.causal:
            attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            # Causal mask
            mask = torch.triu(torch.ones_like(attn_weights), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to [B, S, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Apply gate if enabled
        if self.gate_mode == 'headwise':
            # gate_score: [B, S, H] -> [B, S, H, 1]
            gate_score = gate_score.unsqueeze(-1)
            gate_score = torch.sigmoid(gate_score)
            
            # Reshape attn_output to apply gate per head
            attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim)
            attn_output = (attn_output_reshaped * gate_score).view(batch_size, seq_len, self.hidden_size)
            
        elif self.gate_mode == 'elementwise':
            # gate_score: [B, S, H*D]
            gate_score = torch.sigmoid(gate_score)
            attn_output = attn_output * gate_score
        
        # Output projection
        return self.o_proj(attn_output)

class RecurrenceGateCell(nn.Module):
    """
    GRU-style gating for recurrence updates.
    
    Outer loop gating: controls whether to accept state updates
    
    Args:
        hidden_size: Dimension of hidden states
        context_dim: Dimension of context
        use_reset_gate: Whether to use reset gate (full GRU) or just update gate
    """
    def __init__(self, hidden_size: int, context_dim: int = 0, use_reset_gate: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        self.use_reset_gate = use_reset_gate
        
        # Input: current + candidate + context
        input_size = hidden_size * 2 + context_dim
        
        self.update_gate = CastedLinear(input_size, hidden_size, bias=True)
        
        if use_reset_gate:
            self.reset_gate = CastedLinear(input_size, hidden_size, bias=True)
        
        # Initialize
        with torch.no_grad():
            self.update_gate.bias.fill_(0.0)  # Start neutral
            if use_reset_gate:
                self.reset_gate.bias.fill_(1.0)  # Start open
    
    def forward(self, current_state: torch.Tensor, candidate_state: torch.Tensor, 
                context: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            current_state: [B, L, D]
            candidate_state: [B, L, D]
            context: [B, L, D'] (optional)
        Returns:
            new_state: [B, L, D]
        """
        if context is not None:
            gate_input = torch.cat([current_state, candidate_state, context], dim=-1)
        else:
            gate_input = torch.cat([current_state, candidate_state], dim=-1)
        
        update_gate = torch.sigmoid(self.update_gate(gate_input))
        
        if self.use_reset_gate:
            reset_gate = torch.sigmoid(self.reset_gate(gate_input))
            gated_current = reset_gate * current_state
            new_state = (1 - update_gate) * gated_current + update_gate * candidate_state
        else:
            new_state = (1 - update_gate) * current_state + update_gate * candidate_state
        
        return new_state