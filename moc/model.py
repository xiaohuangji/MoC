"""LLaMA-style Transformer for MoC.

Architecture:
  - RMSNorm (FP32 reduction)
  - Rotary positional embeddings (RoPE)
  - Multi-head causal self-attention via PyTorch SDPA (FlashAttention backend)
  - Pre-norm transformer block
  - Swappable FFN (Dense / MoC / MoC+GCP)

Reference: LLaMA-style decoder-only Transformer architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoCConfig
from .ffn import build_ffn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(x, cos, sin):
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MoCConfig):
        super().__init__()
        d = config.hidden_size
        h = config.num_attention_heads
        self.num_heads = h
        self.head_dim = d // h
        self.qkv_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        cos, sin = precompute_freqs_cis(self.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rotary_emb(q, self.rope_cos, self.rope_sin)
        k = apply_rotary_emb(k, self.rope_cos, self.rope_sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).reshape(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, config: MoCConfig, ffn_type: str = "dense"):
        super().__init__()
        d = config.hidden_size
        d_ffn = config.intermediate_size
        self.attn_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.ffn = build_ffn(d, d_ffn, ffn_type=ffn_type, k=config.k)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LLaMAModel(nn.Module):
    def __init__(self, config: MoCConfig, ffn_type: str = "dense"):
        super().__init__()
        if ffn_type not in (
            "dense",
            "moc",
            "moc_gcp",
            "moc_2_8",
            "moc_post_silu_abs",
        ):
            raise ValueError(
                "ffn_type must be 'dense', 'moc', 'moc_gcp', 'moc_2_8', "
                f"or 'moc_post_silu_abs', got {ffn_type!r}"
            )
        self.config = config
        self.ffn_type = ffn_type
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, ffn_type=ffn_type)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, labels=None):
        x = self.tok_emb(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.lm_head is None:
            logits = x @ self.tok_emb.weight.t()
        else:
            logits = self.lm_head(x)
        if labels is None:
            return logits
        # Shifted causal LM loss: the logits at position t predict token t+1.
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return logits, loss


def build_model(config: MoCConfig, ffn_type: str = "dense") -> LLaMAModel:
    return LLaMAModel(config, ffn_type=ffn_type)

