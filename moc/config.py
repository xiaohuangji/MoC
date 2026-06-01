"""Model configuration for MoC.

Preset configurations follow LLaMA-style 60M, 130M, 350M, and 1B model
families used by the benchmark suite. The public 60M preset is the
``d=512, d_ffn=1376, layers=8`` configuration. The 1B preset uses
24 decoder layers and 32 attention heads.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class MoCConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int = 32000
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    k: Optional[int] = None

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.k is None:
            self.k = self.hidden_size // 2
        if self.k > self.intermediate_size:
            raise ValueError(
                f"k ({self.k}) must not exceed intermediate_size "
                f"({self.intermediate_size})"
            )


PRESETS = {
    "60m": MoCConfig(
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=8,
        num_attention_heads=8,
        k=256,
    ),
    "130m": MoCConfig(
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        k=384,
    ),
    "350m": MoCConfig(
        hidden_size=1024,
        intermediate_size=2736,
        num_hidden_layers=24,
        num_attention_heads=16,
        k=512,
    ),
    "1b": MoCConfig(
        hidden_size=2048,
        intermediate_size=5461,
        num_hidden_layers=24,
        num_attention_heads=32,
        k=1024,
    ),
}


def get_config(preset: str) -> MoCConfig:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
    return PRESETS[preset]


def count_parameters(config: MoCConfig) -> int:
    d = config.hidden_size
    d_ffn = config.intermediate_size
    L = config.num_hidden_layers
    V = config.vocab_size

    emb = V * d
    attn_per_layer = 4 * d * d
    ffn_per_layer = 3 * d * d_ffn
    norm_per_layer = 2 * d
    final_norm = d
    lm_head = 0 if config.tie_word_embeddings else V * d
    return emb + L * (attn_per_layer + ffn_per_layer + norm_per_layer) + final_norm + lm_head

