# C4 Configs

These YAML files provide the C4 pretraining defaults for the four public model presets. The executable preset definitions live in `moc/config.py`; the YAML files repeat the main shape fields so a run configuration is readable on its own.

| File | Preset | Total batch | Micro batch | Sequence length | Steps | Approx. allocated tokens | Learning rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama_60m_c4.yaml` | `60m` | 512 | 256 | 256 | 10,000 | 1.31B | 2.5e-3 |
| `llama_130m_c4.yaml` | `130m` | 512 | 256 | 256 | 20,000 | 2.62B | 2.5e-3 |
| `llama_350m_c4.yaml` | `350m` | 512 | 128 | 256 | 59,510 | 7.80B | 1.0e-3 |
| `llama_1b_c4.yaml` | `1b` | 512 | 64 | 256 | 99,945 | 13.10B | 6.0e-4 |

All configs use local C4, a local `t5-base` tokenizer, per-document padding/truncation to length `256`, model seed `0`, C4 shuffle seed `42`, and cosine decay to `0`.

The `1b` preset uses `num_hidden_layers: 24` and `num_attention_heads: 32`, matching the executable configuration used by the codebase.
