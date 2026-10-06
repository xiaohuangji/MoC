# Third-Party Notices

## LLM-Adapters Prompt Templates

The training and evaluation prompt templates in
`benchmarks/finetuning/protocol.py` are adapted from
[LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), revision
`816657208af4db747803f87ba40a4c71383fed7a`, distributed under the
[Apache License 2.0](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/816657208af4db747803f87ba40a4c71383fed7a/LICENSE).
See its `finetune.py` and `commonsense_evaluate.py`. A copy of the Apache 2.0
license is included in this repository's [LICENSE](LICENSE).

The task validation, fixed-K reconstruction, resource implementation, weighted
objective, and constrained-answer decoding in this benchmark are project-specific
adaptations, not a claim of an unmodified upstream trainer. NVIDIA DoRA is a
recipe reference; its training files are not bundled here.

Model weights, the source Dense SFT adapter, and task datasets are not bundled.
Users must obtain them under their respective upstream terms. This repository's
license does not replace model or dataset licenses.
