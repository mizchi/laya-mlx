---
license: apache-2.0
base_model: convaiinnovations/laya-multilingual
library_name: onnxruntime
tags:
  - onnx
  - onnxruntime-web
  - webgpu
  - typed-decisions
  - laya
---

# Laya multilingual — ONNX (float16)

ONNX export of [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)
(mmBERT-base encoder, 322M parameters) for onnxruntime, including onnxruntime-web with the WebGPU backend.
Produced by [laya-mlx](https://github.com/mizorewww/laya-mlx) with `laya-mlx export-onnx --dtype float16` from the
published [aac6fef/laya-multilingual-mlx](https://huggingface.co/aac6fef/laya-multilingual-mlx) checkpoint
(snapshot `f2b4faf51023039425946074e2cf1361d2db11d5`, itself converted from upstream revision
`052592a15d198d9ad47da779604259b10b47b7aa`).

| File | Purpose |
|---|---|
| `model.onnx` | Encoder, decision head, scorer and action head in one graph. Opset 18, standard operators only, float16 weights with a float32 decision tail. |
| `rl_agent_config.json` | Prompt limits (`max_len`, `head_max_len`) and calibration temperatures. |
| `tokenizer/` | Hugging Face `tokenizer.json` and `tokenizer_config.json`. |
| `onnx_config.json` | Export metadata: dtype, opset, input and output names, source checkpoint. |

Inputs: `input_ids`, `attention_mask` (int64 `[batch, sequence]`), `marker_pos` (int64 `[batch, markers]`),
`marker_mask` (bool `[batch, markers]`), `qtype` (int64 `[batch]`; 0 choice, 1 score, 2 noul). Outputs:
`logits` `[batch, markers]` and `act_logits` `[batch, actions]`, both float32. `logits` still need the
calibration temperature and a softmax. Prompt construction, calibration and answer formatting follow upstream
Laya; the reference implementation is `laya_mlx.common` / `laya_mlx.agent` and the browser port is
`@laya-mlx/web` in the repository.

Validation on the repository's 63-question parity fixtures against the MLX runtime: 63/63 selected answers
agree, maximum probability error 5.1e-4 with the onnxruntime CPU provider and 1.2e-2 with onnxruntime-web
WebGPU (float16 compute). See `docs/ONNX_EXPORT.md` in the repository for the method, the browser
measurements and the size analysis. onnxruntime-web 1.30.0 needs `graphOptimizationLevel: "basic"` on
WebGPU for this graph.

This is an independent port, not an official Convai Innovations release. Weights are Apache-2.0 as upstream.

Live demo: https://huggingface.co/spaces/mizchi/laya-web-demo (Snake, runs the model in your browser) — also
on the repository's GitHub Pages site (for the `mizchi/laya-mlx` fork: https://mizchi.github.io/laya-mlx/).
