# ONNX export and WebGPU inference

`laya-mlx export-onnx` writes a self-contained ONNX bundle for Laya decision models so the same checkpoints can run outside MLX: in onnxruntime on CPU, and in the browser through onnxruntime-web with the WebGPU backend. This document records how the export works, how faithful it is, and what was measured on 2026-09-20.

## Export

```bash
uv sync --extra onnx
uv run laya-mlx export-onnx --model aac6fef/laya-multilingual-mlx --dtype float32 --output models/laya-multilingual-onnx
uv run laya-mlx export-onnx --model aac6fef/laya-multilingual-mlx --dtype float16 --output models/laya-multilingual-onnx-fp16
```

Output layout:

| File | Purpose |
|---|---|
| `model.onnx` | Encoder, decision head, scorer and action head in one graph. Opset 18, default ONNX domain only, no custom operators, no external data. |
| `rl_agent_config.json` | Upstream agent configuration: `max_len`, `head_max_len`, calibration temperatures. A runtime needs it for prompt construction and calibration. |
| `tokenizer/` | Hugging Face `tokenizer.json` and `tokenizer_config.json`. |
| `onnx_config.json` | Format metadata: dtype, opset, input and output names, source checkpoint. |

Graph contract:

| Tensor | dtype | shape |
|---|---|---|
| `input_ids` | int64 | `[batch, sequence]` |
| `attention_mask` | int64 | `[batch, sequence]` (1 = valid token) |
| `marker_pos` | int64 | `[batch, markers]` |
| `marker_mask` | bool | `[batch, markers]` |
| `qtype` | int64 | `[batch]` (0 choice, 1 score, 2 noul) |
| `logits` | float32 | `[batch, markers]`, masked slots hold `-1e4` |
| `act_logits` | float32 | `[batch, actions]` |

These are exactly the tensors `laya_mlx.agent.collate_items` produces, so the Python prompt construction, calibration and answer formatting in `laya_mlx.common` and `laya_mlx.agent` apply unchanged. `logits` must still be divided by the calibration temperature and passed through softmax; the graph does not include calibration.

MLX has no ONNX exporter. The graph is traced with `torch.onnx.export` (dynamo) from `laya_mlx.export_onnx`, a PyTorch mirror of the upstream `DecisionModel` that loads the ModernBERT encoder from Transformers and the decision layers from the checkpoint. Both upstream checkpoints and the published `-mlx` checkpoints are accepted; MLX parameter names are mapped back to PyTorch names. `--dtype float16` halves every weight but keeps the scoring output, softmax, entropy features and action head input in float32, mirroring upstream's explicit `.float()` casts.

## Fidelity

`python -m benchmarks.validate_onnx <bundle> --model aac6fef/laya-multilingual-mlx` compares the ONNX graph on the onnxruntime CPU provider with the MLX runtime on the same 16 parity cases (63 questions) used for the MLX-vs-upstream validation. Reports are stored in [benchmarks/results/onnx-validation-multilingual-float32.json](../benchmarks/results/onnx-validation-multilingual-float32.json) and [float16](../benchmarks/results/onnx-validation-multilingual-float16.json).

| Export | File | Selected answers | Max probability error | Max logit error | Max action logit error |
|---|---:|---:|---:|---:|---:|
| float32 | 1.29 GB | 63/63 | 1.3e-6 | 4.6e-5 | 2.7e-3 |
| float16 | 647 MB | 63/63 | 5.1e-4 | 2.6e-2 | 2.9 |

The float32 figures are at the same level as the published MLX-vs-PyTorch validation, so the export is a faithful copy of the runtime. The float16 action logit error looks large but does not change any `act_probability` after softmax (`action_probability_max_abs_error` is 0.0 in both reports); the action head is saturated on these fixtures.

The MLX reference runs on the CPU. On the machine used here (Apple M5, MLX 0.32.2) Metal float32 matmul has a relative error around 1e-3, which is float16-level precision; comparing against the GPU would attribute a 2e-2 logit error to the ONNX graph that actually comes from MLX. `tests/test_model.py` also fails on that GPU for the same reason. This was not observed on the M3 Max used for the published benchmarks and needs a separate investigation.

## Browser: onnxruntime-web with WebGPU

The float32 and float16 bundles were loaded in Chromium (Playwright headless, `--enable-unsafe-webgpu --use-angle=metal`) with onnxruntime-web 1.23.2 and fed the same 63 fixture batches. Inputs were prepared in Python and exported as JSON; the browser ran only the graph. Timings are the median of repeated runs on one fixed batch after warm-up; the first run of each new input shape adds roughly 100 ms of shader compilation.

| Bundle | Provider | 3 questions, 65 tokens | 20 questions, 91 tokens | 3 questions, 1,024 tokens | Selected answers | Max probability error |
|---|---|---:|---:|---:|---:|---:|
| float32 | WebGPU | 61 ms | 369 ms | – | 63/63 | 1.1e-6 |
| float16 | WebGPU | 48 ms | 270 ms | 1,039 ms | 63/63 | 1.2e-2 |
| float16 | wasm (CPU) | 901 ms | – | – | 63/63 | 9.1e-4 |

Observations:

- WebGPU float32 is exact to the same 1e-6 level as the native CPU provider.
- WebGPU float16 loses precision (1.2e-2, versus 5.1e-4 for the same file on the CPU provider), consistent with float16 accumulation in the onnxruntime-web kernels. Selected answers still agree on every question, but calibrated probabilities are visibly perturbed.
- float16 is only about 25% faster than float32 on WebGPU. For probability-sensitive use, float32 compute is affordable.
- The wasm backend is about 20x slower than WebGPU on this hardware. WebGL is not a practical target for a 322M-parameter model.
- Session creation took 0.7–5 s depending on file size. The 1.29 GB float32 file crashed the Chrome extension tab used for an earlier attempt; the headless run completed.

The browser harness (fixture dump plus a static page and Playwright runner) is not yet part of this repository. A JavaScript runtime would additionally need ports of prompt construction (`laya_mlx.common.build_sequence`, marker positions), calibration and answer formatting, plus the tokenizer via `@huggingface/tokenizers` or Transformers.js.

## Size

Parameter breakdown of the multilingual checkpoint (321.9M parameters):

| Component | Parameters | Share | float16 size |
|---|---:|---:|---:|
| `encoder.embeddings.tok_embeddings` (256,000 × 768) | 196.6M | 61% | 393 MB |
| Everything else (22 encoder layers, decision head, scorer, action head) | 125.3M | 39% | 251 MB |

The mmBERT vocabulary dominates. Weight-only quantization of the linear layers (`MatMulNBits`, 4-bit, block 32, symmetric) produced an 872 MB file with a maximum probability error of 0.13 on the fixtures, which is unusable and barely smaller. Options that respect the model are: keep float32 compute but ship float16 weights and cast at session creation; move the embedding table out of the graph and gather it on the JavaScript side from an int8 table; or use the English checkpoints, whose ModernBERT-large vocabulary (50k) makes the embedding a small fraction of the model. None of these are implemented yet.

## Environment

Apple M5, 34 GB unified memory, macOS 26.6.2, Python 3.14.2, MLX 0.32.2, PyTorch 2.14.0, Transformers 5.17.0, onnx 1.23.0, onnxruntime 1.30.0, onnxscript 0.7.2, onnxruntime-web 1.23.2, Chromium via Playwright.
