---
title: Laya in the browser
emoji: 🐍
colorFrom: green
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
models:
  - mizchi/laya-multilingual-onnx
---

Typed decisions from a Laya checkpoint running in onnxruntime-web (WebGPU). Source and tests:
https://github.com/mizchi/laya-mlx (`web/`). The model is fetched from
https://huggingface.co/mizchi/laya-multilingual-onnx on first load and cached by the browser.
Desktop browsers with WebGPU (Chrome, Edge) are recommended; the wasm fallback is slow.
