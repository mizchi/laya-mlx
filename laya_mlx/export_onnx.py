"""Export a Laya checkpoint to ONNX for onnxruntime (WebGPU, CPU, and others).

MLX has no ONNX exporter, so the graph is traced from a PyTorch mirror of the
upstream `DecisionModel` (see NOTICE) with `torch.onnx.export`. PyTorch,
Transformers, and the ONNX toolchain are only imported here; install the
`onnx` extra to use it. The published MLX runtime remains PyTorch-free.
"""

import json
import shutil
from pathlib import Path

from .agent import resolve_model

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]
DTYPES = ("float32", "float16")
OPSET = 18


def torch_weight_name(name):
    """Map MLX checkpoint parameter names back to upstream PyTorch names."""
    name = name.replace(".in_proj.weight", ".in_proj_weight")
    name = name.replace(".in_proj.bias", ".in_proj_bias")
    for prefix in ("scorer.layers.", "act_head.layers."):
        if name.startswith(prefix):
            name = prefix[: -len("layers.")] + name[len(prefix) :]
    return name


def _torch_model_class():
    import torch
    import torch.nn as nn

    class DecisionModel(nn.Module):
        """Inference-only mirror of upstream laya.common.DecisionModel."""

        def __init__(self, encoder, head_layers, n_act):
            super().__init__()
            self.encoder = encoder
            d = encoder.config.hidden_size
            layer = nn.TransformerEncoderLayer(
                d, max(1, d // 64), 4 * d, 0.0, batch_first=True, norm_first=True
            )
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
            )
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
            self.register_buffer("temperature", torch.ones(3))

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            h = h + self.type_emb(qtype)[:, None, :]
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
            idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            logits = self.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)
            p = torch.softmax(logits, -1)
            k = marker_mask.sum(-1).clamp(min=2).float()
            entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
            top2 = p.topk(2, -1).values
            features = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], entropy, k / 255.0], -1)
            # Upstream pools in fp32; cast to the head's dtype so a half model also traces.
            pooled = torch.cat([h[:, 0].float(), features], -1)
            return logits, self.act_head(pooled.to(self.act_head[0].weight.dtype)).float()

    return DecisionModel


def load_torch_model(model_dir):
    """Build the PyTorch model from an upstream or laya-mlx checkpoint directory."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
    encoder_cfg = AutoConfig.from_pretrained(model_dir / "encoder")
    encoder_cfg.reference_compile = False
    encoder = AutoModel.from_config(encoder_cfg, attn_implementation="sdpa")
    model = _torch_model_class()(
        encoder, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1
    )
    weights = {
        torch_weight_name(k): v.to(torch.float32)
        for k, v in load_file(str(model_dir / "model.safetensors")).items()
    }
    model.load_state_dict(weights, strict=True)
    return model.eval(), cfg


def export_onnx(
    model_id_or_path,
    output,
    *,
    dtype="float32",
    opset=OPSET,
    revision=None,
    subfolder=None,
    token=None,
):
    """Write `model.onnx` plus the config and tokenizer a runtime needs next to it.

    The destination must not exist. Inputs carry dynamic batch, sequence, and
    marker axes; outputs are always float32. `float16` halves the weights but
    keeps the decision tail (scoring, softmax, action head input) in float32.
    """
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {list(DTYPES)}")
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    import torch

    model_dir = resolve_model(model_id_or_path, token=token, subfolder=subfolder, revision=revision)
    model, cfg = load_torch_model(model_dir)
    if dtype == "float16":
        model = model.half()
    example = (
        torch.randint(5, 100, (2, 24)),
        torch.ones(2, 24, dtype=torch.int64),
        torch.tensor([[3, 9, 15], [3, 9, 0]]),
        torch.tensor([[True, True, True], [True, True, False]]),
        torch.tensor([0, 2]),
    )
    example[1][1, 20:] = 0
    dynamic_shapes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "marker_pos": {0: "batch", 1: "markers"},
        "marker_mask": {0: "batch", 1: "markers"},
        "qtype": {0: "batch"},
    }
    output.mkdir(parents=True)
    try:
        program = torch.onnx.export(
            model,
            example,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            opset_version=opset,
            verbose=False,
        )
        program.optimize()
        program.save(str(output / "model.onnx"))
        shutil.copytree(model_dir / "tokenizer", output / "tokenizer")
        (output / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        metadata = {
            "format": "laya-onnx",
            "format_version": 1,
            "dtype": dtype,
            "opset": opset,
            "inputs": INPUT_NAMES,
            "outputs": OUTPUT_NAMES,
            "source": str(model_id_or_path),
            "revision": revision,
            "subfolder": subfolder,
        }
        (output / "onnx_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return output
