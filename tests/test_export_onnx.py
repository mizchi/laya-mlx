import json

import numpy as np
import pytest

from laya_mlx import Agent
from laya_mlx.agent import collate_items

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxscript")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from laya_mlx.export_onnx import (  # noqa: E402
    INPUT_NAMES,
    OUTPUT_NAMES,
    export_onnx,
    torch_weight_name,
)


def test_torch_weight_name_inverts_mlx_sanitization():
    from laya_mlx.model import sanitize_weights

    torch_names = [
        "encoder.layers.0.attn.Wqkv.weight",
        "head.layers.0.self_attn.in_proj_weight",
        "head.layers.0.self_attn.in_proj_bias",
        "head.layers.0.self_attn.out_proj.weight",
        "scorer.0.weight",
        "scorer.3.bias",
        "act_head.2.weight",
        "type_emb.weight",
        "temperature",
    ]
    mlx_names = list(sanitize_weights(dict.fromkeys(torch_names)))
    assert [torch_weight_name(name) for name in mlx_names] == torch_names


def _ort_feeds(batch):
    feeds = {
        "input_ids": batch["input_ids"].astype(np.int64),
        "attention_mask": batch["attention_mask"].astype(np.int64),
        "marker_pos": batch["marker_pos"].astype(np.int64),
        "marker_mask": batch["marker_mask"],
        "qtype": batch["qtype"].astype(np.int64),
    }
    assert list(feeds) == INPUT_NAMES
    return feeds


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_export_matches_mlx_and_writes_bundle(tiny_checkpoint, questions, tmp_path, dtype):
    output = export_onnx(tiny_checkpoint, tmp_path / "onnx", dtype=dtype)
    assert output == tmp_path / "onnx"
    for name in ("model.onnx", "rl_agent_config.json", "onnx_config.json"):
        assert (output / name).is_file()
    assert (output / "tokenizer/tokenizer.json").is_file()
    meta = json.loads((output / "onnx_config.json").read_text())
    assert meta["dtype"] == dtype
    assert meta["inputs"] == INPUT_NAMES and meta["outputs"] == OUTPUT_NAMES

    model = onnx.load(str(output / "model.onnx"))
    onnx.checker.check_model(model)
    assert {node.domain for node in model.graph.node} <= {"", "ai.onnx"}
    element_types = {t.data_type for t in model.graph.initializer if len(t.dims) == 2}
    expected = onnx.TensorProto.FLOAT16 if dtype == "float16" else onnx.TensorProto.FLOAT
    assert expected in element_types

    agent = Agent(tiny_checkpoint, dtype="float32", device="cpu")
    items, _ = agent.prepare("hello hello", questions)
    items[0]["ids"] = items[0]["ids"][:-3]  # force padding in the batch
    batch = collate_items(items, agent.tok.pad_token_id)
    logits, act = (np.asarray(x, dtype=np.float32) for x in agent.forward(batch))

    session = ort.InferenceSession(str(output / "model.onnx"), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == INPUT_NAMES
    ort_logits, ort_act = session.run(OUTPUT_NAMES, _ort_feeds(batch))
    assert ort_logits.dtype == np.float32 and ort_act.dtype == np.float32
    tol = 5e-2 if dtype == "float16" else 1e-4
    valid = batch["marker_mask"]
    np.testing.assert_allclose(ort_logits[valid], logits[valid], atol=tol, rtol=tol)
    np.testing.assert_allclose(ort_act, act, atol=tol, rtol=tol)


def test_export_refuses_existing_output(tiny_checkpoint, tmp_path):
    (tmp_path / "onnx").mkdir()
    with pytest.raises(FileExistsError):
        export_onnx(tiny_checkpoint, tmp_path / "onnx")


def test_export_rejects_unknown_dtype(tiny_checkpoint, tmp_path):
    with pytest.raises(ValueError):
        export_onnx(tiny_checkpoint, tmp_path / "onnx", dtype="bfloat16")


def test_cli_export_onnx_subcommand(tiny_checkpoint, tmp_path, capsys):
    from laya_mlx.cli import main

    main(["export-onnx", "--model", str(tiny_checkpoint), "--output", str(tmp_path / "onnx")])
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"output": str(tmp_path / "onnx"), "dtype": "float32"}
    assert (tmp_path / "onnx/model.onnx").is_file()
