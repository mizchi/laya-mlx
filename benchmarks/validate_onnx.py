"""Compare an exported ONNX bundle with the MLX runtime on the parity fixtures.

The MLX reference runs on the CPU by default: on some Apple GPUs (observed on
M5 with MLX 0.32.2) Metal float32 matmul carries ~1e-3 relative error, which
would be attributed to the ONNX graph otherwise.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

from laya_mlx import Agent
from laya_mlx.agent import collate_items

from .common import digest, distribution, environment, parity_cases, save_json, softmax

ROOT = Path(__file__).resolve().parents[1]


def ort_feeds(batch):
    return {
        "input_ids": batch["input_ids"].astype(np.int64),
        "attention_mask": batch["attention_mask"].astype(np.int64),
        "marker_pos": batch["marker_pos"].astype(np.int64),
        "marker_mask": batch["marker_mask"],
        "qtype": batch["qtype"].astype(np.int64),
    }


def validate(onnx_dir, model, *, device="cpu", provider="CPUExecutionProvider", repeats=5):
    onnx_dir = Path(onnx_dir)
    session = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=[provider])
    agent = Agent(model, dtype="float32", device=device, batch_size=64)
    cases, agree, total = [], 0, 0
    max_logit = max_act = max_prob = max_act_prob = 0.0
    for name, state, questions in parity_cases():
        items, _ = agent.prepare(state, questions)
        batch = collate_items(items, agent.tok.pad_token_id)
        logits, act = (np.asarray(x, dtype=np.float32) for x in agent.forward(batch))
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            ort_logits, ort_act = session.run(None, ort_feeds(batch))
            timings.append(time.perf_counter() - start)
        assert np.isfinite(ort_logits).all() and np.isfinite(ort_act).all()
        valid = batch["marker_mask"]
        probabilities = distribution(agent, logits, items)
        ort_probabilities = distribution(agent, ort_logits, items)
        errors = [float(np.max(np.abs(a - b))) for a, b in zip(probabilities, ort_probabilities)]
        agreement = sum(
            int(a.argmax() == b.argmax()) for a, b in zip(probabilities, ort_probabilities)
        )
        case = {
            "case": name,
            "questions": len(items),
            "sequence_length": int(batch["input_ids"].shape[1]),
            "input_sha256": digest([state, questions]),
            "logits_max_abs_error": float(np.max(np.abs(logits - ort_logits)[valid])),
            "action_logits_max_abs_error": float(np.max(np.abs(act - ort_act))),
            "probability_max_abs_error": max(errors),
            "action_probability_max_abs_error": float(
                np.max(np.abs(softmax(act) - softmax(ort_act)))
            ),
            "argmax_agreements": agreement,
            "onnx_ms_median": round(1000 * float(np.median(timings)), 2),
        }
        cases.append(case)
        agree += agreement
        total += len(items)
        max_logit = max(max_logit, case["logits_max_abs_error"])
        max_act = max(max_act, case["action_logits_max_abs_error"])
        max_prob = max(max_prob, case["probability_max_abs_error"])
        max_act_prob = max(max_act_prob, case["action_probability_max_abs_error"])
    return {
        "onnx": onnx_dir.name,
        "onnx_config": json.loads((onnx_dir / "onnx_config.json").read_text()),
        "provider": provider,
        "reference": {"model": str(model), "dtype": "float32", "device": device},
        "questions": total,
        "argmax_agreements": agree,
        "logits_max_abs_error": max_logit,
        "action_logits_max_abs_error": max_act,
        "probability_max_abs_error": max_prob,
        "action_probability_max_abs_error": max_act_prob,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_dir", type=Path, help="Directory written by `laya-mlx export-onnx`")
    parser.add_argument("--model", default="aac6fef/laya-multilingual-mlx", help="MLX reference")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(
        args.onnx_dir,
        args.model,
        device=args.device,
        provider=args.provider,
        repeats=args.repeats,
    )
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    report["environment"] = environment()
    if args.output:
        save_json(args.output, report)
    summary = {k: v for k, v in report.items() if k not in ("cases", "onnx_config", "environment")}
    print(summary)


if __name__ == "__main__":
    main()
