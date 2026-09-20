"""Dump parity fixtures for the browser runtime (web/packages/laya-web).

Every case records the public inputs, the Python-prepared token sequences,
the collated batch, MLX CPU float32 logits and the public `predict` result,
so the TypeScript port can be checked token-for-token without a model.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from laya_mlx import Agent
from laya_mlx.agent import collate_items

from .common import parity_cases

ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_CASES = [
    "  leading  spaces",
    "a  b",
    "   ",
    "x ",
    " x",
    "line\nbreak",
    "tab\there",
    "\n",
    "\n\n\n",
    "\nhello",
    "hello\n",
    "\t",
    "\thello",
    "\r",
    " b",
    "1",
    "。",
    "<mask>x",
    "x<mask>y",
    "x <mask> y",
    "<mask>",
    "<mask><mask>",
    "\nx\n",
    '{"message": "hi  there"}',
    "▁literal",
    "▁▁x",
    "<2mass>x",
    "[@BOS@]x",
    "<unused0>x",
    "mixed 日本語 and English  double",
    "emoji 😀😀",
    "level 0: not urgent",
    "a.b*c+d?e^f$g{h}(i)|j[k]\\l",
    "Ich wurde zweimal für Rechnung 4411 belastet",
    "मुझसे इनवॉइस 4411",
    "С меня дважды",
    "発票4411被重复扣款，请今天退款。",
    "[MASK] <mask> hello [MASK] <mask>",
    "",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="aac6fef/laya-multilingual-mlx")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "web/fixtures/parity-multilingual.json"
    )
    args = parser.parse_args()
    agent = Agent(args.model, dtype="float32", device="cpu", batch_size=16)
    tok = agent.tok
    cases = []
    for name, state, questions in parity_cases():
        items, _ = agent.prepare(state, questions)
        batch = collate_items(items, tok.pad_token_id)
        logits, act = (np.asarray(x, dtype=np.float32) for x in agent.forward(batch))
        cases.append(
            {
                "name": name,
                "state": state,
                "questions": questions,
                "items": items,
                "batch": {k: v.astype(int).tolist() for k, v in batch.items()},
                "logits": logits.tolist(),
                "act_logits": act.tolist(),
                "result": agent.predict(state, questions),
            }
        )
    fixture = {
        "model": args.model,
        "config": agent.cfg,
        "special_tokens": {
            "cls": tok.cls_token_id,
            "sep": tok.sep_token_id,
            "pad": tok.pad_token_id,
            "mask": tok.mask_token_id,
            "mask_token": tok.mask_token,
        },
        "tokenizer_cases": [{"text": t, "ids": tok(t)["input_ids"]} for t in TOKENIZER_CASES],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"{len(cases)} cases, {sum(len(c['items']) for c in cases)} questions -> {args.output}")


if __name__ == "__main__":
    main()
