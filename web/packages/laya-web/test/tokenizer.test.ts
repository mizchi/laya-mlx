import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";

import { LayaTokenizer } from "../src/tokenizer.ts";
import { fixture } from "./fixtures.ts";

const dir = process.env.LAYA_TOKENIZER_DIR;

describe.skipIf(!dir)("LayaTokenizer (needs LAYA_TOKENIZER_DIR=<bundle>/tokenizer)", () => {
  let tokenizer: LayaTokenizer;
  beforeAll(() => {
    tokenizer = new LayaTokenizer(
      JSON.parse(readFileSync(`${dir}/tokenizer.json`, "utf8")),
      JSON.parse(readFileSync(`${dir}/tokenizer_config.json`, "utf8")),
    );
  });
  it("exposes the special token ids the checkpoint declares", () => {
    expect(tokenizer.clsTokenId).toBe(fixture.special_tokens.cls);
    expect(tokenizer.sepTokenId).toBe(fixture.special_tokens.sep);
    expect(tokenizer.padTokenId).toBe(fixture.special_tokens.pad);
    expect(tokenizer.maskTokenId).toBe(fixture.special_tokens.mask);
    expect(tokenizer.maskToken).toBe(fixture.special_tokens.mask_token);
  });
  it.each(fixture.tokenizer_cases)("encodes $text like Rust tokenizers", ({ text, ids }) => {
    expect(tokenizer.encode(text)).toEqual(ids);
  });
});
