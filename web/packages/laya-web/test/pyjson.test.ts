import { describe, expect, it } from "vitest";

import { pyJson } from "../src/pyjson.ts";

describe("pyJson", () => {
  it("uses Python's default separators", () => {
    expect(pyJson({ task: "choose department", n: [1, 2] })).toBe(
      '{"task": "choose department", "n": [1, 2]}',
    );
  });
  it("keeps non-ASCII when ensureAscii is false and escapes it otherwise", () => {
    expect(pyJson({ message: "請求書" }, { ensureAscii: false })).toBe('{"message": "請求書"}');
    expect(pyJson({ message: "請求書" })).toBe('{"message": "\\u8acb\\u6c42\\u66f8"}');
    expect(pyJson("😀")).toBe('"\\ud83d\\ude00"');
  });
  it("serializes primitives like Python", () => {
    expect(pyJson(true)).toBe("true");
    expect(pyJson(null)).toBe("null");
    expect(pyJson(1.5)).toBe("1.5");
    expect(pyJson('a"b\n')).toBe('"a\\"b\\n"');
  });
});
