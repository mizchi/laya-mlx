import { describe, expect, it } from "vitest";

import { argmax } from "../src/argmax.ts";

describe("argmax", () => {
  it("returns the first maximum on ties", () => {
    const items = [
      { label: "a", value: 1 },
      { label: "b", value: 1 },
      { label: "c", value: 0 },
    ];
    expect(argmax(items, (item) => item.value).label).toBe("a");
  });

  it("returns the single element", () => {
    expect(argmax([42], (item) => item)).toBe(42);
  });

  it("throws on empty input", () => {
    expect(() => argmax([] as number[], (item) => item)).toThrow(/empty/);
  });
});
