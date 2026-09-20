import type { Json } from "./types.ts";

export interface PyJsonOptions {
  /** Python's json.dumps default. false matches `ensure_ascii=False`. */
  ensureAscii?: boolean;
}

/**
 * Serialize like Python's `json.dumps(value, separators=(", ", ": "))`.
 *
 * Differences that cannot be reproduced from JavaScript values are documented
 * rather than hidden: integer-valued floats print as integers (Python prints
 * `1.0`), and integer-like object keys are enumerated first by JavaScript.
 */
export function pyJson(value: Json, options: PyJsonOptions = {}): string {
  return serialize(value, options.ensureAscii ?? true);
}

function serialize(value: Json, ensureAscii: boolean): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new RangeError("Non-finite numbers are not JSON");
    return String(value);
  }
  if (typeof value === "string") return serializeString(value, ensureAscii);
  if (Array.isArray(value))
    return "[" + value.map((v) => serialize(v, ensureAscii)).join(", ") + "]";
  return (
    "{" +
    Object.entries(value)
      .map(([k, v]) => serializeString(k, ensureAscii) + ": " + serialize(v, ensureAscii))
      .join(", ") +
    "}"
  );
}

function serializeString(value: string, ensureAscii: boolean): string {
  const escaped = JSON.stringify(value);
  if (!ensureAscii) return escaped;
  let out = "";
  for (let i = 0; i < escaped.length; i++) {
    const code = escaped.charCodeAt(i);
    out += code < 0x80 ? escaped[i] : "\\u" + code.toString(16).padStart(4, "0");
  }
  return out;
}
