import { Tokenizer } from "@huggingface/tokenizers";

interface AddedToken {
  id: number;
  content: string;
  lstrip: boolean;
  rstrip: boolean;
}

interface TokenizerJson {
  added_tokens: AddedToken[];
  normalizer: { type: string } | null;
  pre_tokenizer: {
    type: string;
    replacement?: string;
    prepend_scheme?: string;
    split?: boolean;
  } | null;
}

interface TokenizerConfig {
  cls_token?: string | { content: string };
  sep_token?: string | { content: string };
  pad_token?: string | { content: string };
  mask_token?: string | { content: string };
}

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const isWhitespace = (ch: string) => /\p{White_Space}/u.test(ch);

interface AddedTokenInfo {
  id: number;
  lstrip: boolean;
  rstrip: boolean;
}

/**
 * Encoder matching the Rust `tokenizers` crate for Metaspace BPE tokenizers.
 *
 * Added tokens are matched on the raw text in two phases, mirroring Rust's
 * `tokenizers` crate: phase 1 finds every added-token CONTENT match (leftmost,
 * longest-content-first, non-overlapping); phase 2 then extends each match's
 * lstrip/rstrip over adjacent whitespace, but never past a neighboring match's
 * own (possibly already-extended) boundary. This is what keeps e.g. the `\n`
 * added token from being swallowed by `<mask>`'s lstrip in `"\n<mask>"`.
 * Each remaining gap segment gets the checkpoint's normalizer (" " -> "▁") and
 * the Metaspace `always` prepend, then is split so every piece starts with "▁"
 * (Rust's MergedWithNext). Pieces contain neither added tokens nor consecutive
 * "▁", which is where tokenizers.js diverges from Rust.
 */
export class LayaTokenizer {
  readonly clsToken: string;
  readonly sepToken: string;
  readonly padToken: string;
  readonly maskToken: string;
  readonly clsTokenId: number;
  readonly sepTokenId: number;
  readonly padTokenId: number;
  readonly maskTokenId: number;
  private readonly inner: Tokenizer;
  private readonly contentPattern: RegExp;
  private readonly addedTokenInfo: Map<string, AddedTokenInfo>;
  private readonly replacement: string;
  private readonly pieceSplitter: RegExp;

  constructor(tokenizerJson: TokenizerJson, tokenizerConfig: TokenizerConfig) {
    const pre = tokenizerJson.pre_tokenizer;
    if (pre?.type !== "Metaspace" || pre.prepend_scheme !== "always" || pre.split !== true) {
      throw new Error(
        "LayaTokenizer supports Metaspace(prepend_scheme=always, split=true) tokenizers",
      );
    }
    this.replacement = pre.replacement ?? "▁";
    // The library's constructor takes untyped `Object`s; its own TokenizerJson/
    // TokenizerConfig types describe a similar but not identical shape (e.g. no
    // `split` on Metaspace), so we pass our already-parsed JSON straight through.
    this.inner = new Tokenizer(tokenizerJson, tokenizerConfig);
    const added = [...tokenizerJson.added_tokens].sort(
      (a, b) => b.content.length - a.content.length,
    );
    this.addedTokenInfo = new Map(
      added.map((a) => [a.content, { id: a.id, lstrip: a.lstrip, rstrip: a.rstrip }]),
    );
    // Content only, no \s* here: lstrip/rstrip are applied as a second pass
    // once every content match's position is known (see `matchAddedTokens`).
    this.contentPattern = new RegExp(added.map((a) => escapeRegExp(a.content)).join("|"), "gu");
    this.pieceSplitter = new RegExp(
      `${escapeRegExp(this.replacement)}[^${escapeRegExp(this.replacement)}]*`,
      "gu",
    );
    const special = (name: keyof TokenizerConfig): [string, number] => {
      const raw = tokenizerConfig[name];
      const content = typeof raw === "string" ? raw : raw?.content;
      const id = content === undefined ? undefined : this.addedTokenInfo.get(content)?.id;
      if (content === undefined || id === undefined)
        throw new Error(`Tokenizer is missing a valid ${name}`);
      return [content, id];
    };
    [this.clsToken, this.clsTokenId] = special("cls_token");
    [this.sepToken, this.sepTokenId] = special("sep_token");
    [this.padToken, this.padTokenId] = special("pad_token");
    [this.maskToken, this.maskTokenId] = special("mask_token");
  }

  /** Token ids without special tokens, equal to Python `tok(text, add_special_tokens=False)`. */
  encode(text: string): number[] {
    const ids: number[] = [];
    let last = 0;
    for (const match of this.matchAddedTokens(text)) {
      ids.push(...this.encodeSegment(text.slice(last, match.start)));
      ids.push(match.id);
      last = match.end;
    }
    ids.push(...this.encodeSegment(text.slice(last)));
    return ids;
  }

  /**
   * Phase 1: find every added-token content match, leftmost first (ties broken
   * by content length via `contentPattern`'s alternation order), non-overlapping.
   * Phase 2: extend each match's [start, end) over adjacent whitespace per its
   * lstrip/rstrip, bounded by the neighboring match so extensions never cross
   * into another added token's content or its own extension.
   */
  private matchAddedTokens(text: string): { start: number; end: number; id: number }[] {
    const raw: { start: number; end: number; info: AddedTokenInfo }[] = [];
    for (const m of text.matchAll(this.contentPattern)) {
      const info = this.addedTokenInfo.get(m[0]);
      if (info === undefined)
        throw new Error("unreachable: contentPattern matched unknown content");
      raw.push({ start: m.index, end: m.index + m[0].length, info });
    }
    const extended: { start: number; end: number; id: number }[] = [];
    let prevEnd = 0;
    for (let i = 0; i < raw.length; i++) {
      const { start: rawStart, end: rawEnd, info } = raw[i]!;
      let start = rawStart;
      let end = rawEnd;
      if (info.lstrip) {
        while (start > prevEnd && isWhitespace(text[start - 1]!)) start--;
      }
      const nextStart = i + 1 < raw.length ? raw[i + 1]!.start : text.length;
      if (info.rstrip) {
        while (end < nextStart && isWhitespace(text[end]!)) end++;
      }
      extended.push({ start, end, id: info.id });
      prevEnd = end;
    }
    return extended;
  }

  private encodeSegment(segment: string): number[] {
    if (segment === "") return [];
    let normalized = segment.replaceAll(" ", this.replacement);
    if (!normalized.startsWith(this.replacement)) normalized = this.replacement + normalized;
    const pieces = normalized.match(this.pieceSplitter) ?? [];
    const ids: number[] = [];
    for (const piece of pieces) {
      ids.push(...this.inner.encode(piece, { add_special_tokens: false }).ids);
    }
    return ids;
  }
}
