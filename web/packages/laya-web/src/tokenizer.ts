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

/**
 * Encoder matching the Rust `tokenizers` crate for Metaspace BPE tokenizers.
 *
 * Added tokens are matched on the raw text, longest first, honoring lstrip/rstrip.
 * Each remaining segment gets the checkpoint's normalizer (" " -> "▁") and the
 * Metaspace `always` prepend, then is split so every piece starts with "▁"
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
  private readonly addedPattern: RegExp;
  private readonly addedIds: Map<string, number>;
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
    this.addedIds = new Map(added.map((a) => [a.content, a.id]));
    this.addedPattern = new RegExp(
      added
        .map(
          (a) =>
            (a.lstrip ? "\\s*" : "") +
            "(" +
            escapeRegExp(a.content) +
            ")" +
            (a.rstrip ? "\\s*" : ""),
        )
        .join("|"),
      "gu",
    );
    this.pieceSplitter = new RegExp(
      `${escapeRegExp(this.replacement)}[^${escapeRegExp(this.replacement)}]*`,
      "gu",
    );
    const special = (name: keyof TokenizerConfig): [string, number] => {
      const raw = tokenizerConfig[name];
      const content = typeof raw === "string" ? raw : raw?.content;
      const id = content === undefined ? undefined : this.addedIds.get(content);
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
    for (const match of text.matchAll(this.addedPattern)) {
      ids.push(...this.encodeSegment(text.slice(last, match.index)));
      // Exactly one capture group matches per alternation branch in addedPattern.
      const content = match.slice(1).find((group) => group !== undefined);
      if (content === undefined) throw new Error("unreachable: no added-token group matched");
      ids.push(this.addedIds.get(content)!);
      last = match.index + match[0].length;
    }
    ids.push(...this.encodeSegment(text.slice(last)));
    return ids;
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
