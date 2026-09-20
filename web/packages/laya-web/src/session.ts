import * as ort from "onnxruntime-web";

import { LayaAgent } from "./agent.ts";
import type { Runner } from "./agent.ts";
import { LayaTokenizer } from "./tokenizer.ts";
import type { TokenizerConfig, TokenizerJson } from "./tokenizer.ts";
import type { AgentConfig, Batch, RunnerOutput } from "./types.ts";

export type Provider = "webgpu" | "wasm";

export interface LoadProgress {
  file: string;
  received: number;
  total: number | null;
}

export interface LoadOptions {
  /** Execution providers to try in order. Defaults to ["webgpu", "wasm"]. */
  providers?: Provider[];
  /** Cache API bucket for model.onnx; null disables caching. Defaults to "laya-models". */
  cacheName?: string | null;
  onProgress?: (progress: LoadProgress) => void;
  batchSize?: number;
  /** Where onnxruntime-web finds its .wasm/.mjs files; required when they are not next to the page. */
  wasmPaths?: string;
}

export interface OnnxConfig {
  format: "laya-onnx";
  format_version: number;
  dtype: "float32" | "float16";
  opset: number;
  inputs: string[];
  outputs: string[];
}

export interface Bundle {
  baseUrl: string;
  config: AgentConfig;
  onnxConfig: OnnxConfig;
  tokenizerJson: TokenizerJson;
  tokenizerConfig: TokenizerConfig;
  model: ArrayBuffer;
}

const join = (base: string, name: string) => (base.endsWith("/") ? base : base + "/") + name;

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  return (await response.json()) as T;
}

/** Fetch a large file with progress; caches the bytes when a cache name is given and storage allows it. */
async function fetchBytes(
  url: string,
  cacheName: string | null,
  onProgress: LoadOptions["onProgress"],
): Promise<ArrayBuffer> {
  let cache: Cache | null = null;
  if (cacheName !== null && typeof caches !== "undefined") {
    try {
      cache = await caches.open(cacheName);
      const hit = await cache.match(url);
      if (hit) {
        const bytes = await hit.arrayBuffer();
        onProgress?.({ file: url, received: bytes.byteLength, total: bytes.byteLength });
        return bytes;
      }
    } catch {
      cache = null;
    }
  }
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || null;
  const reader = response.body.getReader();
  // Annotated as `Uint8Array<ArrayBuffer>`, not the bare `Uint8Array` (which TS now widens to
  // `Uint8Array<ArrayBufferLike>`, including `SharedArrayBuffer`), so `.buffer` below stays an
  // `ArrayBuffer` and this can be passed to `Response`/`fetchBytes`'s own `ArrayBuffer` return type.
  let bytes: Uint8Array<ArrayBuffer>;
  let received = 0;
  if (total !== null) {
    // Preallocate the full buffer up front instead of accumulating a chunk list: one allocation
    // for the whole download keeps peak memory during fetch at ~1x the model size (ORT still
    // makes its own copy when it loads the buffer into a session).
    bytes = new Uint8Array(total);
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (received + value.byteLength > total) {
        throw new Error("Response longer than Content-Length");
      }
      bytes.set(value, received);
      received += value.byteLength;
      onProgress?.({ file: url, received, total });
    }
  } else {
    const chunks: Uint8Array[] = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      onProgress?.({ file: url, received, total });
    }
    bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
  }
  if (cache) {
    try {
      await cache.put(
        url,
        new Response(bytes.subarray(0, received), {
          headers: {
            "content-length": String(received),
            "content-type": "application/octet-stream",
          },
        }),
      );
    } catch {
      // Quota exceeded or storage disabled: the model still runs, it is just fetched again next time.
    }
  }
  // `bytes` is already sized to exactly `received` bytes when the stream had no known length;
  // when it was preallocated to `total` and the stream under-delivered, slice once to trim the
  // unused tail so the returned ArrayBuffer is exactly `received` bytes.
  return received === bytes.byteLength ? bytes.buffer : bytes.buffer.slice(0, received);
}

/** Download every file of a `laya-mlx export-onnx` bundle from a directory URL. */
export async function loadBundle(baseUrl: string, options: LoadOptions = {}): Promise<Bundle> {
  const cacheName = options.cacheName === undefined ? "laya-models" : options.cacheName;
  const [config, onnxConfig, tokenizerJson, tokenizerConfig] = await Promise.all([
    fetchJson<AgentConfig>(join(baseUrl, "rl_agent_config.json")),
    fetchJson<OnnxConfig>(join(baseUrl, "onnx_config.json")),
    fetchJson<TokenizerJson>(join(baseUrl, "tokenizer/tokenizer.json")),
    fetchJson<TokenizerConfig>(join(baseUrl, "tokenizer/tokenizer_config.json")),
  ]);
  if (onnxConfig.format !== "laya-onnx") throw new Error(`Not a Laya ONNX bundle: ${baseUrl}`);
  const model = await fetchBytes(join(baseUrl, "model.onnx"), cacheName, options.onProgress);
  return { baseUrl, config, onnxConfig, tokenizerJson, tokenizerConfig, model };
}

/** Runs a Laya ONNX graph through onnxruntime-web. */
export class OnnxRunner implements Runner {
  readonly session: ort.InferenceSession;
  readonly provider: Provider;

  private constructor(session: ort.InferenceSession, provider: Provider) {
    this.session = session;
    this.provider = provider;
  }

  static async create(model: ArrayBuffer, options: LoadOptions = {}): Promise<OnnxRunner> {
    if (options.wasmPaths) ort.env.wasm.wasmPaths = options.wasmPaths;
    const providers = options.providers ?? ["webgpu", "wasm"];
    const attempts: { provider: Provider; error: unknown }[] = [];
    for (const provider of providers) {
      // `navigator` does not exist in Node, where the wasm provider is smoke-tested.
      if (provider === "webgpu" && (typeof navigator === "undefined" || !("gpu" in navigator))) {
        attempts.push({ provider, error: "unavailable (navigator.gpu missing)" });
        continue;
      }
      try {
        const session = await ort.InferenceSession.create(model, {
          executionProviders: [provider],
          // The "extended"/"all" graph transformations include a SkipLayerNormalization fusion
          // that onnxruntime-web's WebGPU (JSEP) kernel cannot run against this checkpoint's
          // fp16 weights ("Error: Beta must be 1D", reproduced with a 1-row synthetic batch
          // regardless of input shape); "basic" skips that fusion and runs correctly. wasm is
          // unaffected, so it keeps the full optimization level.
          graphOptimizationLevel: provider === "webgpu" ? "basic" : "all",
        });
        return new OnnxRunner(session, provider);
      } catch (error) {
        attempts.push({ provider, error });
      }
    }
    throw new Error(
      `No execution provider could load the model: ${attempts.map((a) => `${a.provider}: ${String(a.error)}`).join("; ")}`,
      { cause: attempts.at(-1)?.error },
    );
  }

  async run(batch: Batch): Promise<RunnerOutput> {
    const feeds = {
      input_ids: new ort.Tensor("int64", batch.inputIds, [batch.rows, batch.length]),
      attention_mask: new ort.Tensor("int64", batch.attentionMask, [batch.rows, batch.length]),
      marker_pos: new ort.Tensor("int64", batch.markerPos, [batch.rows, batch.markers]),
      marker_mask: new ort.Tensor("bool", batch.markerMask, [batch.rows, batch.markers]),
      qtype: new ort.Tensor("int64", batch.qtype, [batch.rows]),
    };
    const output = await this.session.run(feeds);
    const logits = output["logits"];
    const actLogits = output["act_logits"];
    if (!logits || !actLogits) {
      throw new Error("ONNX graph did not return logits/act_logits");
    }
    if (logits.dims[0] !== batch.rows) {
      throw new Error(
        `ONNX graph returned logits for ${logits.dims[0]} rows, expected ${batch.rows}`,
      );
    }
    return {
      rows: batch.rows,
      markers: Number(logits.dims[1]),
      actions: Number(actLogits.dims[1]),
      // `Tensor.data` is typed by element type (`Tensor.DataTypeMap['float32']`), which is
      // already `Float32Array`; the cast is only to narrow away the wider `Tensor.DataType`
      // union that `output[key]`'s untyped `Tensor` carries.
      logits: logits.data as Float32Array,
      actLogits: actLogits.data as Float32Array,
    };
  }
}

export interface LoadedAgent {
  agent: LayaAgent;
  bundle: Bundle;
  provider: Provider;
}

/** One call from bundle URL to a ready agent; what the demos use. */
export async function loadAgent(baseUrl: string, options: LoadOptions = {}): Promise<LoadedAgent> {
  const bundle = await loadBundle(baseUrl, options);
  const runner = await OnnxRunner.create(bundle.model, options);
  const tokenizer = new LayaTokenizer(bundle.tokenizerJson, bundle.tokenizerConfig);
  const agent = new LayaAgent({
    config: bundle.config,
    tokenizer,
    runner,
    ...(options.batchSize ? { batchSize: options.batchSize } : {}),
  });
  return { agent, bundle, provider: runner.provider };
}
