import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { InferenceSession } from "onnxruntime-web";
import type { LoadProgress } from "../src/session.ts";

// `OnnxRunner.create` probes each provider's session with a real `run()` call before accepting
// it (see session.ts), because `InferenceSession.create` alone does not catch every backend
// problem — the WebGPU SkipLayerNormalization failure this guards against only surfaces on the
// first `run`. Mock onnxruntime-web so we can make one provider's session create fine but fail
// on `run`, and assert the runner falls through to the next provider and releases the failed one.
vi.mock("onnxruntime-web", () => {
  class FakeTensor {
    constructor(
      public type: string,
      public data: unknown,
      public dims: number[],
    ) {}
  }
  return {
    env: { wasm: {} },
    Tensor: FakeTensor,
    InferenceSession: { create: vi.fn() },
  };
});

const ort = await import("onnxruntime-web");
const { OnnxRunner, loadBundle } = await import("../src/session.ts");

interface FakeSession {
  run: ReturnType<typeof vi.fn>;
  release: ReturnType<typeof vi.fn>;
}

function fakeOkSession(): FakeSession {
  return {
    // dims[0] must match the probe batch's row count (2, see session.ts's `probeBatch`).
    run: vi.fn().mockResolvedValue({
      logits: { dims: [2, 3], data: new Float32Array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]) },
      act_logits: { dims: [2, 3], data: new Float32Array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]) },
    }),
    release: vi.fn().mockResolvedValue(undefined),
  };
}

function fakeFailingSession(message: string): FakeSession {
  return {
    run: vi.fn().mockRejectedValue(new Error(message)),
    release: vi.fn().mockResolvedValue(undefined),
  };
}

/** A provider entry is either a session to hand back from `create()`, or an error `create()` itself should reject with. */
function mockCreateByProvider(sessions: Record<string, FakeSession | Error>) {
  vi.mocked(ort.InferenceSession.create).mockImplementation(async (_model, opts) => {
    const provider = (opts as { executionProviders: string[] }).executionProviders[0]!;
    const entry = sessions[provider];
    if (!entry) throw new Error(`unexpected provider: ${provider}`);
    if (entry instanceof Error) throw entry;
    return entry as unknown as InferenceSession;
  });
}

describe("OnnxRunner.create (probe fallback)", () => {
  const originalNavigator = globalThis.navigator;

  beforeEach(() => {
    vi.mocked(ort.InferenceSession.create).mockReset();
    // `navigator.gpu` gates whether "webgpu" is even attempted (see session.ts); this repo's
    // vitest environment is "node", so stub it in for these tests.
    Object.defineProperty(globalThis, "navigator", {
      value: { gpu: {} },
      configurable: true,
      writable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(globalThis, "navigator", {
      value: originalNavigator,
      configurable: true,
      writable: true,
    });
  });

  it("falls through to the next provider when a session fails on run(), and releases it", async () => {
    const webgpu = fakeFailingSession(
      '[WebGPU] Kernel "[SkipLayerNormalization] SkipLayerNormalization" failed. Error: Beta must be 1D',
    );
    const wasm = fakeOkSession();
    mockCreateByProvider({ webgpu, wasm });

    const runner = await OnnxRunner.create(new ArrayBuffer(0), { providers: ["webgpu", "wasm"] });

    expect(runner.provider).toBe("wasm");
    expect(webgpu.run).toHaveBeenCalledTimes(1);
    expect(webgpu.release).toHaveBeenCalledTimes(1);
    expect(wasm.run).toHaveBeenCalledTimes(1);
    expect(wasm.release).not.toHaveBeenCalled();
  });

  it("falls through when InferenceSession.create itself throws, without calling release", async () => {
    const wasm = fakeOkSession();
    mockCreateByProvider({ webgpu: new Error("adapter not found"), wasm });

    const runner = await OnnxRunner.create(new ArrayBuffer(0), { providers: ["webgpu", "wasm"] });

    expect(runner.provider).toBe("wasm");
    // No session was ever created for webgpu, so there is nothing to release.
    expect(wasm.run).toHaveBeenCalledTimes(1);
    expect(wasm.release).not.toHaveBeenCalled();
  });

  it("aggregates and names every provider's error when all providers fail", async () => {
    const webgpu = fakeFailingSession("webgpu boom");
    const wasm = fakeFailingSession("wasm boom");
    mockCreateByProvider({ webgpu, wasm });

    let error: unknown;
    try {
      await OnnxRunner.create(new ArrayBuffer(0), { providers: ["webgpu", "wasm"] });
    } catch (e) {
      error = e;
    }
    expect(error).toBeInstanceOf(Error);
    const message = (error as Error).message;
    expect(message).toContain("webgpu");
    expect(message).toContain("webgpu boom");
    expect(message).toContain("wasm");
    expect(message).toContain("wasm boom");
    expect(webgpu.release).toHaveBeenCalledTimes(1);
    expect(wasm.release).toHaveBeenCalledTimes(1);
  });
});

/** Minimal, `loadBundle`-only fake `Response` for one of its three JSON fetches. */
function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

/** A streamed `Response` body, one `read()` per chunk, with an optional `content-length` header. */
function streamResponse(chunks: Uint8Array[], contentLength: number | null): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
  const headers: Record<string, string> = {};
  if (contentLength !== null) headers["content-length"] = String(contentLength);
  return new Response(body, { status: 200, headers });
}

/** `rl_agent_config.json` / `onnx_config.json` / `tokenizer/tokenizer*.json` fetch routes shared by every `loadBundle` test; only `model.onnx` differs per test. */
function baseJsonRoutes(): Record<string, () => Response> {
  return {
    "/rl_agent_config.json": () => jsonResponse({}),
    "/onnx_config.json": () =>
      jsonResponse({
        format: "laya-onnx",
        format_version: 1,
        dtype: "float32",
        opset: 17,
        inputs: [],
        outputs: [],
      }),
    "/tokenizer/tokenizer.json": () => jsonResponse({}),
    "/tokenizer/tokenizer_config.json": () => jsonResponse({}),
  };
}

/** Routes a stubbed `fetch` to a per-URL-suffix factory; throws on an unrouted URL. */
function fetchRouter(routes: Record<string, () => Response>) {
  return vi.fn(async (input: string | URL | Request) => {
    const url = String(input);
    for (const [suffix, factory] of Object.entries(routes)) {
      if (url.endsWith(suffix)) return factory();
    }
    throw new Error(`unrouted fetch in test: ${url}`);
  });
}

describe("loadBundle / fetchBytes", () => {
  const baseUrl = "https://example.test/bundle/";
  const modelUrl = `${baseUrl}model.onnx`;

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns the cached bytes on a cache hit and reports progress once", async () => {
    const cachedBytes = Uint8Array.from([1, 2, 3, 4, 5]);
    const cachePut = vi.fn();
    const cacheMatch = vi.fn(async (url: string) =>
      url === modelUrl ? new Response(cachedBytes) : undefined,
    );
    vi.stubGlobal("caches", { open: vi.fn(async () => ({ match: cacheMatch, put: cachePut })) });
    vi.stubGlobal(
      "fetch",
      fetchRouter({
        ...baseJsonRoutes(),
        "/model.onnx": () => {
          throw new Error("model.onnx must not be fetched on a cache hit");
        },
      }),
    );

    const progress: LoadProgress[] = [];
    const bundle = await loadBundle(baseUrl, { onProgress: (p) => progress.push(p) });

    expect(new Uint8Array(bundle.model)).toEqual(cachedBytes);
    expect(progress).toEqual([{ file: modelUrl, received: 5, total: 5 }]);
    expect(cachePut).not.toHaveBeenCalled();
  });

  it("streams a cache miss, reports progress per chunk, and caches the result", async () => {
    const full = Uint8Array.from({ length: 10 }, (_, i) => i);
    const cachePut = vi.fn(async () => {});
    vi.stubGlobal("caches", {
      open: vi.fn(async () => ({ match: vi.fn(async () => undefined), put: cachePut })),
    });
    vi.stubGlobal(
      "fetch",
      fetchRouter({
        ...baseJsonRoutes(),
        "/model.onnx": () => streamResponse([full.slice(0, 4), full.slice(4, 10)], 10),
      }),
    );

    const progress: LoadProgress[] = [];
    const bundle = await loadBundle(baseUrl, { onProgress: (p) => progress.push(p) });

    expect(new Uint8Array(bundle.model)).toEqual(full);
    expect(bundle.model.byteLength).toBe(10);
    expect(progress.map((p) => [p.received, p.total])).toEqual([
      [4, 10],
      [10, 10],
    ]);
    expect(cachePut).toHaveBeenCalledTimes(1);
    const [putUrl, putResponse] = cachePut.mock.calls[0] as unknown as [string, Response];
    expect(putUrl).toBe(modelUrl);
    expect(new Uint8Array(await putResponse.arrayBuffer())).toEqual(full);
  });

  it("still succeeds, without caching, when caches.open throws", async () => {
    const full = Uint8Array.from([7, 8, 9]);
    vi.stubGlobal("caches", {
      open: vi.fn(async () => {
        throw new Error("storage disabled");
      }),
    });
    vi.stubGlobal(
      "fetch",
      fetchRouter({ ...baseJsonRoutes(), "/model.onnx": () => streamResponse([full], 3) }),
    );

    const bundle = await loadBundle(baseUrl);

    expect(new Uint8Array(bundle.model)).toEqual(full);
  });

  it("rejects a download truncated below Content-Length and does not cache it", async () => {
    const partial = Uint8Array.from([1, 2, 3, 4, 5, 6]);
    const cachePut = vi.fn();
    vi.stubGlobal("caches", {
      open: vi.fn(async () => ({ match: vi.fn(async () => undefined), put: cachePut })),
    });
    vi.stubGlobal(
      "fetch",
      fetchRouter({ ...baseJsonRoutes(), "/model.onnx": () => streamResponse([partial], 10) }),
    );

    await expect(loadBundle(baseUrl)).rejects.toThrow(/Truncated download/);
    expect(cachePut).not.toHaveBeenCalled();
  });

  it("never touches caches when cacheName is null", async () => {
    const full = Uint8Array.from([1, 2, 3]);
    const cachesOpen = vi.fn();
    vi.stubGlobal("caches", { open: cachesOpen });
    vi.stubGlobal(
      "fetch",
      fetchRouter({ ...baseJsonRoutes(), "/model.onnx": () => streamResponse([full], 3) }),
    );

    const bundle = await loadBundle(baseUrl, { cacheName: null });

    expect(new Uint8Array(bundle.model)).toEqual(full);
    expect(cachesOpen).not.toHaveBeenCalled();
  });
});
