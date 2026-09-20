import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { InferenceSession } from "onnxruntime-web";

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
const { OnnxRunner } = await import("../src/session.ts");

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
