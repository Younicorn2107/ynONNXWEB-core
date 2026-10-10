/** Backend selection, which is where the policy lives. */

import { describe, expect, it, vi } from "vitest";

import { ProviderError } from "../../src/errors.js";
import { AUTO_EXCLUDED, BACKEND_CHAIN, resolveAliases } from "../../src/backends/chain.js";
import { selectBackends } from "../../src/backends/select.js";
import {
  AUTO_BITS_MAX,
  AUTO_BITS_MIN,
  AUTO_COMPUTE_TYPE,
  BITS,
  SLOW_ON,
  checkSupported,
  resolveComputeType,
  warnIfSlow,
} from "../../src/types/compute.js";
import type { Capabilities } from "../../src/backends/capabilities.js";

function caps(overrides: Partial<Capabilities> = {}): Capabilities {
  return {
    secureContext: true,
    crossOriginIsolated: true,
    wasmThreads: 4,
    memory64: false,
    webgpu: true,
    webgpuLimits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31 },
    webgpuReason: null,
    webnn: true,
    webnnDeviceTypes: ["npu"],
    webnnReason: null,
    cacheApi: true,
    opfs: true,
    storageEstimate: true,
    usable: [],
    ...overrides,
  };
}

const logger = () => ({ debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() });

describe("the chain is NPU → GPU → CPU", () => {
  it("declares that order", () => {
    expect(BACKEND_CHAIN).toEqual(["webnn", "webgpu", "wasm"]);
  });

  it("uses it when everything is available", () => {
    expect(selectBackends("auto", caps()).order).toEqual(["webnn", "webgpu", "wasm"]);
  });

  it("never puts webgl in the automatic chain", () => {
    expect(selectBackends("auto", caps()).order).not.toContain("webgl");
    expect(AUTO_EXCLUDED.webgl).toBeTruthy();
  });
});

describe("unavailable backends are dropped, and named", () => {
  it("drops WebNN and says why", () => {
    const log = logger();
    const plan = selectBackends(
      "auto",
      caps({ webnn: false, webnnReason: "navigator.ml is undefined — no WebNN in this browser" }),
      log,
    );
    expect(plan.order).toEqual(["webgpu", "wasm"]);
    expect(plan.dropped).toEqual([
      { backend: "webnn", reason: "navigator.ml is undefined — no WebNN in this browser" },
    ]);
    expect(log.warn).toHaveBeenCalled();
  });

  it("distinguishes no-WebGPU from no-adapter", () => {
    const missing = selectBackends("auto", caps({ webgpu: false, webgpuReason: "navigator.gpu is undefined" }));
    const noAdapter = selectBackends(
      "auto",
      caps({ webgpu: false, webgpuReason: "requestAdapter() returned null" }),
    );
    expect(missing.dropped[0]?.reason).not.toBe(noAdapter.dropped[0]?.reason);
  });

  it("falls all the way to CPU when nothing else works", () => {
    expect(selectBackends("auto", caps({ webnn: false, webgpu: false })).order).toEqual(["wasm"]);
  });
});

describe("CPU is always the floor", () => {
  it("is appended even when the caller did not ask for it", () => {
    expect(selectBackends("webgpu", caps()).order).toEqual(["webgpu", "wasm"]);
  });

  it("is not duplicated when the caller did ask", () => {
    expect(selectBackends(["webgpu", "wasm"], caps()).order).toEqual(["webgpu", "wasm"]);
  });

  it("survives every backend being unavailable", () => {
    const plan = selectBackends("webnn", caps({ webnn: false, webgpu: false }));
    expect(plan.order).toEqual(["wasm"]);
  });
});

describe("explicit requests", () => {
  it("are honoured in the caller's order, not the chain's", () => {
    expect(selectBackends(["webgpu", "webnn"], caps()).order).toEqual(["webgpu", "webnn", "wasm"]);
  });

  it("can select a backend that auto excludes", () => {
    expect(selectBackends("webgl", caps()).order).toContain("webgl");
  });

  it("accept CSV, the way the pack's CLIs pass them", () => {
    expect(selectBackends("gpu,cpu", caps()).order).toEqual(["webgpu", "wasm"]);
  });

  it("resolve aliases", () => {
    expect(resolveAliases("npu,gpu,cpu")).toEqual(["webnn", "webgpu", "wasm"]);
  });

  it("refuse an unknown name instead of guessing", () => {
    expect(() => selectBackends("webgpuu", caps())).toThrow(ProviderError);
    expect(() => resolveAliases("cuda")).toThrow(/not a backend/);
  });
});

describe("the probe outranks the capability flags", () => {
  it("drops a backend the probe could not run", () => {
    const plan = selectBackends("auto", caps({ usable: ["webgpu", "wasm"] }));
    expect(plan.order).toEqual(["webgpu", "wasm"]);
    expect(plan.dropped[0]?.backend).toBe("webnn");
  });

  it("is ignored when the probe has not run", () => {
    expect(selectBackends("auto", caps({ usable: [] })).order).toContain("webnn");
  });
});

describe("the single-thread warning", () => {
  it("fires whenever WASM is reachable and threads are unavailable", () => {
    const log = logger();
    selectBackends("auto", caps({ wasmThreads: 1, crossOriginIsolated: false }), log);
    const said = log.warn.mock.calls.flat().join(" ");

    expect(said).toContain("Cross-Origin-Opener-Policy");
    expect(said).toContain("Cross-Origin-Embedder-Policy");
  });

  it("stays quiet when the page is isolated", () => {
    const log = logger();
    selectBackends("auto", caps({ wasmThreads: 4 }), log);
    const said = log.warn.mock.calls.flat().join(" ");
    expect(said).not.toContain("single-threaded");
  });

  it("reports the thread count it settled on", () => {
    expect(selectBackends("auto", caps({ wasmThreads: 1 })).threads).toBe(1);
    expect(selectBackends("auto", caps({ wasmThreads: 4 })).threads).toBe(4);
  });
});

describe("the fallback warning", () => {
  it("stays quiet when CPU is what was asked for", () => {
    const logger = { warn: vi.fn(), info: vi.fn(), debug: vi.fn(), error: vi.fn() };
    selectBackends("wasm", caps({ webgpu: false, webnn: false }), logger as never);
    expect(logger.warn.mock.calls.map(String).join(" ")).not.toContain("falling back");
  });

  it("still fires when the requested backend really is gone", () => {
    const logger = { warn: vi.fn(), info: vi.fn(), debug: vi.fn(), error: vi.fn() };
    selectBackends("webgpu", caps({ webgpu: false, webnn: false }), logger as never);
    expect(logger.warn.mock.calls.map(String).join(" ")).toContain("falling back to CPU");
  });
});

describe("precisions that run badly", () => {
  it("warns about int8 on WebGPU and stays quiet on wasm", () => {
    const logger = { warn: vi.fn() };
    warnIfSlow("int8", "webgpu", logger);

    warnIfSlow("int8_float16", "webgpu", logger);
    expect(logger.warn.mock.calls.length).toBe(2);
    expect(logger.warn.mock.calls.map(String).join(" ")).toContain("MatMulInteger");

    const quiet = { warn: vi.fn() };
    warnIfSlow("int8", "wasm", quiet);
    warnIfSlow("int4", "webgpu", quiet);
    warnIfSlow("not-a-type", "webgpu", quiet);
    expect(quiet.warn).not.toHaveBeenCalled();
  });

  it("does not contradict what checkSupported allows", () => {
    for (const [weights, backends] of Object.entries(SLOW_ON)) {
      for (const backend of Object.keys(backends) as (keyof typeof backends)[]) {
        expect(() => checkSupported(weights, backend)).not.toThrow();
      }
    }
  });
});

describe("auto picks a precision from the backend", () => {
  it("chooses int4 for the GPU paths and int8 for the CPU", () => {
    expect(resolveComputeType("auto", "webgpu")).toBe("int4");
    expect(resolveComputeType("auto", "webnn")).toBe("int4");
    expect(resolveComputeType("auto", "wasm")).toBe("int8");
  });

  it("never picks below 4 bits or above 8", () => {
    for (const backend of ["webnn", "webgpu", "wasm"] as const) {
      const bits = BITS[AUTO_COMPUTE_TYPE[backend]];
      expect(bits).toBeGreaterThanOrEqual(AUTO_BITS_MIN);
      expect(bits).toBeLessThanOrEqual(AUTO_BITS_MAX);
    }
  });

  it("honours an explicit choice, even a bad one", () => {
    expect(resolveComputeType("int8", "webgpu")).toBe("int8");
    expect(resolveComputeType("float16", "wasm")).toBe("float16");

    expect(resolveComputeType("int4", "webgpu")).toBe("int4");
  });
});

describe("the WebAssembly runtime", () => {
  it("is the 32-bit build, whatever the browser supports, until a 64-bit one ships", async () => {
    const { wasmRuntime, WASM_HEAP_CEILING_MB } = await import("../../src/backends/memory.js");
    expect(wasmRuntime(true)).toEqual({ addressing: "wasm32", heapCeilingMb: WASM_HEAP_CEILING_MB, wasmPaths: null });
  });

  it("detects Memory64 by validating a module that declares it", async () => {
    const { detectMemory64 } = await import("../../src/backends/capabilities.js");

    expect(typeof detectMemory64()).toBe("boolean");
  });
});
