/** What this browser can actually do — asked once, cached for the page. */

import { UnsupportedEnvironmentError } from "../errors.js";
import type { Backend } from "./chain.js";

export interface AdapterLimits {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  vendor?: string;
  architecture?: string;
}

export interface Capabilities {
  /** https or localhost. Everything below needs it. */
  secureContext: boolean;
  /** COOP + COEP are set, so SharedArrayBuffer exists. */
  crossOriginIsolated: boolean;
  /** How many WASM threads we may ask for. 1 means the slow path. */
  wasmThreads: number;
  webgpu: boolean;
  webgpuLimits: AdapterLimits | null;
  /** Why WebGPU is unavailable, when it is — the two causes differ. */
  webgpuReason: string | null;
  webnn: boolean;
  webnnDeviceTypes: string[];
  webnnReason: string | null;
  /** Storage backends present, in the order we would prefer them. */
  cacheApi: boolean;
  opfs: boolean;
  storageEstimate: boolean;
  /** Backends that a live probe actually ran a graph on. */
  usable: Backend[];
  /**
   * WebAssembly Memory64 (Chrome 133+, Firefox 134+; not Safari). Detected now; used once a 64-bit
   * build of the runtime is shipped beside the 32-bit one — see `wasmRuntime` in memory.ts.
   */
  memory64: boolean;
}

/** A module that declares one 64-bit memory: valid only where Memory64 is. */
const MEMORY64_PROBE = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x05, 0x03, 0x01, 0x04, 0x01]);

export function detectMemory64(): boolean {
  try {
    return typeof WebAssembly !== "undefined" && WebAssembly.validate(MEMORY64_PROBE);
  } catch {
    return false;
  }
}

let cached: Promise<Capabilities> | null = null;

const ORT_INSTANCE_MARKER = Symbol.for("ynonnxweb.ort.instance");

/** Warn when a second ONNX Runtime instance appears in the page. */
export function assertSingleOrtInstance(logger?: {
  warn(message: string, context?: Record<string, unknown>): void;
}): boolean {
  const scope = globalThis as unknown as Record<symbol, number | undefined>;
  const count = (scope[ORT_INSTANCE_MARKER] ?? 0) + 1;
  scope[ORT_INSTANCE_MARKER] = count;
  if (count > 1) {
    logger?.warn(
      `${count} copies of this library are loaded in one page. They each carry ` +
        `their own onnxruntime-web, so they have separate WASM heaps and their ` +
        `ort.env settings do not reach each other — memory doubles and thread ` +
        `configuration silently applies to the wrong one. Deduplicate the ` +
        `dependency, or use the bundled build OR the npm build, not both.`,
    );
    return false;
  }
  return true;
}

/** Refuse early, with one clear message, rather than five obscure ones later. */
export function assertSecureContext(): void {
  const secure = (globalThis as { isSecureContext?: boolean }).isSecureContext;
  if (secure === false) {
    throw new UnsupportedEnvironmentError(
      "this page is not a secure context, so WebGPU, SharedArrayBuffer, OPFS " +
        "and crypto.subtle are all unavailable. Serve it over https, or from " +
        "http://localhost during development.",
    );
  }
}

async function detectWebGPU(): Promise<{
  ok: boolean;
  limits: AdapterLimits | null;
  reason: string | null;
}> {
  const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
  if (!gpu) {
    return {
      ok: false,
      limits: null,
      reason: "navigator.gpu is undefined — this browser has no WebGPU (Chrome/Edge 113+, or a recent Firefox/Safari)",
    };
  }
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      return {
        ok: false,
        limits: null,
        reason:
          "navigator.gpu exists but requestAdapter() returned null — no usable adapter " +
          "(blocklisted driver, a virtual machine, or a headless browser without a GPU)",
      };
    }
    const info = (adapter as unknown as { info?: { vendor?: string; architecture?: string } }).info;
    return {
      ok: true,
      limits: {
        maxBufferSize: Number(adapter.limits.maxBufferSize),
        maxStorageBufferBindingSize: Number(adapter.limits.maxStorageBufferBindingSize),
        ...(info?.vendor !== undefined ? { vendor: info.vendor } : {}),
        ...(info?.architecture !== undefined ? { architecture: info.architecture } : {}),
      },
      reason: null,
    };
  } catch (error) {
    return { ok: false, limits: null, reason: `requestAdapter() threw: ${String(error)}` };
  }
}

async function detectWebNN(): Promise<{ ok: boolean; deviceTypes: string[]; reason: string | null }> {
  const ml = (navigator as unknown as { ml?: { createContext(o?: unknown): Promise<unknown> } }).ml;
  if (!ml) {
    return {
      ok: false,
      deviceTypes: [],
      reason: "navigator.ml is undefined — no WebNN in this browser",
    };
  }
  const found: string[] = [];
  for (const deviceType of ["npu", "gpu", "cpu"]) {
    try {
      await ml.createContext({ deviceType });
      found.push(deviceType);
    } catch {
      // Not fatal: carry on.
    }
  }
  return {
    ok: found.length > 0,
    deviceTypes: found,
    reason: found.length > 0 ? null : "navigator.ml exists but no device type accepted a context",
  };
}

/** How many WASM threads we may use. */
function detectThreads(): { isolated: boolean; threads: number } {
  const isolated =
    (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true &&
    typeof (globalThis as { SharedArrayBuffer?: unknown }).SharedArrayBuffer !== "undefined";
  if (!isolated) return { isolated: false, threads: 1 };
  const cores = (navigator as { hardwareConcurrency?: number }).hardwareConcurrency ?? 4;
  return { isolated: true, threads: Math.max(1, Math.min(cores, 4)) };
}

/**
 * Detect once, and remember. The answer cannot change during a page's life, and asking WebGPU for
 * an adapter repeatedly is not free.
 */
export function detectCapabilities(options?: { force?: boolean }): Promise<Capabilities> {
  if (cached && options?.force !== true) return cached;
  cached = (async (): Promise<Capabilities> => {
    assertSecureContext();

    const { isolated, threads } = detectThreads();
    const [gpu, nn] = await Promise.all([detectWebGPU(), detectWebNN()]);

    const storage = globalThis.navigator?.storage as
      | { getDirectory?: unknown; estimate?: unknown }
      | undefined;

    return {
      secureContext: (globalThis as { isSecureContext?: boolean }).isSecureContext !== false,
      crossOriginIsolated: isolated,
      wasmThreads: threads,
      webgpu: gpu.ok,
      webgpuLimits: gpu.limits,
      webgpuReason: gpu.reason,
      webnn: nn.ok,
      webnnDeviceTypes: nn.deviceTypes,
      webnnReason: nn.reason,
      cacheApi: typeof (globalThis as { caches?: unknown }).caches !== "undefined",
      opfs: typeof storage?.getDirectory === "function",
      storageEstimate: typeof storage?.estimate === "function",

      usable: [],
      memory64: detectMemory64(),
    };
  })();
  return cached;
}

/** Testing seam, and a way out if a caller really must re-detect. */
export function resetCapabilities(): void {
  cached = null;
}
