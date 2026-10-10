/** What will fit, answered from facts the browser actually reports. */

import { QuotaError } from "../errors.js";
import type { AdapterLimits } from "./capabilities.js";

/** 32-bit addressing: the heap of the WebAssembly runtime this package ships. */
export const WASM_HEAP_CEILING_MB = 4096;

export interface WasmRuntime {
  /** "wasm32" today; "wasm64" once a 64-bit build is shipped and supported here. */
  addressing: "wasm32" | "wasm64";
  heapCeilingMb: number;
  /** `wasmPaths` override for the runtime's artifacts, or null for the package's own. */
  wasmPaths: string | Record<string, string> | null;
}

/** The WebAssembly runtime to load, given what the browser supports. */
export function wasmRuntime(_memory64 = false): WasmRuntime {
  return { addressing: "wasm32", heapCeilingMb: WASM_HEAP_CEILING_MB, wasmPaths: null };
}

/**
 * Above this, a WASM session is very likely to fail — not because the model is bigger than the
 * heap, but because loading it needs the model in the heap AND the JS-side buffer at once. Warn
 * here.
 */
export const WASM_WARN_MB = 2048;

/** Above this it will not fit at all once the copy is accounted for. */
export const WASM_REFUSE_MB = 3584;

export interface StorageBudget {
  quotaBytes: number | null;
  usageBytes: number | null;
  availableBytes: number | null;
  persisted: boolean;
}

export interface FitVerdict {
  ok: boolean;
  level: "ok" | "warn" | "refuse";
  reason: string | null;
}

/** Will a model of `sizeMb` load on the WASM backend? */
export function checkWasmFit(sizeMb: number): FitVerdict {
  if (sizeMb >= WASM_REFUSE_MB) {
    return {
      ok: false,
      level: "refuse",
      reason:
        `${Math.round(sizeMb)} MB will not load on the CPU backend: the WebAssembly ` +
        `heap is 32-bit addressed (${WASM_HEAP_CEILING_MB} MB hard ceiling) and the ` +
        `model is copied into it, so it needs roughly twice its size while loading. ` +
        `Use WebGPU, which does not go through this heap, or a smaller quantization.`,
    };
  }
  if (sizeMb >= WASM_WARN_MB) {
    return {
      ok: true,
      level: "warn",
      reason:
        `${Math.round(sizeMb)} MB is close to what the WebAssembly heap can hold ` +
        `once the load-time copy is counted. It may fail on a busy page. WebGPU ` +
        `is not subject to this limit.`,
    };
  }
  return { ok: true, level: "ok", reason: null };
}

/** Does a single tensor of `bytes` fit this adapter? */
export function checkAdapterFit(bytes: number, limits: AdapterLimits | null): FitVerdict {
  if (!limits) return { ok: true, level: "ok", reason: null };
  if (bytes > limits.maxStorageBufferBindingSize) {
    return {
      ok: false,
      level: "refuse",
      reason:
        `a single ${Math.round(bytes / 1e6)} MB tensor exceeds this adapter's ` +
        `maxStorageBufferBindingSize (${Math.round(limits.maxStorageBufferBindingSize / 1e6)} MB)`,
    };
  }
  return { ok: true, level: "ok", reason: null };
}

/** What the origin is allowed to store, as far as the browser will say. */
export async function readStorageBudget(): Promise<StorageBudget> {
  const storage = globalThis.navigator?.storage as
    | {
        estimate?: () => Promise<{ quota?: number; usage?: number }>;
        persisted?: () => Promise<boolean>;
      }
    | undefined;

  if (typeof storage?.estimate !== "function") {
    return { quotaBytes: null, usageBytes: null, availableBytes: null, persisted: false };
  }
  try {
    const { quota, usage } = await storage.estimate();
    const persisted = (await storage.persisted?.()) ?? false;
    const quotaBytes = typeof quota === "number" ? quota : null;
    const usageBytes = typeof usage === "number" ? usage : null;
    return {
      quotaBytes,
      usageBytes,
      availableBytes:
        quotaBytes !== null && usageBytes !== null ? Math.max(0, quotaBytes - usageBytes) : null,
      persisted,
    };
  } catch {
    return { quotaBytes: null, usageBytes: null, availableBytes: null, persisted: false };
  }
}

/** Refuse a download that will not fit, before it starts. */
export async function assertStorageFor(totalBytes: number, margin = 1.1): Promise<StorageBudget> {
  const budget = await readStorageBudget();
  if (budget.availableBytes === null) return budget;

  const needed = totalBytes * margin;
  if (budget.availableBytes < needed) {
    const mb = (n: number) => `${Math.round(n / 1e6)} MB`;
    throw new QuotaError(
      `not enough storage: this needs about ${mb(needed)} and the origin has ` +
        `${mb(budget.availableBytes)} free of ${mb(budget.quotaBytes ?? 0)}. ` +
        `Free space, or clear cached models with the store's clear().`,
    );
  }
  return budget;
}

/** Ask the browser to keep what we store. */
export async function requestPersistence(): Promise<boolean> {
  const storage = globalThis.navigator?.storage as
    | { persist?: () => Promise<boolean>; persisted?: () => Promise<boolean> }
    | undefined;
  try {
    if ((await storage?.persisted?.()) === true) return true;
    return (await storage?.persist?.()) ?? false;
  } catch {
    return false;
  }
}
