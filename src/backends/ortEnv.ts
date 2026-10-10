/** ONNX Runtime's global settings, which are global to the DOCUMENT. */

import type { Logger } from "../logger.js";

export interface OrtEnvSettings {
  numThreads: number;
  simd: boolean;
  /** Where the .wasm artifacts live; the Django side serves them from static. */
  wasmPaths?: string | Record<string, string>;
  proxy?: boolean;
  logLevel?: "verbose" | "info" | "warning" | "error" | "fatal";
}

interface OrtEnvLike {
  wasm: {
    numThreads?: number;
    simd?: boolean;
    wasmPaths?: string | Record<string, string>;
    proxy?: boolean;
  };
  logLevel?: string;
}

let applied: OrtEnvSettings | null = null;

/**
 * Apply the settings once. Later calls that agree are a no-op; later calls that disagree warn and
 * are ignored, because ignoring them is what ONNX Runtime does anyway — the only choice we have is
 * whether it happens quietly.
 */
export function configureOrtEnv(
  env: OrtEnvLike,
  settings: OrtEnvSettings,
  logger?: Logger,
): OrtEnvSettings {
  if (applied) {
    const differences: string[] = [];
    if (applied.numThreads !== settings.numThreads) {
      differences.push(`numThreads ${applied.numThreads} → ${settings.numThreads}`);
    }
    if (applied.simd !== settings.simd) {
      differences.push(`simd ${applied.simd} → ${settings.simd}`);
    }
    if (differences.length > 0) {
      logger?.warn(
        `onnxruntime-web is already configured for this page and its settings ` +
          `cannot be changed afterwards; ignoring ${differences.join(", ")}. ` +
          `Configure the first engine you create, or create them all the same way.`,
        { alreadyApplied: applied, requested: settings },
      );
    }
    return applied;
  }

  env.wasm.numThreads = settings.numThreads;
  env.wasm.simd = settings.simd;
  if (settings.wasmPaths !== undefined) env.wasm.wasmPaths = settings.wasmPaths;
  if (settings.proxy !== undefined) env.wasm.proxy = settings.proxy;
  if (settings.logLevel !== undefined) env.logLevel = settings.logLevel;

  applied = { ...settings };
  logger?.debug("onnxruntime-web configured for this page", { ...settings });
  return applied;
}

/** What was actually applied, or null if nothing has been yet. */
export function appliedOrtEnv(): OrtEnvSettings | null {
  return applied;
}

/** Testing seam. There is no way to un-apply it in a real page. */
export function resetOrtEnv(): void {
  applied = null;
}
