/** Which backend runs the work, and in what order `auto` tries them. */

import { ProviderError } from "../errors.js";

export type Backend = "webnn" | "webgpu" | "wasm" | "webgl";

/** Every name this library recognizes, usable or not. */
export const KNOWN_BACKENDS: readonly Backend[] = ["webnn", "webgpu", "wasm", "webgl"];

/** The order `auto` walks. NPU, then GPU, then CPU. */
export const BACKEND_CHAIN: readonly Backend[] = ["webnn", "webgpu", "wasm"];

/**
 * Known, but never chosen automatically — with the reason, because a silent omission is
 * indistinguishable from a bug.
 */
export const AUTO_EXCLUDED: Readonly<Partial<Record<Backend, string>>> = {
  webgl:
    "float32 only — no MatMulNBits, so no int4 or int8, which is the quantization this targets. " +
    "Ask for it explicitly if you know your model is float32.",
};

export const BACKEND_ALIASES: Readonly<Record<string, Backend>> = {
  npu: "webnn",
  webnn: "webnn",
  gpu: "webgpu",
  webgpu: "webgpu",
  cpu: "wasm",
  wasm: "wasm",
  webgl: "webgl",
};

/** Human wording for the chain, for logs and errors. */
export const BACKEND_LABELS: Readonly<Record<Backend, string>> = {
  webnn: "NPU (WebNN)",
  webgpu: "GPU (WebGPU)",
  wasm: "CPU (WebAssembly)",
  webgl: "GPU (WebGL, legacy)",
};

export type BackendSpec = "auto" | Backend | readonly Backend[] | string;

/** Turn what the caller wrote into canonical backend names. */
export function resolveAliases(spec: BackendSpec): Backend[] {
  const items: string[] =
    typeof spec === "string"
      ? spec.split(",").map((s) => s.trim()).filter((s) => s.length > 0)
      : [...spec];

  return items.map((item) => {
    const key = item.toLowerCase();
    const resolved = BACKEND_ALIASES[key];
    if (resolved === undefined) {
      throw new ProviderError(
        `'${item}' is not a backend. Known names: ${Object.keys(BACKEND_ALIASES).join(", ")}.`,
      );
    }
    return resolved;
  });
}

export function isKnownBackend(name: string): boolean {
  return Object.prototype.hasOwnProperty.call(BACKEND_ALIASES, name.toLowerCase());
}
