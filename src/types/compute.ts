/** The compute-type grammar, ported from `ynonnx_core.types.compute`. */

import { ComputeTypeError } from "../errors.js";
import type { Backend } from "../backends/chain.js";

export const WEIGHT_TYPES = [
  "float32",
  "float16",
  "bfloat16",
  "float8_e4m3",
  "float8_e5m2",
  "float4_e2m1",
  "int16",
  "uint16",
  "int8",
  "uint8",
  "int4",
  "uint4",
  "int2",
  "uint2",
] as const;

export const ACTIVATION_TYPES = [
  "float32",
  "float16",
  "bfloat16",
  "float8_e4m3",
  "int16",
  "uint16",
  "int8",
  "uint8",
] as const;

/** Weight precisions that are quantized, and may therefore carry a suffix. */
export const QUANTIZED_WEIGHTS: ReadonlySet<string> = new Set([
  "int16",
  "uint16",
  "int8",
  "uint8",
  "int4",
  "uint4",
  "int2",
  "uint2",
  "float8_e4m3",
  "float8_e5m2",
  "float4_e2m1",
]);

export const BITS: Readonly<Record<string, number>> = {
  float32: 32,
  float16: 16,
  bfloat16: 16,
  float8_e4m3: 8,
  float8_e5m2: 8,
  float4_e2m1: 4,
  int16: 16,
  uint16: 16,
  int8: 8,
  uint8: 8,
  int4: 4,
  uint4: 4,
  int2: 2,
  uint2: 2,
};

export interface ComputeSpec {
  readonly weights: string;
  readonly activations: string;
  /** The spelling this ecosystem writes back out. */
  readonly canonical: string;
  readonly isQuantized: boolean;
  readonly bits: number;
}

function makeSpec(weights: string, activations: string): ComputeSpec {
  const isQuantized = QUANTIZED_WEIGHTS.has(weights);
  return {
    weights,
    activations,
    canonical: isQuantized ? `${weights}_${activations}` : weights,
    isQuantized,
    bits: BITS[weights] ?? 32,
  };
}

/** `"int8_float16"` → `{ weights: "int8", activations: "float16" }`. */
export function parseComputeType(value: string): ComputeSpec {
  const text = (value ?? "").trim().toLowerCase();
  if (text.length === 0) {
    throw new ComputeTypeError("compute type is empty");
  }
  if (text === "auto" || text === "default") {
    return makeSpec("auto", "auto");
  }

  if ((WEIGHT_TYPES as readonly string[]).includes(text)) {
    const activations = QUANTIZED_WEIGHTS.has(text) ? "float32" : text;
    return makeSpec(text, activations);
  }

  const byLength = [...WEIGHT_TYPES].sort((a, b) => b.length - a.length);
  for (const weights of byLength) {
    const prefix = `${weights}_`;
    if (!text.startsWith(prefix)) continue;
    const activations = text.slice(prefix.length);
    if (!QUANTIZED_WEIGHTS.has(weights)) {
      throw new ComputeTypeError(
        `'${value}': only quantized weight types take an activation suffix; ` +
          `'${weights}' is already a float type`,
      );
    }
    if (!(ACTIVATION_TYPES as readonly string[]).includes(activations)) {
      throw new ComputeTypeError(
        `'${value}': unknown activation precision '${activations}' ` +
          `(expected one of ${ACTIVATION_TYPES.join(", ")})`,
      );
    }
    return makeSpec(weights, activations);
  }

  throw new ComputeTypeError(
    `unknown compute type '${value}'. Expected one of ${WEIGHT_TYPES.join(", ")}, ` +
      `optionally suffixed with an activation precision for quantized weights ` +
      `(for example int8_float16, int4_float32).`,
  );
}

/** Canonical spelling, so two callers writing the same thing agree. */
export function normalizeComputeType(value: string): string {
  return parseComputeType(value).canonical;
}

/** Does the grammar accept it? Says nothing about whether it runs. */
export function isSpellable(value: string): boolean {
  try {
    parseComputeType(value);
    return true;
  } catch {
    return false;
  }
}

/** What each BROWSER backend can actually execute. */
export const BACKEND_SUPPORTED: Readonly<Record<Backend, ReadonlySet<string>>> = {
  wasm: new Set([
    "float32",
    "float16",
    "int8",
    "int8_float32",
    "uint8",
    "uint8_float32",
    "int4",
    "int4_float32",
    "uint4",
    "uint4_float32",
  ]),
  webgpu: new Set([
    "float32",
    "float16",
    "int8",
    "int8_float32",
    "int8_float16",
    "uint8",
    "uint8_float32",
    "uint8_float16",
    "int4",
    "int4_float32",
    "int4_float16",
    "uint4",
    "uint4_float32",
    "uint4_float16",
    "int2",
    "int2_float16",
    "int2_float32",
  ]),
  webnn: new Set([
    "float32",
    "float16",
    "int8",
    "int8_float32",
    "int8_float16",
    "uint8",
    "uint8_float32",
  ]),
  webgl: new Set(["float32"]),
};

/**
 * Can `backend` execute `value`? Returns the canonical spelling, or throws naming who can — the
 * same shape as the Python `check_supported`.
 */
export function checkSupported(value: string, backend: Backend): string {
  const canonical = normalizeComputeType(value);
  if (canonical === "auto") return canonical;

  if (BACKEND_SUPPORTED[backend].has(canonical)) return canonical;

  const elsewhere = (Object.keys(BACKEND_SUPPORTED) as Backend[]).filter((name) =>
    BACKEND_SUPPORTED[name].has(canonical),
  );
  const who =
    elsewhere.length > 0
      ? ` It runs on: ${elsewhere.join(", ")}.`
      : " No browser backend runs it.";
  throw new ComputeTypeError(
    `'${canonical}' cannot be executed on the ${backend} backend.${who}`,
  );
}

/** Backends that could execute this type, in no particular order. */
export function backendsSupporting(value: string): Backend[] {
  const canonical = normalizeComputeType(value);
  return (Object.keys(BACKEND_SUPPORTED) as Backend[]).filter((name) =>
    BACKEND_SUPPORTED[name].has(canonical),
  );
}

/** Weight formats a backend will run, but badly, with the reason. */
export const SLOW_ON: Readonly<Record<string, Partial<Record<Backend, string>>>> = {
  int8: {
    webgpu:
      "int8 weights use MatMulInteger, which has no WebGPU kernel: every " +
      "matmul falls back to the CPU and copies both ways. int4 (MatMulNBits) " +
      "is the quantized format WebGPU actually runs — several times faster " +
      "here — and int8 is the faster choice on the wasm backend.",
  },
};

/** Warn when a precision runs on a backend that handles it poorly. */
export function warnIfSlow(
  value: string,
  backend: Backend,
  logger?: { warn: (message: string) => void },
): void {
  let weights: string;
  try {
    weights = parseComputeType(value).weights;
  } catch {
    return;
  }
  const reason = SLOW_ON[weights]?.[backend];
  if (reason) logger?.warn(`'${value}' on ${backend}: ${reason}`);
}

/** The precision to use when the caller said `auto`. */
export const AUTO_COMPUTE_TYPE: Readonly<Record<Backend, string>> = {
  webnn: "int4",
  webgpu: "int4",
  wasm: "int8",
  webgl: "float32",
};

/** The bit widths `auto` will ever pick, inclusive. */
export const AUTO_BITS_MIN = 4;
export const AUTO_BITS_MAX = 8;

/** Resolve a requested compute type against the backend that will run it. */
export function resolveComputeType(requested: string | undefined, backend: Backend): string {
  if (requested && requested !== "auto") {
    normalizeComputeType(requested);
    return requested;
  }

  const chosen = AUTO_COMPUTE_TYPE[backend] ?? "int8";
  const bits = BITS[chosen];
  if (bits !== undefined && (bits < AUTO_BITS_MIN || bits > AUTO_BITS_MAX)) {
    throw new ComputeTypeError(
      `the automatic choice for ${backend} is '${chosen}' at ${bits} bits, ` +
        `outside the ${AUTO_BITS_MIN}-${AUTO_BITS_MAX} range auto is allowed to pick`,
    );
  }
  normalizeComputeType(chosen);
  return chosen;
}
