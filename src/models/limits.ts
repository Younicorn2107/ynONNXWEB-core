/** The sizes a download, a model or a tensor may take, in one place. */

import { ManifestError } from "../errors.js";

export interface Limits {
  /** The largest single file (one buffer): a graph, a sidecar or one part of it, a tokenizer. */
  maxFileBytes: number;
  /** The largest model: every file it needs, together. */
  maxModelBytes: number;
  /** The largest single tensor a graph may declare. */
  maxTensorBytes: number;
}

const GIB = 2 ** 30;

export const DEFAULT_LIMITS: Readonly<Limits> = Object.freeze({
  maxFileBytes: 4 * GIB,
  maxModelBytes: 32 * GIB,
  maxTensorBytes: 4 * GIB,
});

/** The defaults with the caller's overrides, each checked to be a size. */
export function resolveLimits(overrides?: Partial<Limits>): Limits {
  const limits = { ...DEFAULT_LIMITS, ...(overrides ?? {}) };
  for (const [name, value] of Object.entries(limits)) {
    if (!isSize(value) || value === 0) throw new ManifestError(`limits.${name} must be a positive number of bytes, not ${String(value)}`);
  }
  return limits;
}

/** Whether `value` can be a number of bytes: a finite, non-negative integer. */
export function isSize(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

/**
 * `value` when it is a size within `max`; otherwise the error `make` builds (each caller throws its
 * own class: a download says `IntegrityError`, a graph `ManifestError`).
 */
export function checkSize(value: unknown, max: number, what: string, make: (message: string) => Error = (m) => new ManifestError(m)): number {
  if (!isSize(value)) throw make(`${what}: ${String(value)} is not a size`);
  if (value > max) throw make(`${what}: ${value} bytes is over the limit of ${max}`);
  return value;
}
