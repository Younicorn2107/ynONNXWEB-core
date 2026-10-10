/** Turn what the caller asked for into an ordered list of backends to try. */

import { ProviderError } from "../errors.js";
import type { Logger } from "../logger.js";
import {
  AUTO_EXCLUDED,
  BACKEND_CHAIN,
  BACKEND_LABELS,
  resolveAliases,
  type Backend,
  type BackendSpec,
} from "./chain.js";
import type { Capabilities } from "./capabilities.js";

export interface BackendPlan {
  /** In the order they will be tried. Always ends with `wasm`. */
  order: Backend[];
  /** Why each dropped backend was dropped, for a message the user can act on. */
  dropped: { backend: Backend; reason: string }[];
  /** WASM threads this page may use. 1 means the slow path. */
  threads: number;
}

function unavailableReason(backend: Backend, caps: Capabilities): string | null {
  switch (backend) {
    case "webnn":
      return caps.webnn ? null : (caps.webnnReason ?? "WebNN is unavailable");
    case "webgpu":
      return caps.webgpu ? null : (caps.webgpuReason ?? "WebGPU is unavailable");
    case "wasm":

      return null;
    case "webgl":
      return caps.webgpu === false && !("WebGLRenderingContext" in globalThis)
        ? "no WebGL context in this browser"
        : null;
  }
}

/** Build the ordered plan. */
export function selectBackends(
  spec: BackendSpec,
  caps: Capabilities,
  logger?: Logger,
): BackendPlan {
  const dropped: { backend: Backend; reason: string }[] = [];
  const wanted: Backend[] =
    spec === "auto" ? [...BACKEND_CHAIN] : resolveAliases(spec);
  const explicit = spec !== "auto";

  const order: Backend[] = [];
  for (const backend of wanted) {
    if (!explicit && AUTO_EXCLUDED[backend]) continue;

    const reason = unavailableReason(backend, caps);
    if (reason) {
      dropped.push({ backend, reason });
      logger?.warn(`${BACKEND_LABELS[backend]} is unavailable: ${reason}`);
      continue;
    }

    if (caps.usable.length > 0 && !caps.usable.includes(backend)) {
      const why = "a probe graph would not run on it";
      dropped.push({ backend, reason: why });
      logger?.warn(`${BACKEND_LABELS[backend]} is unavailable: ${why}`);
      continue;
    }
    if (!order.includes(backend)) order.push(backend);
  }

  if (!order.includes("wasm")) order.push("wasm");

  if (order.length === 1 && order[0] === "wasm" && explicit && !wanted.includes("wasm")) {
    const asked = wanted.map((b) => BACKEND_LABELS[b]).join(", ");
    logger?.warn(
      `none of the requested backends (${asked}) can be used here; falling back to CPU`,
    );
  }

  if (caps.wasmThreads === 1 && order.includes("wasm")) {
    logger?.warn?.(
      "WebAssembly will run single-threaded (SharedArrayBuffer is unavailable — " +
        "this page is not cross-origin isolated). Set Cross-Origin-Opener-Policy: " +
        "same-origin and Cross-Origin-Embedder-Policy: require-corp to get " +
        "multi-threading; it is worth several times the CPU throughput.",
    );
  }

  return { order, dropped, threads: caps.wasmThreads };
}

/** Validate names without touching capabilities — for early argument checks. */
export function assertKnownBackends(spec: BackendSpec): Backend[] {
  if (spec === "auto") return [...BACKEND_CHAIN];
  const resolved = resolveAliases(spec);
  if (resolved.length === 0) {
    throw new ProviderError("no backend was named; pass 'auto' or at least one name");
  }
  return resolved;
}
