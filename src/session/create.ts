/** Creating a session, one backend at a time. */

import type { InferenceSession as OrtSession } from "onnxruntime-web";

import { SessionError } from "../errors.js";
import type { Logger } from "../logger.js";
import { BACKEND_LABELS, type Backend } from "../backends/chain.js";
import { configureOrtEnv } from "../backends/ortEnv.js";
import { exclusive } from "./exclusive.js";
import type { BackendPlan } from "../backends/select.js";

export interface TensorSpec {
  name: string;
  /** Absent on older onnxruntime-web builds, which expose names only. */
  type?: string;
  dims?: readonly (number | string)[];
}

export interface SessionConfig {
  graphOptimizationLevel?: "disabled" | "basic" | "extended" | "all";
  executionMode?: "sequential" | "parallel";
  enableProfiling?: boolean;
  logSeverityLevel?: number;
  /** External-data sidecars, as `{ path, data }` pairs ORT expects. */
  externalData?: { path: string; data: Uint8Array }[];
  freeDimensionOverrides?: Record<string, number>;
  /**
   * The WebNN device: `npu` first in `auto`, then `gpu`, then `cpu`. Without it ONNX Runtime asks
   * WebNN for its CPU device, which is not what "NPU first" means.
   */
  webnnDeviceType?: "npu" | "gpu" | "cpu";
  /** Declared in v1 and deliberately not read yet. */
  /**
   * `"gpu-buffer"` for everything, or a map naming the outputs to keep on the device. A map is what
   * a decoder wants: the KV cache stays put, the logits come back, because the logits are the one
   * output the loop reads.
   */
  preferredOutputLocation?: "cpu" | "gpu-buffer" | Record<string, "cpu" | "gpu-buffer">;
  /**
   * WebGPU: record the run once and replay it. Every node must run on WebGPU, every shape be fixed
   * (`freeDimensionOverrides`) and the inputs and outputs be GPU buffers; in return the hundreds of
   * kernel launches of a vision model cost one submission. Refused at creation otherwise.
   */
  enableGraphCapture?: boolean;
}

export interface OrtLike {
  InferenceSession: {
    create(model: Uint8Array | ArrayBuffer | string, options?: unknown): Promise<OrtSession>;
  };
  Tensor: unknown;
  env: {
    wasm: {
      numThreads?: number;
      simd?: boolean;
      wasmPaths?: string | Record<string, string>;
      proxy?: boolean;
    };
    logLevel?: string;
    /** Set by onnxruntime-web once a WebGPU session exists: the device it runs on. */
    webgpu?: { device?: unknown };
  };
}

export interface BuildOptions {
  ort: OrtLike;
  model: Uint8Array | ArrayBuffer | string;
  plan: BackendPlan;
  config?: SessionConfig;
  logger?: Logger;
  /** For messages: which file this is. */
  label?: string;
  wasmPaths?: string | Record<string, string>;
}

export interface BuiltSession {
  session: OrtSession;
  backend: Backend;
  /** Backends tried and rejected, with why — worth surfacing, not swallowing. */
  attempts: { backend: Backend; error: string }[];
  createMs: number;
}

export function toOrtOptions(
  backend: Backend,
  config: SessionConfig | undefined,
): Record<string, unknown> {
  const options: Record<string, unknown> = {
    executionProviders: [
      backend === "webnn" ? { name: "webnn", deviceType: config?.webnnDeviceType ?? "npu" } : backend,
    ],
    graphOptimizationLevel: config?.graphOptimizationLevel ?? "all",
    executionMode: config?.executionMode ?? "sequential",
  };
  if (config?.logSeverityLevel !== undefined) options["logSeverityLevel"] = config.logSeverityLevel;
  if (config?.enableProfiling === true) options["enableProfiling"] = true;
  if (config?.externalData && config.externalData.length > 0) {
    options["externalData"] = config.externalData;
  }
  if (config?.freeDimensionOverrides) {
    options["freeDimensionOverrides"] = config.freeDimensionOverrides;
  }

  if (config?.preferredOutputLocation && backend === "webgpu") {
    options["preferredOutputLocation"] =
      typeof config.preferredOutputLocation === "string"
        ? config.preferredOutputLocation
        : config.preferredOutputLocation;
  }
  if (config?.enableGraphCapture === true && backend === "webgpu") options["enableGraphCapture"] = true;
  return options;
}

/** Try each backend in the plan, return the first that builds. */
export async function buildSession(options: BuildOptions): Promise<BuiltSession> {
  const { ort, model, plan, config, logger } = options;
  const label = options.label ?? "model";

  configureOrtEnv(
    ort.env,
    {
      numThreads: plan.threads,
      simd: true,
      ...(options.wasmPaths !== undefined ? { wasmPaths: options.wasmPaths } : {}),
    },
    logger,
  );

  const attempts: { backend: Backend; error: string }[] = [];
  const started = performance.now();

  for (const backend of plan.order) {
    try {
      const session = await exclusive(() => ort.InferenceSession.create(model, toOrtOptions(backend, config)));
      if (attempts.length > 0) {
        logger?.warn(
          `${label} is running on ${BACKEND_LABELS[backend]} — ` +
            `${attempts.map((a) => BACKEND_LABELS[a.backend]).join(", ")} could not be used`,
        );
      } else {
        logger?.debug(`${label} created on ${BACKEND_LABELS[backend]}`);
      }
      return { session, backend, attempts, createMs: performance.now() - started };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      attempts.push({ backend, error: message });
      logger?.debug(`${label} would not build on ${backend}: ${message}`);
    }
  }

  throw new SessionError(
    `${label} could not be created on any backend.\n` +
      attempts.map((a) => `  ${BACKEND_LABELS[a.backend]}: ${a.error}`).join("\n"),
  );
}

/** Read the IO description, degrading to names when the build offers no more. */
export function readSpecs(
  session: OrtSession,
  side: "input" | "output",
): TensorSpec[] {
  const names = side === "input" ? session.inputNames : session.outputNames;
  const metadata = (session as unknown as Record<string, unknown>)[
    side === "input" ? "inputMetadata" : "outputMetadata"
  ];

  type MetadataEntry = { name?: unknown; type?: unknown; shape?: unknown; dims?: unknown };
  const byName = new Map<string, MetadataEntry>();
  if (Array.isArray(metadata)) {
    for (const [index, entry] of metadata.entries()) {
      const record = entry as MetadataEntry;
      const name = typeof record?.name === "string" ? record.name : names[index];
      if (name) byName.set(name, record);
    }
  } else if (metadata && typeof metadata === "object") {
    for (const [name, entry] of Object.entries(metadata as Record<string, MetadataEntry>)) {
      byName.set(name, entry);
    }
  }

  return names.map((name) => {
    const entry = byName.get(name);
    if (!entry) return { name };
    const shape = Array.isArray(entry.shape) ? entry.shape : entry.dims;
    return {
      name,
      ...(typeof entry.type === "string" ? { type: entry.type } : {}),
      ...(Array.isArray(shape) ? { dims: shape as (number | string)[] } : {}),
    };
  });
}
