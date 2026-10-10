/** A live probe: does this backend actually run a graph? */

import { exclusive } from "../session/exclusive.js";
import type { InferenceSession, Tensor } from "onnxruntime-web";

import type { Backend } from "./chain.js";

/**
 * The smallest valid ONNX model that computes something: one Add node over a 1-element float
 * tensor, opset 17, IR 10. 84 bytes.
 */
const PROBE_MODEL_B64 =
  "CAo6SgoOCgF4CgF5EgF6IgNBZGQSBXByb2JlWg8KAXgSCgoICAESBAoCCAFaDwoBeRIKCggIARIECgIIAWIPCgF6EgoKCAgBEgQKAggBQgQKABAR";

function decodeProbeModel(): Uint8Array {
  const binary = atob(PROBE_MODEL_B64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export interface ProbeResult {
  backend: Backend;
  ok: boolean;
  ms: number;
  error?: string;
}

/** Run the probe graph on one backend, queued with every other runtime call (see `exclusive`). */
export function probeBackend(...args: Parameters<typeof probeOnce>): Promise<ProbeResult> {
  return exclusive(() => probeOnce(...args));
}

/** Run the probe graph on one backend. */
async function probeOnce(
  ort: {
    InferenceSession: {
      create(model: Uint8Array, options?: unknown): Promise<InferenceSession>;
    };
    Tensor: new (type: string, data: Float32Array, dims: number[]) => Tensor;
  },
  backend: Backend,
): Promise<ProbeResult> {
  const started = performance.now();
  let session: InferenceSession | undefined;
  try {
    session = await ort.InferenceSession.create(decodeProbeModel(), {
      executionProviders: [backend === "webnn" ? { name: "webnn", deviceType: "npu" } : backend],

      logSeverityLevel: 4,
    });
    const one = new ort.Tensor("float32", new Float32Array([1]), [1]);
    const outputs = await session.run({ x: one, y: one });
    const z = outputs["z"];
    const value = z ? (z.data as Float32Array)[0] : undefined;

    const ok = value === 2;
    return {
      backend,
      ok,
      ms: performance.now() - started,
      ...(ok ? {} : { error: `the probe graph returned ${String(value)} instead of 2` }),
    };
  } catch (error) {
    return {
      backend,
      ok: false,
      ms: performance.now() - started,
      error: error instanceof Error ? error.message : String(error),
    };
  } finally {
    await session?.release?.().catch(() => undefined);
  }
}

/** Probe several backends, keeping only those that ran the graph correctly. */
export async function probeBackends(
  ort: Parameters<typeof probeBackend>[0],
  candidates: readonly Backend[],
): Promise<ProbeResult[]> {
  const results: ProbeResult[] = [];
  for (const backend of candidates) {
    results.push(await probeBackend(ort, backend));
  }
  return results;
}
