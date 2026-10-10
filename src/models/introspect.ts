/** Describe a set of graphs that came without a manifest. */

import { readGraphInfo, ONNX_DTYPES, type GraphDim, type GraphInfo } from "./onnxProto.js";
import {
  MANIFEST_VERSION_WITH_EXTERNAL_DATA,
  detectRole,
  type GraphEntry,
  type GraphManifest,
  type KVGeometry,
} from "../types/manifest.js";

const MIB = 1024 * 1024;

function asInt(value: GraphDim | undefined): number {
  return typeof value === "number" && value > 0 ? value : 0;
}

/** Layers, heads and head size, read off the `past_key_values.*` inputs. */
export function kvGeometry(info: GraphInfo): KVGeometry | null {
  const layers = new Set<number>();
  let heads = 0;
  let headDim = 0;
  let dtype = "float32";
  let hasCross = false;

  for (const value of info.inputs) {
    const name = value.name;
    if (!name.startsWith("past_key_values")) continue;
    for (const part of name.split(".").slice(1)) {
      if (/^\d+$/.test(part)) {
        layers.add(Number(part));
        break;
      }
    }
    if (name.includes(".encoder.") || name.endsWith(".encoder")) hasCross = true;
    if (value.dims.length === 4) {
      heads ||= asInt(value.dims[1]);
      headDim ||= asInt(value.dims[3]);
    }
    dtype = ONNX_DTYPES[value.elemType] ?? dtype;
  }

  if (layers.size === 0) return null;
  return { numLayers: layers.size, numHeads: heads, headDim, hasCrossAttention: hasCross, dtype };
}

/** The static width of the logits output, when the export declares one. */
export function logitsVocabSize(info: GraphInfo): number | null {
  for (const value of info.outputs) {
    if (value.name !== "logits" && value.name !== "output") continue;
    if (value.dims.length > 0) return asInt(value.dims[value.dims.length - 1]) || null;
  }
  return null;
}

export interface IntrospectedGraph {
  filename: string;
  /** The graph file itself, external data excluded. */
  bytes: Uint8Array;
  /** Sizes of the sidecars present beside it, by name, for `size_mb`. */
  sidecarBytes?: Readonly<Record<string, number>>;
}

export interface BuildManifestOptions {
  modelKey: string;
  computeType?: string;
  architecture?: string | null;
  tokenizerFile?: string | null;
}

/** The manifest of a folder of graphs, as Python's `build_manifest` writes it. */
export function buildManifest(
  graphs: readonly IntrospectedGraph[],
  options: BuildManifestOptions,
): GraphManifest {
  const ordered = [...graphs].sort((a, b) => (a.filename < b.filename ? -1 : a.filename > b.filename ? 1 : 0));
  const entries: GraphEntry[] = [];
  let kv: KVGeometry | null = null;
  let vocab: number | null = null;
  let exposesCrossQk = false;

  for (const graph of ordered) {
    const info = readGraphInfo(graph.bytes);
    const inputs = info.inputs.map((v) => v.name);
    const outputs = info.outputs.map((v) => v.name);
    const role = detectRole(inputs, outputs, graph.filename);
    const sidecars = info.externalData.filter((name) => graph.sidecarBytes?.[name] !== undefined);
    const total =
      graph.bytes.byteLength + sidecars.reduce((sum, name) => sum + (graph.sidecarBytes?.[name] ?? 0), 0);
    entries.push({
      filename: graph.filename,
      role,
      sha256: null,
      sizeMb: Math.round((total / MIB) * 100) / 100,
      inputs,
      outputs,
      externalData: sidecars,
    });
    if (role === "decoder" || role === "decoder_with_past" || role === "decoder_merged" || role === "single") {
      kv ??= kvGeometry(info);
      vocab ??= logitsVocabSize(info);
      exposesCrossQk ||= outputs.some((name) => name.includes("cross_qk") || name.includes("cross_attentions"));
    }
  }

  return {
    modelKey: options.modelKey,
    computeType: options.computeType ?? "float32",
    architecture: options.architecture ?? null,
    graphs: entries,
    tokenizerFile: options.tokenizerFile ?? null,
    logitsVocabSize: vocab,
    kv,
    exposesCrossQk,
    extra: {},
    version: MANIFEST_VERSION_WITH_EXTERNAL_DATA,
  };
}
