/** The graph manifest — the one artifact genuinely shared with the Python pack. */

import { ManifestError } from "../errors.js";

export type GraphRole =
  | "single"
  | "encoder"
  | "decoder"
  | "decoder_with_past"
  | "decoder_merged"
  | "fused";

export interface KVGeometry {
  numLayers: number;
  numHeads: number;
  headDim: number;
  hasCrossAttention: boolean;
  dtype: string;
}

export interface GraphEntry {
  filename: string;
  role: GraphRole;
  sha256: string | null;
  sizeMb: number;
  inputs: string[];
  outputs: string[];
  /**
   * External-data sidecars (`model.onnx_data`). Absent from manifests written before ynONNX-core
   * grew the field, in which case the fetcher probes the conventional suffixes instead.
   */
  externalData: string[];
}

/** The manifest version from which `external_data` is always written. */
export const MANIFEST_VERSION_WITH_EXTERNAL_DATA = 2;

export interface GraphManifest {
  modelKey: string;
  computeType: string;
  architecture: string | null;
  graphs: GraphEntry[];
  tokenizerFile: string | null;
  logitsVocabSize: number | null;
  kv: KVGeometry | null;
  exposesCrossQk: boolean;
  extra: Record<string, unknown>;
  version: number;
}

const ROLES: readonly GraphRole[] = [
  "single",
  "encoder",
  "decoder",
  "decoder_with_past",
  "decoder_merged",
  "fused",
];

/** Order used by `decoderRoles`, matching `GraphManifest.decoder_roles`. */
const DECODER_ORDER: readonly GraphRole[] = [
  "decoder_merged",
  "decoder",
  "decoder_with_past",
  "single",
];

const KV_PREFIX = "past_key_values";
const PRESENT_PREFIX = "present";

/** Decide a graph's role from its IO, falling back on its name. */
export function detectRole(
  inputs: readonly string[],
  outputs: readonly string[],
  filename = "",
): GraphRole {
  const hasPast = inputs.some((name) => name.startsWith(KV_PREFIX));
  const hasPresent = outputs.some((name) => name.startsWith(PRESENT_PREFIX));
  const hasCacheBranch = inputs.includes("use_cache_branch");
  const hasEncoderStates = inputs.some(
    (name) => name === "encoder_hidden_states" || name === "encoder_outputs",
  );
  const hasIds = inputs.some(
    (name) => name === "input_ids" || name === "decoder_input_ids" || name === "tokens",
  );

  if (hasCacheBranch) return "decoder_merged";
  if (hasPast && (hasIds || hasEncoderStates)) return "decoder_with_past";
  if (hasPresent && hasIds) return "decoder";
  if (!hasIds && (hasPresent || outputs.includes("last_hidden_state"))) return "encoder";

  if (inputs.includes("max_length") || inputs.includes("num_beams") || outputs.includes("sequences")) {
    return "fused";
  }

  const lowered = filename.toLowerCase();
  if (lowered.includes("encoder")) return "encoder";
  if (lowered.includes("merged")) return "decoder_merged";
  if (lowered.includes("with_past") || lowered.includes("withpast")) return "decoder_with_past";
  if (lowered.includes("decoder")) return "decoder";
  return "single";
}

function asRole(value: unknown, where: string): GraphRole {
  if (typeof value === "string" && (ROLES as readonly string[]).includes(value)) {
    return value as GraphRole;
  }
  throw new ManifestError(
    `${where}: '${String(value)}' is not a graph role (expected one of ${ROLES.join(", ")})`,
  );
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];
}

/** Parse a `ynonnx_manifest.json` payload. */
export function parseManifest(payload: unknown): GraphManifest {
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    throw new ManifestError("the manifest is not a JSON object");
  }
  const raw = payload as Record<string, unknown>;

  const modelKey = raw["model_key"];
  if (typeof modelKey !== "string" || modelKey.length === 0) {
    throw new ManifestError("the manifest has no model_key");
  }

  const graphsRaw = raw["graphs"];
  if (!Array.isArray(graphsRaw)) {
    throw new ManifestError(`manifest '${modelKey}' has no graphs array`);
  }

  const graphs: GraphEntry[] = graphsRaw.map((item, index) => {
    if (item === null || typeof item !== "object") {
      throw new ManifestError(`manifest '${modelKey}': graph ${index} is not an object`);
    }
    const entry = item as Record<string, unknown>;
    const filename = entry["filename"];
    if (typeof filename !== "string" || filename.length === 0) {
      throw new ManifestError(`manifest '${modelKey}': graph ${index} has no filename`);
    }
    const sha = entry["sha256"];
    const size = entry["size_mb"];
    return {
      filename,
      role: asRole(entry["role"] ?? "single", `manifest '${modelKey}', graph '${filename}'`),
      sha256: typeof sha === "string" ? sha : null,
      sizeMb: typeof size === "number" ? size : 0,
      inputs: asStringArray(entry["inputs"]),
      outputs: asStringArray(entry["outputs"]),
      externalData: asStringArray(entry["external_data"]),
    };
  });

  const kvRaw = raw["kv"];
  let kv: KVGeometry | null = null;
  if (kvRaw !== null && typeof kvRaw === "object" && !Array.isArray(kvRaw)) {
    const k = kvRaw as Record<string, unknown>;
    kv = {
      numLayers: typeof k["num_layers"] === "number" ? k["num_layers"] : 0,
      numHeads: typeof k["num_heads"] === "number" ? k["num_heads"] : 0,
      headDim: typeof k["head_dim"] === "number" ? k["head_dim"] : 0,
      hasCrossAttention: k["has_cross_attention"] === true,
      dtype: asString(k["dtype"], "float32"),
    };
  }

  const vocab = raw["logits_vocab_size"];
  const architecture = raw["architecture"];
  const tokenizer = raw["tokenizer_file"];
  const extra = raw["extra"];
  const version = raw["version"];

  return {
    modelKey,
    computeType: asString(raw["compute_type"], "float32"),
    architecture: typeof architecture === "string" ? architecture : null,
    graphs,
    tokenizerFile: typeof tokenizer === "string" ? tokenizer : null,
    logitsVocabSize: typeof vocab === "number" ? vocab : null,
    kv,
    exposesCrossQk: raw["exposes_cross_qk"] === true,
    extra: extra !== null && typeof extra === "object" && !Array.isArray(extra)
      ? (extra as Record<string, unknown>)
      : {},
    version: typeof version === "number" ? version : 1,
  };
}

/** The exact inverse of `parseManifest`. */
export function serializeManifest(manifest: GraphManifest): Record<string, unknown> {
  return {
    model_key: manifest.modelKey,
    compute_type: manifest.computeType,
    architecture: manifest.architecture,
    graphs: manifest.graphs.map((g) => ({
      filename: g.filename,
      role: g.role,
      sha256: g.sha256,
      size_mb: g.sizeMb,
      inputs: g.inputs,
      outputs: g.outputs,
      external_data: g.externalData,
    })),
    tokenizer_file: manifest.tokenizerFile,
    logits_vocab_size: manifest.logitsVocabSize,
    kv: manifest.kv
      ? {
          num_layers: manifest.kv.numLayers,
          num_heads: manifest.kv.numHeads,
          head_dim: manifest.kv.headDim,
          has_cross_attention: manifest.kv.hasCrossAttention,
          dtype: manifest.kv.dtype,
        }
      : null,
    exposes_cross_qk: manifest.exposesCrossQk,
    extra: manifest.extra,
    version: manifest.version,
  };
}

export function graphByRole(manifest: GraphManifest, role: GraphRole): GraphEntry | null {
  return manifest.graphs.find((g) => g.role === role) ?? null;
}

export function hasRole(manifest: GraphManifest, role: GraphRole): boolean {
  return manifest.graphs.some((g) => g.role === role);
}

export function isEncoderDecoder(manifest: GraphManifest): boolean {
  return hasRole(manifest, "encoder");
}

export function usesMergedDecoder(manifest: GraphManifest): boolean {
  return hasRole(manifest, "decoder_merged");
}

/** Decoder roles present, in the pack's canonical order. */
export function decoderRoles(manifest: GraphManifest): GraphRole[] {
  return DECODER_ORDER.filter((role) => hasRole(manifest, role));
}

/** Total bytes to fetch, from the manifest alone — before anything moves. */
export function totalSizeMb(manifest: GraphManifest): number {
  return manifest.graphs.reduce((sum, g) => sum + g.sizeMb, 0);
}
