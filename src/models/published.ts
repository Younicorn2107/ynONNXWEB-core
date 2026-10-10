/** Which graphs of a repository to load, when nothing names them. */

import { ModelNotFoundError } from "../errors.js";
import type { GraphManifest, GraphRole } from "../types/manifest.js";

/** File suffixes by requested precision, most specific first. */
export const PRECISION_SUFFIXES: Readonly<Record<string, readonly string[]>> = {
  float32: [""],
  float16: ["_fp16"],
  int8: ["_quantized", "_int8", "_uint8"],
  int8_float16: ["_quantized", "_int8"],
  int4: ["_q4", "_bnb4"],
  int4_float16: ["_q4f16"],
  uint8: ["_uint8", "_quantized"],
};

/** Files outside the graph folder that a model may need. */
export const SIDE_FILES: readonly string[] = [
  "tokenizer.json",
  "config.json",
  "generation_config.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "preprocessor_config.json",
];

export interface PublishedChoice {
  /** Graph filenames to fetch, relative to the graph folder. */
  graphs: string[];
  /** The suffix that matched, for messages. */
  suffix: string;
}

function first(available: ReadonlySet<string>, names: readonly string[]): string | null {
  return names.find((name) => available.has(name)) ?? null;
}

/** The graphs that serve `computeType`, from the names a folder holds. */
export function choosePublishedGraphs(
  files: readonly string[],
  computeType: string,
): PublishedChoice {
  const available = new Set(files);
  const suffixes = PRECISION_SUFFIXES[computeType];
  if (!suffixes) {
    throw new ModelNotFoundError(
      `no published naming convention for '${computeType}' ` +
        `(known: ${Object.keys(PRECISION_SUFFIXES).join(", ")})`,
    );
  }

  for (const suffix of suffixes) {
    const encoder = first(available, [`encoder_model${suffix}.onnx`, "encoder_model.onnx"]);
    const merged = first(available, [`decoder_model_merged${suffix}.onnx`, "decoder_model_merged.onnx"]);
    if (encoder && merged) return { graphs: [encoder, merged], suffix };

    const withPast = first(available, [`decoder_with_past_model${suffix}.onnx`, "decoder_with_past_model.onnx"]);
    const decoder = first(available, [`decoder_model${suffix}.onnx`, "decoder_model.onnx"]);

    if (encoder && withPast && decoder) return { graphs: [encoder, decoder, withPast], suffix };
    if (encoder && decoder && !withPast) return { graphs: [encoder, decoder], suffix };

    const single = available.has(`model${suffix}.onnx`) ? `model${suffix}.onnx` : null;
    if (single) return { graphs: [single], suffix };
  }

  const onnx = files.filter((name) => name.endsWith(".onnx"));
  throw new ModelNotFoundError(
    `no graph at '${computeType}' (looked for the suffixes ${suffixes.map((s) => `'${s || "none"}'`).join(", ")}). ` +
      `Graphs present: ${onnx.join(", ") || "none"}.`,
  );
}

/** The graphs of a manifest a model actually runs. */
export function neededGraphs(manifest: GraphManifest): GraphManifest {
  const roles = new Set(manifest.graphs.map((g) => g.role));
  const drop = new Set<GraphRole>();
  if (roles.has("decoder_merged")) {
    drop.add("decoder");
    drop.add("decoder_with_past");
  }
  if (roles.has("decoder_merged") || roles.has("decoder") || roles.has("single")) {
    drop.add("fused");
  }
  const graphs = manifest.graphs.filter((g) => !drop.has(g.role));
  return graphs.length === manifest.graphs.length ? manifest : { ...manifest, graphs };
}
