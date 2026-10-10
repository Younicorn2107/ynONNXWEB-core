/** Graphs that came without a manifest are described exactly as Python describes them. */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { buildManifest, type IntrospectedGraph } from "../../src/models/introspect.js";
import { readGraphInfo } from "../../src/models/onnxProto.js";
import { serializeManifest } from "../../src/types/manifest.js";

const GRAPHS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "graphs");

function folder(name: string): { expected: Record<string, unknown>; graphs: IntrospectedGraph[] } {
  const dir = join(GRAPHS, name);
  const files = readdirSync(dir);
  const sidecarBytes: Record<string, number> = {};
  for (const file of files) {
    if (!file.endsWith(".onnx") && file !== "manifest.json") sidecarBytes[file] = statSync(join(dir, file)).size;
  }
  const graphs = files
    .filter((file) => file.endsWith(".onnx"))
    .map((file) => ({ filename: file, bytes: new Uint8Array(readFileSync(join(dir, file))), sidecarBytes }));
  const expected = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf-8")) as Record<string, unknown>;
  return { expected, graphs };
}

describe("buildManifest agrees with ynonnx_core.build_manifest", () => {
  it.each(["decoder_only", "encoder_decoder"])("%s", (name) => {
    const { expected, graphs } = folder(name);
    const manifest = buildManifest(graphs, {
      modelKey: expected["model_key"] as string,
      computeType: expected["compute_type"] as string,
      architecture: expected["architecture"] as string,
      tokenizerFile: expected["tokenizer_file"] as string,
    });
    expect(serializeManifest(manifest)).toEqual(expected);
  });
});

describe("readGraphInfo", () => {
  it("reads symbolic and fixed dimensions and the element type", () => {
    const { graphs } = folder("decoder_only");
    const info = readGraphInfo(graphs[0]!.bytes);
    const past = info.inputs.find((v) => v.name === "past_key_values.0.key")!;
    expect(past.dims).toEqual(["batch", 4, "past", 16]);
    expect(past.elemType).toBe(10);
    const logits = info.outputs.find((v) => v.name === "logits")!;
    expect(logits.dims).toEqual(["batch", "seq", 1000]);
  });

  it("finds the sidecar an initializer points at", () => {
    const { graphs } = folder("decoder_only");
    expect(readGraphInfo(graphs[0]!.bytes).externalData).toEqual(["model_q4f16.onnx_data"]);
  });

  it("refuses bytes that are not a model", () => {
    expect(() => readGraphInfo(new Uint8Array([8, 1]))).toThrow(/no graph/);
    expect(() => readGraphInfo(new Uint8Array([0x3a, 0x50, 1]))).toThrow(/truncated/);
  });
});
