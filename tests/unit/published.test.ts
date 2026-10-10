import { describe, expect, it } from "vitest";

import { ModelNotFoundError } from "../../src/errors.js";
import { choosePublishedGraphs, neededGraphs } from "../../src/models/published.js";
import { parseManifest } from "../../src/types/manifest.js";

const WHISPER = [
  "decoder_model.onnx",
  "decoder_model_merged.onnx",
  "decoder_model_merged_fp16.onnx",
  "decoder_model_merged_q4.onnx",
  "decoder_model_merged_quantized.onnx",
  "decoder_with_past_model.onnx",
  "encoder_model.onnx",
  "encoder_model_fp16.onnx",
  "encoder_model_q4.onnx",
  "encoder_model_quantized.onnx",
];

const LLM = [
  "model.onnx",
  "model.onnx_data",
  "model_fp16.onnx",
  "model_fp16.onnx_data",
  "model_q4.onnx",
  "model_q4f16.onnx",
  "model_q4f16.onnx_data",
];

describe("choosePublishedGraphs", () => {
  it("takes the encoder and the merged decoder at the precision asked", () => {
    expect(choosePublishedGraphs(WHISPER, "int4").graphs).toEqual([
      "encoder_model_q4.onnx",
      "decoder_model_merged_q4.onnx",
    ]);
    expect(choosePublishedGraphs(WHISPER, "int8").graphs).toEqual([
      "encoder_model_quantized.onnx",
      "decoder_model_merged_quantized.onnx",
    ]);
    expect(choosePublishedGraphs(WHISPER, "float32").graphs).toEqual([
      "encoder_model.onnx",
      "decoder_model_merged.onnx",
    ]);
  });

  it("reads int4 weights with float16 arithmetic as q4f16", () => {
    expect(choosePublishedGraphs(LLM, "int4_float16").graphs).toEqual(["model_q4f16.onnx"]);
    expect(choosePublishedGraphs(LLM, "int4").graphs).toEqual(["model_q4.onnx"]);
  });

  it("falls back to the full-precision graph of a role, as Python does", () => {
    const files = ["encoder_model.onnx", "decoder_model_merged_q4.onnx"];
    expect(choosePublishedGraphs(files, "int4").graphs).toEqual([
      "encoder_model.onnx",
      "decoder_model_merged_q4.onnx",
    ]);
  });

  it("needs both halves of a split decoder", () => {
    const files = ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"];
    expect(choosePublishedGraphs(files, "float32").graphs).toEqual([
      "encoder_model.onnx",
      "decoder_model.onnx",
      "decoder_with_past_model.onnx",
    ]);
  });

  it("says what is there when nothing matches", () => {
    let thrown: unknown;
    try {
      choosePublishedGraphs(LLM, "int8");
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBeInstanceOf(ModelNotFoundError);
    expect(String((thrown as Error).message)).toContain("model_q4.onnx");
  });

  it("refuses a precision with no naming convention", () => {
    expect(() => choosePublishedGraphs(LLM, "int2")).toThrow(ModelNotFoundError);
  });
});

describe("neededGraphs", () => {
  const graph = (filename: string, role: string) => ({ filename, role, size_mb: 1, inputs: [], outputs: [] });

  it("drops the split pair a merged decoder replaces", () => {
    const manifest = parseManifest({
      model_key: "base_int8",
      graphs: [
        graph("encoder_model.onnx", "encoder"),
        graph("decoder_model.onnx", "decoder"),
        graph("decoder_with_past_model.onnx", "decoder_with_past"),
        graph("decoder_model_merged.onnx", "decoder_merged"),
      ],
    });
    expect(neededGraphs(manifest).graphs.map((g) => g.role)).toEqual(["encoder", "decoder_merged"]);
  });

  it("keeps a fused graph only when it is all there is", () => {
    const fused = parseManifest({ model_key: "x", graphs: [graph("beam.onnx", "fused")] });
    expect(neededGraphs(fused).graphs).toHaveLength(1);
    const both = parseManifest({
      model_key: "x",
      graphs: [graph("beam.onnx", "fused"), graph("model.onnx", "single")],
    });
    expect(neededGraphs(both).graphs.map((g) => g.role)).toEqual(["single"]);
  });

  it("leaves a manifest with nothing redundant untouched", () => {
    const manifest = parseManifest({ model_key: "x", graphs: [graph("model.onnx", "single")] });
    expect(neededGraphs(manifest)).toBe(manifest);
  });
});
