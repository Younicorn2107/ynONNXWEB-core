/** Reading a session's input and output shapes. */

import { describe, expect, it } from "vitest";

import { readSpecs } from "../../src/session/create.js";

const NAMES = ["input_ids", "past_key_values.0.decoder.key"];

function session(metadata: unknown): never {
  return { inputNames: NAMES, outputNames: [], inputMetadata: metadata } as never;
}

describe("readSpecs", () => {
  it("reads the array form used by onnxruntime-web 1.21+", () => {
    const specs = readSpecs(
      session([
        { name: "input_ids", isTensor: true, type: "int64", shape: ["batch_size", "seq"] },
        {
          name: "past_key_values.0.decoder.key",
          isTensor: true,
          type: "float32",
          shape: ["batch_size", 8, "past_decoder_sequence_length", 64],
        },
      ]),
      "input",
    );

    expect(specs[0]).toEqual({ name: "input_ids", type: "int64", dims: ["batch_size", "seq"] });
    expect(specs[1]?.dims).toEqual(["batch_size", 8, "past_decoder_sequence_length", 64]);
  });

  it("still reads the older record form", () => {
    const specs = readSpecs(
      session({
        input_ids: { type: "int64", dims: ["batch_size", "seq"] },
        "past_key_values.0.decoder.key": { type: "float32", dims: [1, 8, 0, 64] },
      }),
      "input",
    );

    expect(specs[0]?.dims).toEqual(["batch_size", "seq"]);
    expect(specs[1]?.dims).toEqual([1, 8, 0, 64]);
  });

  it("falls back to position when an array entry has no name", () => {
    const specs = readSpecs(session([{ type: "int64", shape: [1] }, {}]), "input");
    expect(specs[0]).toEqual({ name: "input_ids", type: "int64", dims: [1] });
    expect(specs[1]).toEqual({ name: "past_key_values.0.decoder.key" });
  });

  it("returns names alone when a build exposes no metadata", () => {
    expect(readSpecs(session(undefined), "input")).toEqual([
      { name: "input_ids" },
      { name: "past_key_values.0.decoder.key" },
    ]);
  });
});
