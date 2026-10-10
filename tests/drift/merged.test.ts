import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import { splitMergedDecoder } from "../../src/models/splitMerged.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "models");

async function run(bytes: Uint8Array, feeds: Record<string, ort.Tensor>): Promise<number[]> {
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  try {
    return Array.from((await session.run(feeds))["y"]!.data as Float32Array);
  } finally {
    await session.release();
  }
}

describe("a merged decoder split into its branches (WebGPU)", () => {
  const merged = new Uint8Array(readFileSync(join(MODELS, "merged_toy.onnx")));
  const x = new ort.Tensor("float32", Float32Array.of(1, 2, 3), [3]);
  const past = new ort.Tensor("float32", Float32Array.of(10, 20, 30), [3]);
  const flag = (on: boolean) => new ort.Tensor("bool", Uint8Array.of(on ? 1 : 0), [1]);

  it("gives each branch's outputs, the first pass with no past input", async () => {
    const split = splitMergedDecoder(merged)!;
    expect(split).not.toBeNull();
    const first = await ort.InferenceSession.create(split.firstPass, { executionProviders: ["wasm"] });
    expect(first.inputNames).toEqual(["x"]);
    expect(first.outputNames).toEqual(["y"]);
    await first.release();
    expect(await run(split.firstPass, { x })).toEqual(await run(merged, { x, past, use_cache_branch: flag(false) }));
    expect(await run(split.withPast, { x, past })).toEqual(await run(merged, { x, past, use_cache_branch: flag(true) }));
  });

  it("leaves a graph with no If alone", () => {
    expect(splitMergedDecoder(new Uint8Array(readFileSync(join(MODELS, "argmax_k1.onnx"))))).toBeNull();
  });
});
