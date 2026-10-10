import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import { externalizeWeights, splitExternalData } from "../../src/models/externalize.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "models");
const K = 256;
const input = Float32Array.from({ length: 4 * K }, (_, i) => Math.sin(i / 7));

async function run(bytes: Uint8Array, externalData?: { path: string; data: Uint8Array }[]): Promise<Float32Array> {
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], ...(externalData ? { externalData } : {}) });
  try {
    return (await session.run({ A: new ort.Tensor("float32", input, [1, 4, K]) }))["Y"]!.data as Float32Array;
  } finally {
    await session.release();
  }
}

describe("weights moved out of the graph", () => {
  const model = new Uint8Array(readFileSync(join(MODELS, "mnb_b64_2d.onnx")));

  it("loads as external data and computes the same, bit for bit", async () => {
    const moved = externalizeWeights(model, "m.weights", { minTotal: 0, chunkBytes: 4096 })!;
    expect(moved.files.length).toBeGreaterThan(1);
    expect(moved.bytes.length).toBeLessThan(model.length);
    expect(Array.from(await run(moved.bytes, moved.files))).toEqual(Array.from(await run(model)));
  });

  it("leaves a small model alone", () => {
    expect(externalizeWeights(model, "m.weights")).toBeNull();
  });

  it("cuts a weight file into parts between tensors, the graph pointing into them", async () => {
    const one = externalizeWeights(model, "m.weights", { minTotal: 0 })!;
    expect(one.files.length).toBe(1);
    const file = one.files[0]!;
    const split = splitExternalData(one.bytes, file.path, 4096);
    expect(split.parts.length).toBeGreaterThan(1);
    const parts = split.parts.map((p) => ({ path: p.name, data: file.data.slice(p.start, p.end) }));
    expect(Array.from(await run(split.bytes, parts))).toEqual(Array.from(await run(model)));
  });
});
