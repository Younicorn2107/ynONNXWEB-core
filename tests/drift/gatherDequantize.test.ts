import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import { gatherBeforeDequantize } from "../../src/models/gatherDequantize.js";
import { operatorTypes } from "../../src/models/protobuf.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "models");

async function run(bytes: Uint8Array, feeds: Record<string, ort.Tensor>, optimized = true): Promise<Float32Array> {
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], ...(optimized ? {} : { graphOptimizationLevel: "disabled" as const }) });
  try {
    return (await session.run(feeds))["y"]!.data as Float32Array;
  } finally {
    await session.release();
  }
}

describe("a quantized table gathered before it is dequantized", () => {
  const model = new Uint8Array(readFileSync(join(MODELS, "dequant_gather_toy.onnx")));
  const ids = new ort.Tensor("int64", BigInt64Array.of(0n, 6n, 3n, 3n, 1n, 5n), [2, 3]);
  const flag = (on: boolean) => new ort.Tensor("bool", Uint8Array.of(on ? 1 : 0), [1]);

  it("gives the same values, bit for bit, in both branches", async () => {
    const { bytes, rewritten } = gatherBeforeDequantize(model);
    expect(rewritten).toBe(2);
    expect(operatorTypes(bytes).has("DequantizeLinear")).toBe(false);
    for (const on of [true, false]) {
      expect(Array.from(await run(bytes, { ids, flag: flag(on) }))).toEqual(Array.from(await run(model, { ids, flag: flag(on) })));
    }
  });

  it("reads a tied head's transpose from the transposed 8-bit table, bit for bit", async () => {
    const tied = new Uint8Array(readFileSync(join(MODELS, "dequant_transpose_toy.onnx")));
    const { bytes, rewritten } = gatherBeforeDequantize(tied);
    expect(rewritten).toBe(2);
    expect(operatorTypes(bytes).has("Transpose")).toBe(false);
    const ids3 = new ort.Tensor("int64", BigInt64Array.of(0n, 6n, 3n), [3]);
    await expect(run(tied, { ids: ids3 })).rejects.toThrow(/vector/);
    expect(Array.from(await run(bytes, { ids: ids3 }))).toEqual(Array.from(await run(tied, { ids: ids3 }, false)));
  });

  it("leaves a table something else reads", () => {
    const { rewritten } = gatherBeforeDequantize(new Uint8Array(readFileSync(join(MODELS, "merged_toy.onnx"))));
    expect(rewritten).toBe(0);
  });
});
