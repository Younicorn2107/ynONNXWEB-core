import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import { castGatherIndices, INT32_SUFFIX } from "../../src/models/gatherIndices.js";
import { readGraphInfo } from "../../src/models/onnxProto.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "models");

async function run(bytes: Uint8Array, flag: boolean): Promise<Record<string, number[]>> {
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  try {
    const out = await session.run({ ids: new ort.Tensor("int64", BigInt64Array.of(3n, 1n, 4n), [1, 3]), flag: new ort.Tensor("bool", Uint8Array.of(flag ? 1 : 0), []) });
    return Object.fromEntries(Object.entries(out).map(([k, t]) => [k, Array.from(t.data as Float32Array)]));
  } finally {
    await session.release();
  }
}

describe("computed Gather indices cast to int32 (WebGPU)", () => {
  const original = new Uint8Array(readFileSync(join(MODELS, "gather_positions.onnx")));

  it("casts only the computed indices, the If's branches included", () => {
    const { bytes, casts } = castGatherIndices(original);

    expect(casts).toBe(3);
    const text = new TextDecoder("latin1").decode(bytes);
    expect(text.split(INT32_SUFFIX).length - 1).toBeGreaterThanOrEqual(6);
    expect(readGraphInfo(bytes).inputs.map((i) => i.name)).toEqual(["ids", "flag"]);
  });

  it("computes the same outputs after the rewrite", async () => {
    const { bytes } = castGatherIndices(original);
    for (const flag of [true, false]) {
      expect(await run(bytes, flag)).toEqual(await run(original, flag));
    }
    expect((await run(bytes, true))["Y"]!.slice(0, 4)).toEqual([8, 9, 10, 11]);
  });

  it("leaves a graph with no Gather untouched", () => {
    const plain = new Uint8Array(readFileSync(join(MODELS, "tiny_matmul.onnx")));
    const { bytes, casts } = castGatherIndices(plain);
    expect(casts).toBe(0);
    expect(bytes).toBe(plain);
  });
});
