import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import { setNBitsAccuracy } from "../../src/models/nbitsAccuracy.js";
import { padMatMulNBitsRows } from "../../src/models/padNBits.js";
import { operatorTypes } from "../../src/models/protobuf.js";
import { repackMatMulNBits } from "../../src/models/repackNBits.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "models");
const K = 256;
const input = Float32Array.from({ length: 4 * K }, (_, i) => -1 + (2 * i) / (4 * K - 1));

async function run(bytes: Uint8Array): Promise<Float32Array> {
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  try {
    const out = await session.run({ A: new ort.Tensor("float32", input, [1, 4, K]) });
    return out["Y"]!.data as Float32Array;
  } finally {
    await session.release();
  }
}

describe("MatMulNBits repacked to blocks of 32", () => {
  for (const file of ["mnb_b64_2d.onnx", "mnb_b128_1d.onnx"]) {
    it(`${file} computes the same product after the rewrite`, async () => {
      const original = new Uint8Array(readFileSync(join(MODELS, file)));
      const { bytes, rewritten } = repackMatMulNBits(original);
      expect(rewritten).toBe(1);
      const [before, after] = [await run(original), await run(bytes)];
      let max = 0;
      before.forEach((v, i) => (max = Math.max(max, Math.abs(v - after[i]!))));
      expect(max).toBeLessThan(1e-4);
    });
  }

  it("flattens 2-D scales at blocks of 32 (WebGPU wants them flat), same product", async () => {
    const original = new Uint8Array(readFileSync(join(MODELS, "mnb_b32_2d.onnx")));
    const { bytes, rewritten } = repackMatMulNBits(original);
    expect(rewritten).toBe(1);
    const [before, after] = [await run(original), await run(bytes)];
    expect(Array.from(after)).toEqual(Array.from(before));
  });

  it("leaves a graph already at 32 untouched", () => {
    const original = new Uint8Array(readFileSync(join(MODELS, "mnb_b32_1d.onnx")));
    const { bytes, rewritten } = repackMatMulNBits(original);
    expect(rewritten).toBe(0);
    expect(bytes).toBe(original);
  });
});

describe("MatMulNBits zero points padded to whole words (WebGPU)", () => {
  it("adds a zero row and a Slice, and computes the same product", async () => {
    const original = new Uint8Array(readFileSync(join(MODELS, "mnb_k384_n7_zp.onnx")));
    const { bytes, padded } = padMatMulNBitsRows(original);
    expect(padded).toBe(1);
    const input = Float32Array.from({ length: 384 }, (_, i) => Math.sin(i));
    const runK = async (model: Uint8Array) => {
      const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
      try {
        const out = await session.run({ A: new ort.Tensor("float32", input, [1, 1, 384]) });
        return { data: Array.from(out["Y"]!.data as Float32Array), dims: out["Y"]!.dims };
      } finally {
        await session.release();
      }
    };
    const [before, after] = [await runK(original), await runK(bytes)];
    expect(after.dims).toEqual([1, 1, 7]);
    after.data.forEach((v, i) => expect(Math.abs(v - before.data[i]!)).toBeLessThan(1e-5));
  });

  it("leaves whole-word zero points alone", () => {
    const original = new Uint8Array(readFileSync(join(MODELS, "mnb_b32_1d.onnx")));
    expect(padMatMulNBitsRows(original).bytes).toBe(original);
  });
});

describe("MatMulNBits at accuracy level 4 (int8 compute on WebAssembly)", () => {
  it("is set where absent, and the product stays within int8 activation rounding", async () => {
    const original = new Uint8Array(readFileSync(join(MODELS, "mnb_b32_1d.onnx")));
    const { bytes, rewritten } = setNBitsAccuracy(original);
    expect(rewritten).toBe(1);
    expect(operatorTypes(bytes).has("MatMulNBits")).toBe(true);
    expect(setNBitsAccuracy(bytes).rewritten).toBe(0);
    const [before, after] = [await run(original), await run(bytes)];
    let max = 0;
    let scale = 0;
    before.forEach((v, i) => {
      max = Math.max(max, Math.abs(v - after[i]!));
      scale = Math.max(scale, Math.abs(v));
    });
    expect(max / scale).toBeLessThan(0.02);
  });
});
