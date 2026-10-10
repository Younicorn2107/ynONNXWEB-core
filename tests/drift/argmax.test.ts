import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import { ARGMAX_SUFFIX, MAX_SUFFIX, argmaxHead, replaceArgMax } from "../../src/models/argmaxHead.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "models");

describe("a scores output reduced to its best class in the graph", () => {
  for (const opset of [13, 18]) {
    it(`gives each step's best class and score (opset ${opset})`, async () => {
      const original = new Uint8Array(readFileSync(join(MODELS, `softmax_head_o${opset}.onnx`)));
      const { bytes, rewritten } = argmaxHead(original, "probs");
      expect(rewritten).toBe(true);
      const x = Float32Array.from({ length: 2 * 3 * 6 }, (_, i) => Math.sin(i * 1.7) * 3);

      x[6] = x[7] = 9;
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      try {
        expect(session.outputNames).toEqual([`probs${ARGMAX_SUFFIX}`, `probs${MAX_SUFFIX}`]);
        const out = await session.run({ x: new ort.Tensor("float32", x, [2, 3, 6]) });
        const ids = Array.from(out[`probs${ARGMAX_SUFFIX}`]!.data as Float32Array);
        const best = Array.from(out[`probs${MAX_SUFFIX}`]!.data as Float32Array);
        expect(out[`probs${ARGMAX_SUFFIX}`]!.dims).toEqual([2, 3, 1]);
        for (let r = 0; r < 6; r += 1) {
          const row = Array.from(x.subarray(r * 6, r * 6 + 6));
          const max = Math.max(...row);
          expect(ids[r]).toBe(row.indexOf(max));
          const sum = row.reduce((s, v) => s + Math.exp(v - max), 0);
          expect(best[r]).toBeCloseTo(1 / sum, 5);
        }
      } finally {
        await session.release();
      }
    });
  }

  it("leaves a model without that output as it was", () => {
    const original = new Uint8Array(readFileSync(join(MODELS, "softmax_head_o13.onnx")));
    expect(argmaxHead(original, "logits")).toEqual({ bytes: original, rewritten: false });
  });
});

describe("ArgMax rebuilt without ArgMax (WebGPU)", () => {
  for (const keep of [0, 1]) {
    it(`gives ArgMax's indices (keepdims ${keep})`, async () => {
      const original = new Uint8Array(readFileSync(join(MODELS, `argmax_k${keep}.onnx`)));
      const { bytes, replaced } = replaceArgMax(original);
      expect(replaced).toBe(1);
      const x = Float32Array.from({ length: 3 * 4 * 257 }, (_, i) => Math.sin(i * 0.37) * 5);
      x[257 + 3] = x[257 + 200] = 99;
      const run = async (model: Uint8Array) => {
        const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
        try {
          const out = await session.run({ x: new ort.Tensor("float32", x, [3, 4, 257]) });
          return { ids: Array.from(out["ids_plus"]!.data as BigInt64Array, Number), dims: out["ids_plus"]!.dims, type: out["ids_plus"]!.type };
        } finally {
          await session.release();
        }
      };
      const [want, got] = [await run(original), await run(bytes)];
      expect(got).toEqual(want);
      expect(got.ids[1]).toBe(4);
    });
  }
});
