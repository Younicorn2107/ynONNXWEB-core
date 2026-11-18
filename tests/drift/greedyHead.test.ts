import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import { LOGSUMEXP_SUFFIX, TOP_IDS_SUFFIX, TOP_VALUES_SUFFIX, greedyHead } from "../../src/models/argmaxHead.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "models");

/** The k best of a row, ties to the lower id, as the head must give them. */
function topOf(row: readonly number[], k: number): { ids: number[]; values: number[] } {
  const order = row.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k);
  return { ids: order.map(([, i]) => i), values: order.map(([v]) => v) };
}

function logSumExp(row: readonly number[]): number {
  const max = Math.max(...row);
  return max + Math.log(row.reduce((s, v) => s + Math.exp(v - max), 0));
}

describe("a decoder's logits with the best candidates beside them (greedyHead)", () => {
  for (const opset of [13, 18]) {
    it(`gives the k best, their scores and the log-sum-exp, the output kept (opset ${opset})`, async () => {
      const original = new Uint8Array(readFileSync(join(MODELS, `softmax_head_o${opset}.onnx`)));
      const { bytes, rewritten } = greedyHead(original, "probs", { candidates: 3 });
      expect(rewritten).toBe(true);
      const x = Float32Array.from({ length: 2 * 3 * 6 }, (_, i) => Math.sin(i * 1.7) * 3);
      x[6] = x[7] = 9;
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      try {
        expect(session.outputNames).toEqual(["probs", `probs${TOP_IDS_SUFFIX}`, `probs${TOP_VALUES_SUFFIX}`, `probs${LOGSUMEXP_SUFFIX}`]);
        const out = await session.run({ x: new ort.Tensor("float32", x, [2, 3, 6]) });
        const probs = Array.from(out["probs"]!.data as Float32Array);
        const ids = Array.from(out[`probs${TOP_IDS_SUFFIX}`]!.data as Float32Array);
        const values = Array.from(out[`probs${TOP_VALUES_SUFFIX}`]!.data as Float32Array);
        const lse = Array.from(out[`probs${LOGSUMEXP_SUFFIX}`]!.data as Float32Array);
        expect(out[`probs${TOP_IDS_SUFFIX}`]!.dims).toEqual([2, 3, 3]);
        expect(out[`probs${LOGSUMEXP_SUFFIX}`]!.dims).toEqual([2, 3, 1]);
        for (let r = 0; r < 6; r += 1) {
          const row = probs.slice(r * 6, r * 6 + 6);
          const expected = topOf(row, 3);
          expect(ids.slice(r * 3, r * 3 + 3)).toEqual(expected.ids);
          values.slice(r * 3, r * 3 + 3).forEach((v, i) => expect(v).toBeCloseTo(expected.values[i]!, 6));
          expect(lse[r]).toBeCloseTo(logSumExp(row), 5);
        }
      } finally {
        await session.release();
      }
    });
  }

  it("takes one candidate when asked, and as many as there are classes at most", async () => {
    const original = new Uint8Array(readFileSync(join(MODELS, "softmax_head_o13.onnx")));
    for (const [asked, kept] of [[1, 1], [50, 6]] as const) {
      const session = await ort.InferenceSession.create(greedyHead(original, "probs", { candidates: asked }).bytes, { executionProviders: ["wasm"] });
      try {
        const x = Float32Array.from({ length: 6 }, (_, i) => i % 3);
        const out = await session.run({ x: new ort.Tensor("float32", x, [1, 1, 6]) });
        expect(out[`probs${TOP_IDS_SUFFIX}`]!.dims).toEqual([1, 1, kept]);
        expect(Array.from(out[`probs${TOP_IDS_SUFFIX}`]!.data as Float32Array)).toEqual(topOf(Array.from(out["probs"]!.data as Float32Array), kept).ids);
      } finally {
        await session.release();
      }
    }
  });

  it("rewrites a decoder with a KV cache, its other outputs untouched", async () => {
    const original = new Uint8Array(readFileSync(join(MODELS, "table_decoder.onnx")));
    const { bytes, rewritten } = greedyHead(original, "logits");
    expect(rewritten).toBe(true);
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.of(4n, 9n, 17n, 2n), [2, 2]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(4).fill(1n), [2, 2]),
      "past_key_values.0.key": new ort.Tensor("float32", new Float32Array(0), [2, 1, 0, 1]),
      "past_key_values.0.value": new ort.Tensor("float32", new Float32Array(0), [2, 1, 0, 1]),
    };
    const run = async (model: Uint8Array) => {
      const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
      try {
        return await session.run(feeds);
      } finally {
        await session.release();
      }
    };
    const [before, after] = [await run(original), await run(bytes)];
    for (const name of ["logits", "present.0.key", "present.0.value"]) expect(after[name]!.data).toEqual(before[name]!.data);
    const logits = Array.from(before["logits"]!.data as Float32Array);
    const ids = Array.from(after[`logits${TOP_IDS_SUFFIX}`]!.data as Float32Array);
    for (let r = 0; r < 4; r += 1) {
      expect(ids.slice(r * 4, r * 4 + 4)).toEqual(topOf(logits.slice(r * 24, r * 24 + 24), 4).ids);
    }
  });

  it("leaves a model without that output, or with no fixed class count, as it was", () => {
    const original = new Uint8Array(readFileSync(join(MODELS, "softmax_head_o13.onnx")));
    expect(greedyHead(original, "logits")).toEqual({ bytes: original, rewritten: false });
  });
});
