/**
 * The protobuf reader every graph rewrite goes through, fed what a network can send: truncated
 * graphs, flipped bytes, garbage, hostile shapes.
 */

import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { ManifestError } from "../../src/errors.js";
import { argmaxHead, greedyHead, replaceArgMax, replaceIsNaN } from "../../src/models/argmaxHead.js";
import { externalizeWeights, splitExternalData } from "../../src/models/externalize.js";
import { gatherBeforeDequantize } from "../../src/models/gatherDequantize.js";
import { castGatherIndices } from "../../src/models/gatherIndices.js";
import { setNBitsAccuracy } from "../../src/models/nbitsAccuracy.js";
import { readGraphInfo } from "../../src/models/onnxProto.js";
import { padMatMulNBitsRows } from "../../src/models/padNBits.js";
import { MAX_GRAPH_DEPTH, encode, operatorTypes, packedInts, parse, readVarint, type Field } from "../../src/models/protobuf.js";
import { repackMatMulNBits } from "../../src/models/repackNBits.js";
import { splitMergedDecoder } from "../../src/models/splitMerged.js";

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");
const models = [
  ...readdirSync(join(FIXTURES, "models")).filter((f) => f.endsWith(".onnx")).map((f) => join(FIXTURES, "models", f)),
  join(FIXTURES, "graphs", "decoder_only", "model_q4f16.onnx"),
  join(FIXTURES, "graphs", "encoder_decoder", "decoder_model_merged_q4.onnx"),
].map((path) => new Uint8Array(readFileSync(path)));

/** Every entry point that reads a graph from the network. */
const READERS: [string, (bytes: Uint8Array) => unknown][] = [
  ["parse", parse],
  ["operatorTypes", operatorTypes],
  ["readGraphInfo", readGraphInfo],
  ["gatherBeforeDequantize", gatherBeforeDequantize],
  ["repackMatMulNBits", repackMatMulNBits],
  ["padMatMulNBitsRows", padMatMulNBitsRows],
  ["castGatherIndices", castGatherIndices],
  ["splitMergedDecoder", splitMergedDecoder],
  ["setNBitsAccuracy", setNBitsAccuracy],
  ["replaceArgMax", replaceArgMax],
  ["replaceIsNaN", replaceIsNaN],
  ["argmaxHead", (b) => argmaxHead(b, "Y")],
  ["greedyHead", (b) => greedyHead(b, "logits")],
  ["externalizeWeights", (b) => externalizeWeights(b, "w", { minTotal: 0, chunkBytes: 1024 })],
  ["splitExternalData", (b) => splitExternalData(b, "w", 1 << 20)],
];

/** Per call. Generous for a slow CI machine; an unbounded loop never meets it. */
const BUDGET_MS = 1000;

function expectContained(bytes: Uint8Array, label: string): void {
  for (const [name, read] of READERS) {
    const started = performance.now();
    try {
      read(bytes);
    } catch (error) {
      if (!(error instanceof ManifestError)) {
        throw new Error(`${name} on ${label} threw ${(error as Error)?.name}: ${(error as Error)?.message}`);
      }
    }
    const took = performance.now() - started;
    expect(took, `${name} on ${label} took ${Math.round(took)} ms`).toBeLessThan(BUDGET_MS);
  }
}

/** A small deterministic generator: the same inputs on every run. */
function prng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

describe("readVarint", () => {
  it("throws on a varint cut short instead of reading past the end (used to spin forever)", () => {
    expect(() => readVarint(new Uint8Array([0x80]), 0)).toThrow(ManifestError);
    expect(() => parse(new Uint8Array([0x80]))).toThrow(ManifestError);
  });

  it("refuses an eleventh byte, so a run of 0xFF never builds an unbounded BigInt", () => {
    expect(() => readVarint(new Uint8Array(1_000_000).fill(0xff), 0)).toThrow(/too long/);

    const minusOne = new Uint8Array([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]);
    expect(readVarint(minusOne, 0)).toEqual([2n ** 64n - 1n, 10]);
  });

  it("is what packed ints are read with", () => {
    expect(packedInts(new Uint8Array([0x01, 0x96, 0x01]))).toEqual([1, 150]);
    expect(() => packedInts(new Uint8Array([0x01, 0x96]))).toThrow(ManifestError);
  });
});

describe("parse", () => {
  it("refuses a length-delimited field that runs past its message, rather than clamping it", () => {
    expect(() => parse(new Uint8Array([0x0a, 100, 1, 2, 3]))).toThrow(/past the end/);
  });

  it("refuses fixed-size fields cut short and unknown wire types", () => {
    expect(() => parse(new Uint8Array([0x09, 1, 2]))).toThrow(ManifestError);
    expect(() => parse(new Uint8Array([0x0d, 1]))).toThrow(ManifestError);
    expect(() => parse(new Uint8Array([0x0b]))).toThrow(/wire type 3/);
  });

  it("refuses a message read from a field that is not length-delimited", () => {
    expect(() => parse((undefined as unknown) as Uint8Array)).toThrow(ManifestError);

    expect(() => gatherBeforeDequantize(encode([{ num: 7, wire: 0, int: 5n }]))).not.toThrow(TypeError);
  });
});

describe("subgraphs nested without end", () => {
  /** A model whose graph holds an If whose branch holds an If… `depth` times. */
  function nested(depth: number, op = "If"): Uint8Array {
    const text = (s: string) => new TextEncoder().encode(s);
    let graph = encode([{ num: 1, wire: 2, data: encode([{ num: 4, wire: 2, data: text("Gather") }]) }]);
    for (let i = 0; i < depth; i += 1) {
      const attribute: Field[] = [{ num: 1, wire: 2, data: text("then_branch") }, { num: 6, wire: 2, data: graph }];
      const node: Field[] = [{ num: 4, wire: 2, data: text(op) }, { num: 5, wire: 2, data: encode(attribute) }];
      graph = encode([{ num: 1, wire: 2, data: encode(node) }]);
    }
    return encode([{ num: 7, wire: 2, data: graph }]);
  }

  it("are walked to a sane depth", () => {
    expect(operatorTypes(nested(3))).toEqual(new Set(["If", "Gather"]));
  });

  it(`stop at ${MAX_GRAPH_DEPTH} levels with a ManifestError, in every walker, not a stack overflow`, () => {
    const deep = nested(5000);
    for (const [name, read] of READERS) {
      try {
        read(deep);
      } catch (error) {
        expect(error, name).toBeInstanceOf(ManifestError);
      }
    }
    expect(() => operatorTypes(nested(MAX_GRAPH_DEPTH + 1))).toThrow(/nested/);

    for (const op of ["MatMulNBits", "Gather"]) {
      const model = nested(MAX_GRAPH_DEPTH + 5, op);
      expect(() => setNBitsAccuracy(model), op).toThrow(ManifestError);
    }
  });
});

describe("graphs from the network, damaged", () => {
  it("every truncation of every fixture is read or refused, quickly", () => {
    for (const [index, model] of models.entries()) {
      const step = Math.max(1, Math.floor(model.length / 400));
      for (let cut = 0; cut < model.length; cut += step) expectContained(model.slice(0, cut), `fixture ${index} cut at ${cut}`);
    }
  }, 120000);

  it("random byte changes are read or refused, quickly", () => {
    const random = prng(0x5eed);
    for (const [index, model] of models.entries()) {
      for (let trial = 0; trial < 60; trial += 1) {
        const bytes = model.slice();
        const flips = 1 + Math.floor(random() * 4);
        for (let f = 0; f < flips; f += 1) {
          const at = Math.floor(random() * bytes.length);

          bytes[at] = random() < 0.5 ? 0xff : Math.floor(random() * 256);
        }
        expectContained(bytes, `fixture ${index}, trial ${trial}`);
      }
    }
  }, 120000);

  it("garbage is read or refused, quickly", () => {
    const random = prng(42);
    for (let trial = 0; trial < 200; trial += 1) {
      const bytes = new Uint8Array(1 + Math.floor(random() * 2048));
      for (let i = 0; i < bytes.length; i += 1) bytes[i] = Math.floor(random() * 256);
      expectContained(bytes, `garbage ${trial}`);
    }
  }, 120000);
});
