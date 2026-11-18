/** Shapes a graph merely declares, against the bytes it holds. */

import { describe, expect, it } from "vitest";

import { ManifestError } from "../../src/errors.js";
import { argmaxHead, greedyHead } from "../../src/models/argmaxHead.js";
import { gatherBeforeDequantize } from "../../src/models/gatherDequantize.js";
import { readGraphInfo } from "../../src/models/onnxProto.js";
import { padMatMulNBitsRows } from "../../src/models/padNBits.js";
import { encode, type Field } from "../../src/models/protobuf.js";
import { repackMatMulNBits } from "../../src/models/repackNBits.js";

const text = (s: string) => new TextEncoder().encode(s);

function tensor(name: string, dataType: number, dims: (number | bigint)[], raw: Uint8Array): Field {
  return {
    num: 5,
    wire: 2,
    data: encode([
      ...dims.map((d) => ({ num: 1, wire: 0, int: BigInt(d) })),
      { num: 2, wire: 0, int: BigInt(dataType) },
      { num: 8, wire: 2, data: text(name) },
      { num: 9, wire: 2, data: raw },
    ]),
  };
}

function attrInt(name: string, value: bigint): Field {
  return { num: 5, wire: 2, data: encode([{ num: 1, wire: 2, data: text(name) }, { num: 3, wire: 0, int: value }, { num: 20, wire: 0, int: 2n }]) };
}

function node(op: string, inputs: string[], outputs: string[], attributes: Field[] = []): Field {
  return {
    num: 1,
    wire: 2,
    data: encode([
      ...inputs.map((i) => ({ num: 1, wire: 2, data: text(i) })),
      ...outputs.map((o) => ({ num: 2, wire: 2, data: text(o) })),
      { num: 4, wire: 2, data: text(op) },
      ...attributes,
    ]),
  };
}

function model(graph: Field[]): Uint8Array {
  const opset = encode([{ num: 1, wire: 2, data: new Uint8Array() }, { num: 2, wire: 0, int: 17n }]);
  return encode([{ num: 8, wire: 2, data: opset }, { num: 7, wire: 2, data: encode(graph) }]);
}

/** A graph output of float type with these dims. */
function output(name: string, dims: bigint[]): Field {
  const shape = encode(dims.map((d) => ({ num: 1, wire: 2, data: encode([{ num: 1, wire: 0, int: d }]) })));
  const tensorType = encode([{ num: 1, wire: 0, int: 1n }, { num: 2, wire: 2, data: shape }]);
  return { num: 12, wire: 2, data: encode([{ num: 1, wire: 2, data: text(name) }, { num: 2, wire: 2, data: encode([{ num: 1, wire: 2, data: tensorType }]) }]) };
}

const quick = <T>(run: () => T): T => {
  const started = performance.now();
  const out = run();
  expect(performance.now() - started).toBeLessThan(1000);
  return out;
};

describe("rewrites check what a graph claims against what it holds", () => {
  it("gatherBeforeDequantize: a table declared [1e6, 1e6] with ten bytes is not transposed (was 10¹² iterations)", () => {
    const graph = [
      tensor("W", 3, [1_000_000, 1_000_000], new Uint8Array(10)),
      tensor("s", 1, [1_000_000], new Uint8Array(16)),
      node("DequantizeLinear", ["W", "s"], ["table"], [attrInt("axis", 0n)]),
      node("Transpose", ["table"], ["head"]),
    ];
    expect(quick(() => gatherBeforeDequantize(model(graph))).rewritten).toBe(0);
  });

  it("padMatMulNBitsRows: an N past 2^53, where `target += 1` stops being exact, is left alone", () => {
    const graph = [
      tensor("W", 2, [1, 1, 16], new Uint8Array(16)),
      tensor("S", 1, [1], new Uint8Array(4)),
      tensor("Z", 2, [1], new Uint8Array(1)),
      node("MatMulNBits", ["A", "W", "S", "Z"], ["Y"], [attrInt("N", 2n ** 60n + 1n), attrInt("K", 32n), attrInt("bits", 4n), attrInt("block_size", 32n)]),
    ];
    expect(quick(() => padMatMulNBitsRows(model(graph))).padded).toBe(0);
  });

  it("repackMatMulNBits: a block size of 2^40 over sixteen bytes of weights is left alone (was a buffer of that size)", () => {
    const graph = [
      tensor("W", 2, [1, 1, 2 ** 39], new Uint8Array(16)),
      tensor("S", 1, [1], new Uint8Array(4)),
      node("MatMulNBits", ["A", "W", "S"], ["Y"], [attrInt("N", 1n), attrInt("K", 2n ** 40n), attrInt("bits", 4n), attrInt("block_size", 2n ** 40n)]),
    ];
    expect(quick(() => repackMatMulNBits(model(graph))).rewritten).toBe(0);
  });

  it("repackMatMulNBits: zero points shorter than their dims say are left alone", () => {
    const graph = [
      tensor("W", 2, [4, 1, 32], new Uint8Array(128)),
      tensor("S", 1, [4], new Uint8Array(16)),
      tensor("Z", 2, [4], new Uint8Array(1)),
      node("MatMulNBits", ["A", "W", "S", "Z"], ["Y"], [attrInt("N", 4n), attrInt("K", 64n), attrInt("bits", 4n), attrInt("block_size", 64n)]),
    ];
    expect(quick(() => repackMatMulNBits(model(graph))).rewritten).toBe(0);
  });

  it("argmaxHead and greedyHead: a class count no buffer could hold is refused, not allocated", () => {
    const graph = [node("Identity", ["X"], ["Y"]), output("Y", [1n, 2n ** 40n])];
    expect(() => quick(() => argmaxHead(model(graph), "Y"))).toThrow(ManifestError);
    expect(() => quick(() => greedyHead(model(graph), "Y"))).toThrow(ManifestError);
  });

  it("readGraphInfo: an external-data location that reduces to a dot segment is refused", () => {
    const location = (value: string): Field => ({
      num: 13,
      wire: 2,
      data: encode([{ num: 1, wire: 2, data: text("location") }, { num: 2, wire: 2, data: text(value) }]),
    });
    const external = (value: string) =>
      model([{ num: 5, wire: 2, data: encode([{ num: 1, wire: 0, int: 4n }, { num: 2, wire: 0, int: 1n }, { num: 8, wire: 2, data: text("w") }, location(value), { num: 14, wire: 0, int: 1n }]) }]);
    expect(readGraphInfo(external("../../model.onnx_data")).externalData).toEqual(["model.onnx_data"]);
    expect(() => readGraphInfo(external("weights/.."))).toThrow(ManifestError);
    expect(() => readGraphInfo(external(".hidden"))).toThrow(ManifestError);
    expect(() => readGraphInfo(external("stored_digests.json"))).toThrow(ManifestError);
  });
});
