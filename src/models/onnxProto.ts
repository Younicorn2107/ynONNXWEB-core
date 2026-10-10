/**
 * Read what a graph declares — its inputs, outputs and external data — from the ONNX protobuf
 * itself, without building a session.
 */

import { ManifestError } from "../errors.js";

/** One dimension: a fixed size, a symbolic name, or unknown. */
export type GraphDim = number | string | null;

export interface GraphValue {
  name: string;
  /** ONNX `TensorProto.DataType` code: 1 float32, 10 float16, 7 int64… */
  elemType: number;
  dims: GraphDim[];
}

export interface GraphInfo {
  inputs: GraphValue[];
  outputs: GraphValue[];
  /** Sidecar files the initializers point at, in the order found. */
  externalData: string[];
  /**
   * Bytes of the largest weight, from its declared shape and type. WebGPU binds one tensor per
   * buffer, and a buffer has an adapter limit: an embedding table over it fails inside the runtime,
   * after the download.
   */
  largestInitializerBytes: number;
  /**
   * The model's `metadata_props`: what an exporter wrote beside the graph — Ultralytics puts its
   * class names, task, `end2end` and `kpt_shape` there.
   */
  metadata: Record<string, string>;
}

/** Bytes per element by ONNX type; sub-byte types round to a fraction. */
const ELEMENT_BYTES: Readonly<Record<number, number>> = {
  1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2,
  17: 1, 18: 1, 19: 1, 20: 1, 21: 0.5, 22: 0.5, 23: 0.5,
};

/** ONNX element types, by the names the manifest uses. */
export const ONNX_DTYPES: Readonly<Record<number, string>> = {
  1: "float32",
  2: "uint8",
  3: "int8",
  5: "int16",
  6: "int32",
  7: "int64",
  9: "bool",
  10: "float16",
  11: "float64",
  16: "bfloat16",
};

class Reader {
  pos: number;

  constructor(
    readonly bytes: Uint8Array,
    start = 0,
    readonly end = bytes.length,
  ) {
    this.pos = start;
  }

  get done(): boolean {
    return this.pos >= this.end;
  }

  varint(): number {
    let result = 0;
    let scale = 1;
    for (let shift = 0; shift < 70; shift += 7) {
      if (this.pos >= this.end) throw new ManifestError("truncated ONNX graph (varint)");
      const byte = this.bytes[this.pos++]!;
      result += (byte & 0x7f) * scale;
      if ((byte & 0x80) === 0) return result;
      scale *= 128;
    }
    throw new ManifestError("malformed ONNX graph (varint too long)");
  }

  /** A signed int64 varint, as protobuf writes negative numbers. */
  int64(): number {
    const value = this.varint();
    return value >= 2 ** 63 ? value - 2 ** 64 : value;
  }

  /** The bounds of a length-delimited field, consumed. */
  slice(): [number, number] {
    const length = this.varint();
    const start = this.pos;
    const end = start + length;
    if (end > this.end) throw new ManifestError("truncated ONNX graph (length-delimited field)");
    this.pos = end;
    return [start, end];
  }

  string(): string {
    const [start, end] = this.slice();
    return new TextDecoder().decode(this.bytes.subarray(start, end));
  }

  skip(wireType: number): void {
    switch (wireType) {
      case 0:
        this.varint();
        return;
      case 1:
        this.pos += 8;
        return;
      case 2:
        this.slice();
        return;
      case 5:
        this.pos += 4;
        return;
      default:
        throw new ManifestError(`malformed ONNX graph (wire type ${wireType})`);
    }
  }

  /** Walk the fields of the message between `pos` and `end`. */
  *fields(): Generator<[field: number, wireType: number]> {
    while (!this.done) {
      const key = this.varint();
      yield [Math.floor(key / 8), key & 7];
    }
  }

  sub([start, end]: [number, number]): Reader {
    return new Reader(this.bytes, start, end);
  }
}

function readDim(reader: Reader): GraphDim {
  let dim: GraphDim = null;
  for (const [field, wire] of reader.fields()) {
    if (field === 1 && wire === 0) dim = reader.int64();
    else if (field === 2 && wire === 2) dim = reader.string();
    else reader.skip(wire);
  }
  return dim;
}

function readShape(reader: Reader): GraphDim[] {
  const dims: GraphDim[] = [];
  for (const [field, wire] of reader.fields()) {
    if (field === 1 && wire === 2) dims.push(readDim(reader.sub(reader.slice())));
    else reader.skip(wire);
  }
  return dims;
}

function readValue(reader: Reader): GraphValue {
  const value: GraphValue = { name: "", elemType: 0, dims: [] };
  for (const [field, wire] of reader.fields()) {
    if (field === 1 && wire === 2) {
      value.name = reader.string();
    } else if (field === 2 && wire === 2) {
      const type = reader.sub(reader.slice());
      for (const [typeField, typeWire] of type.fields()) {
        if (typeField !== 1 || typeWire !== 2) {
          type.skip(typeWire);
          continue;
        }
        const tensor = type.sub(type.slice());
        for (const [tensorField, tensorWire] of tensor.fields()) {
          if (tensorField === 1 && tensorWire === 0) value.elemType = tensor.varint();
          else if (tensorField === 2 && tensorWire === 2) value.dims = readShape(tensor.sub(tensor.slice()));
          else tensor.skip(tensorWire);
        }
      }
    } else {
      reader.skip(wire);
    }
  }
  return value;
}

/** An initializer's external `location` (or null) and its size in bytes. */
function readInitializer(reader: Reader): { location: string | null; bytes: number } {
  let location: string | null = null;
  const dims: number[] = [];
  let type = 0;
  for (const [field, wire] of reader.fields()) {
    if (field === 1 && wire === 0) {
      dims.push(reader.int64());
      continue;
    }
    if (field === 1 && wire === 2) {
      const packed = reader.sub(reader.slice());
      while (!packed.done) dims.push(packed.int64());
      continue;
    }
    if (field === 2 && wire === 0) {
      type = reader.varint();
      continue;
    }
    if (field !== 13 || wire !== 2) {
      reader.skip(wire);
      continue;
    }
    const entry = reader.sub(reader.slice());
    let key = "";
    let value = "";
    for (const [entryField, entryWire] of entry.fields()) {
      if (entryField === 1 && entryWire === 2) key = entry.string();
      else if (entryField === 2 && entryWire === 2) value = entry.string();
      else entry.skip(entryWire);
    }
    if (key === "location") location = value;
  }
  const count = dims.reduce((product, dim) => product * Math.max(dim, 0), 1);
  return { location, bytes: count * (ELEMENT_BYTES[type] ?? 4) };
}

/** Inputs, outputs and sidecars of a serialized `ModelProto`. */
export function readGraphInfo(model: Uint8Array): GraphInfo {
  const top = new Reader(model);
  let graph: [number, number] | null = null;
  const metadata: Record<string, string> = {};
  for (const [field, wire] of top.fields()) {
    if (field === 7 && wire === 2) graph = top.slice();
    else if (field === 14 && wire === 2) {
      const entry = top.sub(top.slice());
      let key = "";
      let value = "";
      for (const [f, w] of entry.fields()) {
        if (f === 1 && w === 2) key = entry.string();
        else if (f === 2 && w === 2) value = entry.string();
        else entry.skip(w);
      }
      if (key) metadata[key] = value;
    } else top.skip(wire);
  }
  if (!graph) throw new ManifestError("not an ONNX model: no graph in the protobuf");

  const info: GraphInfo = { inputs: [], outputs: [], externalData: [], largestInitializerBytes: 0, metadata };
  const reader = top.sub(graph);
  for (const [field, wire] of reader.fields()) {
    if (field === 11 && wire === 2) {
      info.inputs.push(readValue(reader.sub(reader.slice())));
    } else if (field === 12 && wire === 2) {
      info.outputs.push(readValue(reader.sub(reader.slice())));
    } else if (field === 5 && wire === 2) {
      const { location, bytes } = readInitializer(reader.sub(reader.slice()));
      info.largestInitializerBytes = Math.max(info.largestInitializerBytes, bytes);
      if (location) {
        const name = location.split(/[\\/]/).pop() ?? "";
        if (name && !info.externalData.includes(name)) info.externalData.push(name);
      }
    } else {
      reader.skip(wire);
    }
  }

  return info;
}
