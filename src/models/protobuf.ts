/**
 * The few protobuf primitives the graph rewrites need: a message as a flat list of fields (unknown
 * fields kept as they are), and back to bytes.
 */

import { ManifestError } from "../errors.js";

export interface Field {
  num: number;
  wire: number;
  /** Varint value, for wire 0. */
  int?: bigint;
  /** Bytes, for wires 1, 2 and 5. */
  data?: Uint8Array;
}

/** A 64-bit varint takes ten bytes at most; an eleventh is a malformed (or hostile) input. */
export const MAX_VARINT_BYTES = 10;

/**
 * How deep subgraphs may nest (an `If` in a `Loop` in an `If`…). Real exports go two or three
 * levels down; each level of a walk is a stack frame, and a crafted graph a few megabytes long
 * could otherwise nest a million times and overflow the stack.
 */
export const MAX_GRAPH_DEPTH = 32;

export function readVarint(bytes: Uint8Array, at: number, end = bytes.length): [bigint, number] {
  let result = 0n;
  for (let i = 0; i < MAX_VARINT_BYTES; i += 1) {
    if (at >= end) throw new ManifestError("truncated protobuf (varint)");
    const byte = bytes[at++]!;
    result |= BigInt(byte & 0x7f) << BigInt(7 * i);
    if (byte < 0x80) return [BigInt.asUintN(64, result), at];
  }
  throw new ManifestError("malformed protobuf (varint too long)");
}

/** The same varint as a double: exact up to 2⁵³, which every length and shape is. */
export function readVarintNumber(bytes: Uint8Array, at: number, end = bytes.length): [number, number] {
  let result = 0;
  let scale = 1;
  for (let i = 0; i < MAX_VARINT_BYTES; i += 1) {
    if (at >= end) throw new ManifestError("truncated protobuf (varint)");
    const byte = bytes[at++]!;
    result += (byte & 0x7f) * scale;
    if (byte < 0x80) return [result, at];
    scale *= 128;
  }
  throw new ManifestError("malformed protobuf (varint too long)");
}

export function writeVarint(value: bigint, out: number[]): void {
  let v = BigInt.asUintN(64, value);
  for (;;) {
    const byte = Number(v & 0x7fn);
    v >>= 7n;
    if (v === 0n) {
      out.push(byte);
      return;
    }
    out.push(byte | 0x80);
  }
}

/**
 * The end of a fixed-size or length-delimited value starting at `at`, checked against the message.
 */
function within(at: number, length: number, end: number, what: string): number {
  const stop = at + length;
  if (stop > end) throw new ManifestError(`truncated protobuf (${what} runs past the end of its message)`);
  return stop;
}

export function parse(bytes: Uint8Array): Field[] {
  if (!(bytes instanceof Uint8Array)) throw new ManifestError("malformed protobuf (a message field is not length-delimited)");
  const fields: Field[] = [];
  const end = bytes.length;
  let at = 0;
  while (at < end) {
    const [key, next] = readVarint(bytes, at, end);
    at = next;
    const num = Number(key >> 3n);
    const wire = Number(key & 7n);
    if (num === 0) throw new ManifestError("malformed protobuf (field number 0)");
    if (wire === 0) {
      const [value, after] = readVarint(bytes, at, end);
      fields.push({ num, wire, int: value });
      at = after;
    } else if (wire === 1 || wire === 5) {
      const stop = within(at, wire === 1 ? 8 : 4, end, "a fixed-size field");
      fields.push({ num, wire, data: bytes.subarray(at, stop) });
      at = stop;
    } else if (wire === 2) {
      const [length, after] = readVarintNumber(bytes, at, end);
      const stop = within(after, length, end, "a length-delimited field");
      fields.push({ num, wire, data: bytes.subarray(after, stop) });
      at = stop;
    } else {
      throw new ManifestError(`malformed protobuf (wire type ${wire})`);
    }
  }
  return fields;
}

/** Varints packed in one length-delimited field, as signed 64-bit numbers (dims, ints). */
export function packedInts(data: Uint8Array): number[] {
  const out: number[] = [];
  let at = 0;
  while (at < data.length) {
    const [value, next] = readVarint(data, at);
    out.push(Number(BigInt.asIntN(64, value)));
    at = next;
  }
  return out;
}

/** A repeated int64 field, packed or not, as numbers. */
export function repeatedInts(fields: readonly Field[], num: number): number[] {
  const out: number[] = [];
  for (const f of fields) {
    if (f.num !== num) continue;
    if (f.int !== undefined) out.push(Number(BigInt.asIntN(64, f.int)));
    else if (f.wire === 2 && f.data) out.push(...packedInts(f.data));
  }
  return out;
}

/**
 * How many elements `dims` declare, or null when that is not a count: a negative or fractional
 * dimension, or a product past 2⁵³. What a rewrite checks against the bytes actually present before
 * it loops over a tensor.
 */
export function elementCount(dims: readonly number[]): number | null {
  let count = 1;
  for (const d of dims) {
    if (!Number.isSafeInteger(d) || d < 0) return null;
    count *= d;
    if (!Number.isSafeInteger(count)) return null;
  }
  return count;
}

/** The depth one level further into a subgraph, refused past `MAX_GRAPH_DEPTH`. */
export function deeper(depth: number): number {
  if (depth >= MAX_GRAPH_DEPTH) throw new ManifestError(`malformed ONNX graph (subgraphs nested more than ${MAX_GRAPH_DEPTH} deep)`);
  return depth + 1;
}

export function encode(fields: readonly Field[]): Uint8Array {
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (const field of fields) {
    const head: number[] = [];
    writeVarint(BigInt((field.num << 3) | field.wire), head);
    if (field.wire === 0) writeVarint(field.int!, head);
    else if (field.wire === 2) writeVarint(BigInt(field.data!.length), head);
    const headBytes = Uint8Array.from(head);
    chunks.push(headBytes);
    total += headBytes.length;
    if (field.wire !== 0) {
      chunks.push(field.data!);
      total += field.data!.length;
    }
  }
  const out = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out;
}

const text = new TextDecoder();

/** A string field's value; empty when absent. */
export const str = (field: Field | undefined): string => (field?.data ? text.decode(field.data) : "");

/** Whether `needle` occurs in `bytes`: a cheap test before parsing a whole model. */
export function contains(bytes: Uint8Array, needle: Uint8Array): boolean {
  outer: for (let i = 0; i + needle.length <= bytes.length; i += 1) {
    for (let j = 0; j < needle.length; j += 1) if (bytes[i + j] !== needle[j]) continue outer;
    return true;
  }
  return false;
}

/**
 * A cursor over one message, for a reader that visits a few fields and skips the rest by length
 * (`readGraphInfo`): the same bounds as `parse`, without building the field list.
 */
export class Reader {
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

  /** Up to 64 bits; doubles hold every value a shape or a length can take. */
  varint(): number {
    const [value, next] = readVarintNumber(this.bytes, this.pos, this.end);
    this.pos = next;
    return value;
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
    this.pos = within(start, length, this.end, "a length-delimited field");
    return [start, this.pos];
  }

  string(): string {
    const [start, end] = this.slice();
    return text.decode(this.bytes.subarray(start, end));
  }

  skip(wireType: number): void {
    switch (wireType) {
      case 0:
        this.varint();
        return;
      case 1:
        this.pos = within(this.pos, 8, this.end, "a fixed-size field");
        return;
      case 2:
        this.slice();
        return;
      case 5:
        this.pos = within(this.pos, 4, this.end, "a fixed-size field");
        return;
      default:
        throw new ManifestError(`malformed protobuf (wire type ${wireType})`);
    }
  }

  /** Walk the fields of the message between `pos` and `end`. */
  *fields(): Generator<[field: number, wireType: number]> {
    while (!this.done) {
      const key = this.varint();
      yield [Math.floor(key / 8), key % 8];
    }
  }

  sub([start, end]: [number, number]): Reader {
    return new Reader(this.bytes, start, end);
  }
}

const opsCache = new WeakMap<Uint8Array, ReadonlySet<string>>();

/** Every operator type a serialized `ModelProto` uses, subgraphs included. */
export function operatorTypes(model: Uint8Array): ReadonlySet<string> {
  const known = opsCache.get(model);
  if (known) return known;
  const found = new Set<string>();
  const walkGraph = (graph: Uint8Array, depth: number) => {
    for (const f of parse(graph)) {
      if (f.num !== 1 || !f.data) continue;
      for (const n of parse(f.data)) {
        if (n.num === 4) found.add(str(n));
        else if (n.num === 5 && n.data) {
          for (const a of parse(n.data)) if ((a.num === 6 || a.num === 11) && a.data) walkGraph(a.data, deeper(depth));
        }
      }
    }
  };
  for (const f of parse(model)) if (f.num === 7 && f.data) walkGraph(f.data, 0);
  opsCache.set(model, found);
  return found;
}
