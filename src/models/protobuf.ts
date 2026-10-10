/**
 * The few protobuf primitives the graph rewrites need: a message as a flat list of fields (unknown
 * fields kept as they are), and back to bytes.
 */

export interface Field {
  num: number;
  wire: number;
  /** Varint value, for wire 0. */
  int?: bigint;
  /** Bytes, for wires 1, 2 and 5. */
  data?: Uint8Array;
}

export function readVarint(bytes: Uint8Array, at: number): [bigint, number] {
  let result = 0n;
  let shift = 0n;
  for (;;) {
    const byte = bytes[at++]!;
    result |= BigInt(byte & 0x7f) << shift;
    if (byte < 0x80) return [result, at];
    shift += 7n;
  }
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

export function parse(bytes: Uint8Array): Field[] {
  const fields: Field[] = [];
  let at = 0;
  while (at < bytes.length) {
    const [key, next] = readVarint(bytes, at);
    at = next;
    const num = Number(key >> 3n);
    const wire = Number(key & 7n);
    if (wire === 0) {
      const [value, after] = readVarint(bytes, at);
      fields.push({ num, wire, int: value });
      at = after;
    } else if (wire === 1) {
      fields.push({ num, wire, data: bytes.subarray(at, at + 8) });
      at += 8;
    } else if (wire === 5) {
      fields.push({ num, wire, data: bytes.subarray(at, at + 4) });
      at += 4;
    } else if (wire === 2) {
      const [length, after] = readVarint(bytes, at);
      const end = after + Number(length);
      fields.push({ num, wire, data: bytes.subarray(after, end) });
      at = end;
    } else {
      throw new Error(`unsupported protobuf wire type ${wire}`);
    }
  }
  return fields;
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

const opsCache = new WeakMap<Uint8Array, ReadonlySet<string>>();

/** Every operator type a serialized `ModelProto` uses, subgraphs included. */
export function operatorTypes(model: Uint8Array): ReadonlySet<string> {
  const known = opsCache.get(model);
  if (known) return known;
  const found = new Set<string>();
  const walkGraph = (graph: Uint8Array) => {
    for (const f of parse(graph)) {
      if (f.num !== 1 || !f.data) continue;
      for (const n of parse(f.data)) {
        if (n.num === 4) found.add(str(n));
        else if (n.num === 5 && n.data) {
          for (const a of parse(n.data)) if ((a.num === 6 || a.num === 11) && a.data) walkGraph(a.data);
        }
      }
    }
  };
  for (const f of parse(model)) if (f.num === 7 && f.data) walkGraph(f.data);
  opsCache.set(model, found);
  return found;
}
