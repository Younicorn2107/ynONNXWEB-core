/** `MatMulNBits` with zero points, padded to whole 32-bit words for WebGPU. */

import { encode, operatorTypes, parse, readVarint, str, writeVarint, type Field } from "./protobuf.js";

const MODEL_GRAPH = 7;
const GRAPH_NODE = 1;
const GRAPH_INITIALIZER = 5;
const NODE_INPUT = 1;
const NODE_OUTPUT = 2;
const NODE_NAME = 3;
const NODE_OP_TYPE = 4;
const NODE_ATTRIBUTE = 5;
const ATTR_NAME = 1;
const ATTR_I = 3;
const ATTR_G = 6;
const ATTR_GRAPHS = 11;
const TENSOR_DIMS = 1;
const TENSOR_DATA_TYPE = 2;
const TENSOR_FLOAT_DATA = 4;
const TENSOR_INT32_DATA = 5;
const TENSOR_NAME = 8;
const TENSOR_RAW_DATA = 9;
const TENSOR_DATA_LOCATION = 14;
const FLOAT16 = 10;
const INT64 = 7;

export const PADDED_SUFFIX = "__ynonnx_padded";

interface Tensor {
  fields: Field[];
  name: string;
  dims: number[];
  dataType: number;
}

function readTensor(bytes: Uint8Array): Tensor {
  const fields = parse(bytes);
  const dims: number[] = [];
  for (const f of fields) {
    if (f.num !== TENSOR_DIMS) continue;
    if (f.wire === 0) dims.push(Number(f.int!));
    else {
      let at = 0;
      while (at < f.data!.length) {
        const [v, next] = readVarint(f.data!, at);
        dims.push(Number(v));
        at = next;
      }
    }
  }
  return { fields, name: str(fields.find((f) => f.num === TENSOR_NAME)), dims, dataType: Number(fields.find((f) => f.num === TENSOR_DATA_TYPE)?.int ?? 0n) };
}

/** Raw bytes of an inline tensor (float16 bits from int32_data rebuilt); null when external. */
function rawOf(t: Tensor): Uint8Array | null {
  if (t.fields.some((f) => f.num === TENSOR_DATA_LOCATION && f.int === 1n)) return null;
  const raw = t.fields.find((f) => f.num === TENSOR_RAW_DATA)?.data;
  if (raw) return raw;
  const floats = t.fields.find((f) => f.num === TENSOR_FLOAT_DATA && f.wire === 2)?.data;
  if (floats) return floats;
  const ints = t.fields.find((f) => f.num === TENSOR_INT32_DATA);
  if (ints && t.dataType === FLOAT16) {
    const values: number[] = [];
    let at = 0;
    while (at < ints.data!.length) {
      const [v, next] = readVarint(ints.data!, at);
      values.push(Number(v));
      at = next;
    }
    return new Uint8Array(Uint16Array.from(values).buffer);
  }
  return null;
}

function withRows(t: Tensor, dims: readonly number[], raw: Uint8Array): Uint8Array {
  const dimBytes: number[] = [];
  for (const d of dims) writeVarint(BigInt(d), dimBytes);
  const fields: Field[] = [{ num: TENSOR_DIMS, wire: 2, data: Uint8Array.from(dimBytes) }];
  for (const f of t.fields) {
    if (f.num === TENSOR_DIMS || f.num === TENSOR_RAW_DATA || f.num === TENSOR_FLOAT_DATA || f.num === TENSOR_INT32_DATA) continue;
    fields.push(f);
  }
  fields.push({ num: TENSOR_RAW_DATA, wire: 2, data: raw });
  return encode(fields);
}

function padded(raw: Uint8Array, extra: number): Uint8Array {
  const out = new Uint8Array(raw.length + extra);
  out.set(raw);
  return out;
}

const bytesOf = (text: string) => new TextEncoder().encode(text);

function int64Tensor(name: string, value: number): Field {
  const raw = new Uint8Array(8);
  new DataView(raw.buffer).setBigInt64(0, BigInt(value), true);
  const dims: number[] = [];
  writeVarint(1n, dims);
  return {
    num: GRAPH_INITIALIZER,
    wire: 2,
    data: encode([
      { num: TENSOR_DIMS, wire: 2, data: Uint8Array.from(dims) },
      { num: TENSOR_DATA_TYPE, wire: 0, int: BigInt(INT64) },
      { num: TENSOR_NAME, wire: 2, data: bytesOf(name) },
      { num: TENSOR_RAW_DATA, wire: 2, data: raw },
    ]),
  };
}

interface State {
  /** New bytes of each padded initializer, by name. */
  tensors: Map<string, Uint8Array>;
  /** The padded row count of each padded weight, by name. */
  rows: Map<string, number>;
  count: number;
}

function attrOf(node: Field[], name: string): [number, number] | null {
  for (const [i, f] of node.entries()) {
    if (f.num !== NODE_ATTRIBUTE) continue;
    const attr = parse(f.data!);
    if (str(attr.find((a) => a.num === ATTR_NAME)) === name) return [Number(attr.find((a) => a.num === ATTR_I)?.int ?? 0n), i];
  }
  return null;
}

function rewriteGraph(graphBytes: Uint8Array, outer: ReadonlyMap<string, Tensor>, state: State): Uint8Array {
  const fields = parse(graphBytes);
  const local = new Map(outer);
  const initializerAt = new Map<string, number>();
  fields.forEach((f, i) => {
    if (f.num !== GRAPH_INITIALIZER) return;
    const t = readTensor(f.data!);
    local.set(t.name, t);
    initializerAt.set(t.name, i);
  });
  const out: Field[] = [];
  const added: Field[] = [];
  let changed = false;
  for (const f of fields) {
    if (f.num !== GRAPH_NODE) {
      out.push(f);
      continue;
    }
    const node = parse(f.data!);
    let nodeChanged = false;
    for (const [j, attrField] of node.entries()) {
      if (attrField.num !== NODE_ATTRIBUTE) continue;
      const attr = parse(attrField.data!);
      if (!attr.some((a) => a.num === ATTR_G || a.num === ATTR_GRAPHS)) continue;
      let attrChanged = false;
      for (const [k, a] of attr.entries()) {
        if (a.num !== ATTR_G && a.num !== ATTR_GRAPHS) continue;
        const rewritten = rewriteGraph(a.data!, local, state);
        if (rewritten !== a.data) {
          attr[k] = { ...a, data: rewritten };
          attrChanged = true;
        }
      }
      if (attrChanged) {
        node[j] = { ...attrField, data: encode(attr) };
        nodeChanged = true;
      }
    }
    let slice: Field | null = null;
    if (str(node.find((x) => x.num === NODE_OP_TYPE)) === "MatMulNBits") {
      const result = padNode(node, local, state);
      if (result) {
        slice = result.slice;
        added.push(...result.initializers);
        nodeChanged = true;
      }
    }
    if (nodeChanged) {
      out.push({ ...f, data: encode(node) });
      changed = true;
    } else {
      out.push(f);
    }
    if (slice) out.push(slice);
  }
  for (const [name, index] of initializerAt) {
    const bytes = state.tensors.get(name);
    if (bytes) {
      out[out.indexOf(fields[index]!)] = { ...fields[index]!, data: bytes };
      changed = true;
    }
  }
  if (added.length > 0) out.push(...added);
  return changed ? encode(out) : graphBytes;
}

function padNode(node: Field[], tensors: ReadonlyMap<string, Tensor>, state: State): { slice: Field; initializers: Field[] } | null {
  const inputs = node.filter((f) => f.num === NODE_INPUT).map(str);
  const n = attrOf(node, "N");
  const k = attrOf(node, "K");
  const bits = attrOf(node, "bits")?.[0] ?? 4;
  const block = attrOf(node, "block_size")?.[0] ?? 0;
  if (!n || !k || bits !== 4 || block <= 0 || !inputs[3] || (inputs.length > 4 && inputs[4])) return null;
  const blocks = Math.ceil(k[0] / block);
  const zpRow = Math.ceil(blocks / 2);
  const rows = n[0];
  if ((rows * zpRow) % 4 === 0) return null;
  let target = rows;
  while ((target * zpRow) % 4 !== 0) target += 1;
  const weights = tensors.get(inputs[1] ?? "");
  const scales = tensors.get(inputs[2] ?? "");
  const zeros = tensors.get(inputs[3]);
  if (!weights || !scales || !zeros || zeros.dataType !== 2) return null;
  const done = state.rows.get(weights.name);
  if (done === undefined) {
    const w = rawOf(weights);
    const s = rawOf(scales);
    const z = rawOf(zeros);
    if (!w || !s || !z) return null;
    const blob = block / 2;
    const scaleSize = s.length / (rows * blocks);
    if (w.length !== rows * blocks * blob || z.length !== rows * zpRow || !Number.isInteger(scaleSize)) return null;
    const extra = target - rows;
    state.tensors.set(weights.name, withRows(weights, [target, blocks, blob], padded(w, extra * blocks * blob)));
    state.tensors.set(scales.name, withRows(scales, [target * blocks], padded(s, extra * blocks * scaleSize)));
    state.tensors.set(zeros.name, withRows(zeros, [target * zpRow], padded(z, extra * zpRow)));
    state.rows.set(weights.name, target);
  } else if (done !== target) {
    return null;
  }

  const [, nAt] = n;
  const nAttr = parse(node[nAt]!.data!).map((a) => (a.num === ATTR_I ? { ...a, int: BigInt(target) } : a));
  node[nAt] = { ...node[nAt]!, data: encode(nAttr) };
  const outAt = node.findIndex((f) => f.num === NODE_OUTPUT);
  const output = str(node[outAt]);
  const id = `${output}${PADDED_SUFFIX}${state.count}`;
  node[outAt] = { ...node[outAt]!, data: bytesOf(id) };
  state.count += 1;
  const names = { starts: `${id}_starts`, ends: `${id}_ends`, axes: `${id}_axes` };
  const slice: Field = {
    num: GRAPH_NODE,
    wire: 2,
    data: encode([
      { num: NODE_INPUT, wire: 2, data: bytesOf(id) },
      { num: NODE_INPUT, wire: 2, data: bytesOf(names.starts) },
      { num: NODE_INPUT, wire: 2, data: bytesOf(names.ends) },
      { num: NODE_INPUT, wire: 2, data: bytesOf(names.axes) },
      { num: NODE_OUTPUT, wire: 2, data: bytesOf(output) },
      { num: NODE_NAME, wire: 2, data: bytesOf(`${id}_slice`) },
      { num: NODE_OP_TYPE, wire: 2, data: bytesOf("Slice") },
    ]),
  };
  return { slice, initializers: [int64Tensor(names.starts, 0), int64Tensor(names.ends, rows), int64Tensor(names.axes, -1)] };
}

/**
 * The model with each `MatMulNBits` whose packed zero points do not fill whole 32-bit words padded
 * by up to three zero rows, and how many nodes changed. The same bytes come back when none needed
 * it.
 */
export function padMatMulNBitsRows(model: Uint8Array): { bytes: Uint8Array; padded: number } {
  if (!operatorTypes(model).has("MatMulNBits")) return { bytes: model, padded: 0 };
  const fields = parse(model);
  const state: State = { tensors: new Map(), rows: new Map(), count: 0 };
  let changed = false;
  for (const [i, f] of fields.entries()) {
    if (f.num !== MODEL_GRAPH) continue;
    const graph = rewriteGraph(f.data!, new Map(), state);
    if (graph !== f.data) {
      fields[i] = { ...f, data: graph };
      changed = true;
    }
  }
  return changed ? { bytes: encode(fields), padded: state.count } : { bytes: model, padded: 0 };
}
