/** `MatMulNBits` brought to blocks of 32 for WebGPU. */

import { encode, operatorTypes, parse, readVarint, str, writeVarint, type Field } from "./protobuf.js";

/** The block size WebGPU computes correctly. */
export const WEBGPU_NBITS_BLOCK = 32;

const MODEL_GRAPH = 7;
const GRAPH_NODE = 1;
const GRAPH_INITIALIZER = 5;
const NODE_INPUT = 1;
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
const FLOAT = 1;
const UINT8 = 2;
const FLOAT16 = 10;

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

/**
 * The tensor's element bytes, one element per `size` bytes; null when not stored inline as raw
 * data.
 */
function rawOf(tensor: Tensor): Uint8Array | null {
  if (tensor.fields.some((f) => f.num === TENSOR_DATA_LOCATION && f.int === 1n)) return null;
  const raw = tensor.fields.find((f) => f.num === TENSOR_RAW_DATA)?.data;
  if (raw) return raw;

  const floats = tensor.fields.find((f) => f.num === TENSOR_FLOAT_DATA && f.wire === 2)?.data;
  if (floats && tensor.dataType === FLOAT) return floats;
  const ints = tensor.fields.find((f) => f.num === TENSOR_INT32_DATA);
  if (ints && tensor.dataType === FLOAT16) {
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

function writeTensor(tensor: Tensor, dims: readonly number[], raw: Uint8Array | null): Uint8Array {
  const dimBytes: number[] = [];
  for (const d of dims) writeVarint(BigInt(d), dimBytes);
  const fields: Field[] = [{ num: TENSOR_DIMS, wire: 2, data: Uint8Array.from(dimBytes) }];
  for (const f of tensor.fields) {
    if (f.num === TENSOR_DIMS) continue;
    if (raw && (f.num === TENSOR_RAW_DATA || f.num === TENSOR_FLOAT_DATA || f.num === TENSOR_INT32_DATA)) continue;
    fields.push(f);
  }
  if (raw) fields.push({ num: TENSOR_RAW_DATA, wire: 2, data: raw });
  return encode(fields);
}

/** Each element (of `size` bytes) repeated `factor` times. */
function repeatElements(raw: Uint8Array, size: number, factor: number): Uint8Array {
  const count = raw.length / size;
  const out = new Uint8Array(raw.length * factor);
  for (let i = 0; i < count; i += 1) {
    const element = raw.subarray(i * size, (i + 1) * size);
    for (let k = 0; k < factor; k += 1) out.set(element, (i * factor + k) * size);
  }
  return out;
}

/**
 * 4-bit zero points packed two per byte, per row of `blocks`: each repeated `factor` times,
 * repacked.
 */
function repeatNibbles(raw: Uint8Array, rows: number, blocks: number, factor: number): Uint8Array {
  const inRow = Math.ceil(blocks / 2);
  const outBlocks = blocks * factor;
  const outRow = Math.ceil(outBlocks / 2);
  const out = new Uint8Array(rows * outRow);
  for (let r = 0; r < rows; r += 1) {
    for (let b = 0; b < outBlocks; b += 1) {
      const source = Math.floor(b / factor);
      const byte = raw[r * inRow + (source >> 1)]!;
      const nibble = source & 1 ? byte >> 4 : byte & 0x0f;
      out[r * outRow + (b >> 1)]! |= b & 1 ? nibble << 4 : nibble;
    }
  }
  return out;
}

interface Rewrite {
  /** New bytes of each initializer, by name. */
  tensors: Map<string, Uint8Array>;
  /** The original block size of each rewritten weight or scale tensor. */
  shared: Map<string, number>;
  count: number;
  /**
   * Tensors another kind of node reads too — a tied embedding's GatherBlockQuantized shares the
   * output head's weights and scales (Gemma 3): rewritten for MatMulNBits, they would break it
   * ("scales must have the same rank as data"). Their MatMulNBits get copies of their own
   * (`<name>__mnb32`), the originals left to the other nodes.
   */
  foreign: Set<string>;
  /** Those copies, by the name of the tensor they copy (added beside it). */
  copies: Map<string, Uint8Array>;
}

const COPY_SUFFIX = "__mnb32";

/** `tensor` under another name. */
function renamed(tensor: Tensor, name: string): Tensor {
  const bytes = new TextEncoder().encode(name);
  return { ...tensor, name, fields: tensor.fields.map((f) => (f.num === TENSOR_NAME ? { ...f, data: bytes } : f)) };
}

/** Every name a node that is not a MatMulNBits reads, subgraphs included. */
function foreignReads(graphBytes: Uint8Array, into: Set<string>): void {
  for (const f of parse(graphBytes)) {
    if (f.num !== GRAPH_NODE) continue;
    const node = parse(f.data!);
    const matmul = str(node.find((x) => x.num === NODE_OP_TYPE)) === "MatMulNBits";
    for (const x of node) {
      if (x.num === NODE_INPUT && !matmul) into.add(str(x));
      else if (x.num === NODE_ATTRIBUTE) {
        for (const a of parse(x.data!)) if ((a.num === ATTR_G || a.num === ATTR_GRAPHS) && a.data) foreignReads(a.data, into);
      }
    }
  }
}

/**
 * Find the MatMulNBits nodes of a graph (and its subgraphs) to rewrite, and rewrite their
 * initializers.
 */
function planGraph(graphBytes: Uint8Array, outer: Map<string, Tensor>, rewrite: Rewrite): Uint8Array {
  const fields = parse(graphBytes);
  const local = new Map(outer);
  const initializerAt = new Map<string, number>();
  fields.forEach((f, i) => {
    if (f.num !== GRAPH_INITIALIZER) return;
    const tensor = readTensor(f.data!);
    local.set(tensor.name, tensor);
    initializerAt.set(tensor.name, i);
  });
  let changed = false;
  for (const [i, f] of fields.entries()) {
    if (f.num !== GRAPH_NODE) continue;
    const node = parse(f.data!);
    let nodeChanged = false;

    for (const [j, attrField] of node.entries()) {
      if (attrField.num !== NODE_ATTRIBUTE) continue;
      const attr = parse(attrField.data!);
      let attrChanged = false;
      for (const [k, a] of attr.entries()) {
        if (a.num === ATTR_G || a.num === ATTR_GRAPHS) {
          const rewritten = planGraph(a.data!, local, rewrite);
          if (rewritten !== a.data) {
            attr[k] = { ...a, data: rewritten };
            attrChanged = true;
          }
        }
      }
      if (attrChanged) {
        node[j] = { ...attrField, data: encode(attr) };
        nodeChanged = true;
      }
    }
    if (str(node.find((x) => x.num === NODE_OP_TYPE)) === "MatMulNBits") {
      const done = rewriteNode(node, local, rewrite);
      nodeChanged ||= done;
    }
    if (nodeChanged) {
      fields[i] = { ...f, data: encode(node) };
      changed = true;
    }
  }

  for (const [name, index] of initializerAt) {
    const bytes = rewrite.tensors.get(name);
    if (bytes) {
      fields[index] = { ...fields[index]!, data: bytes };
      changed = true;
    }
    const copy = rewrite.copies.get(name);
    if (copy) {
      fields.push({ num: GRAPH_INITIALIZER, wire: 2, data: copy });
      rewrite.copies.delete(name);
      changed = true;
    }
  }
  return changed ? encode(fields) : graphBytes;
}

function rewriteNode(node: Field[], tensors: Map<string, Tensor>, rewrite: Rewrite): boolean {
  const inputs = node.filter((f) => f.num === NODE_INPUT).map(str);
  if (inputs.length > 4 && inputs[4]) return false;
  const attrs = node.map((f, i) => [f, i] as const).filter(([f]) => f.num === NODE_ATTRIBUTE);
  let block = 0;
  let bits = 4;
  let blockAt = -1;
  for (const [f, i] of attrs) {
    const attr = parse(f.data!);
    const name = str(attr.find((a) => a.num === ATTR_NAME));
    const value = Number(attr.find((a) => a.num === ATTR_I)?.int ?? 0n);
    if (name === "block_size") {
      block = value;
      blockAt = i;
    } else if (name === "bits") bits = value;
  }

  const put = (t: Tensor, dims: readonly number[], raw: Uint8Array | null): void => {
    if (!rewrite.foreign.has(t.name)) {
      rewrite.tensors.set(t.name, writeTensor(t, dims, raw));
      return;
    }
    const copy = `${t.name}${COPY_SUFFIX}`;
    if (!rewrite.copies.has(t.name)) rewrite.copies.set(t.name, writeTensor(renamed(t, copy), dims, raw));
    const bytes = new TextEncoder().encode(copy);
    let index = 0;
    for (const [i, f] of node.entries()) {
      if (f.num !== NODE_INPUT) continue;
      if (index > 0 && str(f) === t.name) node[i] = { ...f, data: bytes };
      index += 1;
    }
  };
  const weights = tensors.get(inputs[1] ?? "");
  const scales = tensors.get(inputs[2] ?? "");
  const zeros = inputs[3] ? tensors.get(inputs[3]) : undefined;
  if (bits === 4 && block === WEBGPU_NBITS_BLOCK && scales) {
    let flattened = false;
    for (const t of [scales, zeros]) {
      if (!t || t.dims.length < 2) continue;
      if (!rewrite.tensors.has(t.name)) put(t, [t.dims.reduce((a, b) => a * b, 1)], null);
      flattened = true;
    }
    if (flattened) rewrite.count += 1;
    return flattened;
  }
  if (bits !== 4 || block <= WEBGPU_NBITS_BLOCK || block % WEBGPU_NBITS_BLOCK !== 0 || blockAt < 0) return false;
  const factor = block / WEBGPU_NBITS_BLOCK;
  if (!weights || !scales || (inputs[3] && !zeros)) return false;
  const setBlock = () => {
    const attrField = node[blockAt]!;
    const attr = parse(attrField.data!).map((a) => (a.num === ATTR_I ? { ...a, int: BigInt(WEBGPU_NBITS_BLOCK) } : a));
    node[blockAt] = { ...attrField, data: encode(attr) };
    rewrite.count += 1;
    return true;
  };

  const done = rewrite.shared.get(weights.name);
  if (done !== undefined) {
    if (done !== block || rewrite.shared.get(scales.name) !== block) return false;

    for (const t of [weights, scales, zeros]) {
      if (!t || !rewrite.foreign.has(t.name)) continue;
      const bytes = new TextEncoder().encode(`${t.name}${COPY_SUFFIX}`);
      let index = 0;
      for (const [i, f] of node.entries()) {
        if (f.num !== NODE_INPUT) continue;
        if (index > 0 && str(f) === t.name) node[i] = { ...f, data: bytes };
        index += 1;
      }
    }
    return setBlock();
  }
  const scaleRaw = rawOf(scales);
  const weightRaw = rawOf(weights);
  if (!scaleRaw || !weightRaw || weights.dims.length !== 3) return false;
  const [n, blocks] = weights.dims as [number, number, number];
  const scaleSize = scales.dataType === FLOAT16 ? 2 : 4;
  if (scaleRaw.length !== n * blocks * scaleSize) return false;
  let zeroBytes: Uint8Array | null = null;
  if (zeros) {
    const zeroRaw = rawOf(zeros);
    if (!zeroRaw) return false;
    if (zeros.dataType === UINT8) zeroBytes = writeTensor(zeros, [n * Math.ceil((blocks * factor) / 2)], repeatNibbles(zeroRaw, n, blocks, factor));
    else zeroBytes = writeTensor(zeros, [n * blocks * factor], repeatElements(zeroRaw, zeros.dataType === FLOAT16 ? 2 : 4, factor));
  }
  put(weights, [n, blocks * factor, WEBGPU_NBITS_BLOCK / 2], null);
  put(scales, [n * blocks * factor], repeatElements(scaleRaw, scaleSize, factor));
  rewrite.shared.set(weights.name, block);
  rewrite.shared.set(scales.name, block);
  if (zeros && zeroBytes) {
    if (rewrite.foreign.has(zeros.name)) put(zeros, readTensor(zeroBytes).dims, rawOf(readTensor(zeroBytes)));
    else rewrite.tensors.set(zeros.name, zeroBytes);
  }
  return setBlock();
}

/**
 * The model with every 4-bit `MatMulNBits` of block 64, 128… rewritten to blocks of 32, and how
 * many nodes changed. The same bytes come back when nothing needed it (no such node, act-order
 * layouts, external data).
 */
export function repackMatMulNBits(model: Uint8Array): { bytes: Uint8Array; rewritten: number } {
  if (!operatorTypes(model).has("MatMulNBits")) return { bytes: model, rewritten: 0 };
  const fields = parse(model);
  const foreign = new Set<string>();
  for (const f of fields) if (f.num === MODEL_GRAPH) foreignReads(f.data!, foreign);
  const rewrite: Rewrite = { tensors: new Map(), shared: new Map(), count: 0, foreign, copies: new Map() };
  let changed = false;
  for (const [i, f] of fields.entries()) {
    if (f.num !== MODEL_GRAPH) continue;
    const graph = planGraph(f.data!, new Map(), rewrite);
    if (graph !== f.data) {
      fields[i] = { ...f, data: graph };
      changed = true;
    }
  }
  return changed ? { bytes: encode(fields), rewritten: rewrite.count } : { bytes: model, rewritten: 0 };
}
