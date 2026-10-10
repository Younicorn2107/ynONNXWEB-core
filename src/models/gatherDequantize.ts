/** A quantized embedding table read row by row, not dequantized whole. */

import { encode, operatorTypes, parse, str, type Field } from "./protobuf.js";

const MODEL_OPSET = 8;
const MODEL_GRAPH = 7;
const OPSET_DOMAIN = 1;
const OPSET_VERSION = 2;
const GRAPH_NODE = 1;
const GRAPH_INITIALIZER = 5;
const NODE_INPUT = 1;
const NODE_OUTPUT = 2;
const NODE_NAME = 3;
const NODE_OP_TYPE = 4;
const NODE_ATTRIBUTE = 5;
const NODE_DOMAIN = 7;
const ATTR_NAME = 1;
const ATTR_I = 3;
const ATTR_G = 6;
const ATTR_INTS = 8;
const ATTR_GRAPHS = 11;
const ATTR_TYPE = 20;
const ATTR_TYPE_INT = 2;
const ATTR_TYPE_INTS = 7;
const TENSOR_DIMS = 1;
const TENSOR_DATA_TYPE = 2;
const TENSOR_NAME = 8;
const TENSOR_RAW = 9;
const INT64 = 7;
const UINT8 = 2;
const INT8 = 3;
const SUFFIX = "__rows";

const bytesOf = (text: string) => new TextEncoder().encode(text);

function attribute(name: string, value: number | number[]): Field {
  const fields: Field[] = [{ num: ATTR_NAME, wire: 2, data: bytesOf(name) }];
  if (Array.isArray(value)) {
    for (const v of value) fields.push({ num: ATTR_INTS, wire: 0, int: BigInt(v) });
    fields.push({ num: ATTR_TYPE, wire: 0, int: BigInt(ATTR_TYPE_INTS) });
  } else {
    fields.push({ num: ATTR_I, wire: 0, int: BigInt(value) }, { num: ATTR_TYPE, wire: 0, int: BigInt(ATTR_TYPE_INT) });
  }
  return { num: NODE_ATTRIBUTE, wire: 2, data: encode(fields) };
}

function node(op: string, inputs: string[], output: string, attributes: Field[] = []): Field {
  return {
    num: GRAPH_NODE,
    wire: 2,
    data: encode([
      ...inputs.map((i) => ({ num: NODE_INPUT, wire: 2, data: bytesOf(i) })),
      { num: NODE_OUTPUT, wire: 2, data: bytesOf(output) },
      { num: NODE_NAME, wire: 2, data: bytesOf(`${output}_node`) },
      { num: NODE_OP_TYPE, wire: 2, data: bytesOf(op) },
      ...attributes,
    ]),
  };
}

function opsetOf(fields: readonly Field[]): number {
  for (const f of fields) {
    if (f.num !== MODEL_OPSET) continue;
    const entry = parse(f.data!);
    if (str(entry.find((x) => x.num === OPSET_DOMAIN)) !== "") continue;
    const version = entry.find((x) => x.num === OPSET_VERSION);
    if (version?.int !== undefined) return Number(version.int);
  }
  return 0;
}

function intsAttribute(n: readonly Field[], name: string): number[] | undefined {
  for (const a of n) {
    if (a.num !== NODE_ATTRIBUTE) continue;
    const attr = parse(a.data!);
    if (str(attr.find((x) => x.num === ATTR_NAME)) !== name) continue;
    return attr.filter((x) => x.num === ATTR_INTS).flatMap((x) => (x.int !== undefined ? [Number(BigInt.asIntN(64, x.int))] : x.data ? packedInts(x.data) : []));
  }
  return undefined;
}

/** Varints packed in one length-delimited field. */
function packedInts(data: Uint8Array): number[] {
  const out: number[] = [];
  let at = 0;
  while (at < data.length) {
    let value = 0n;
    let shift = 0n;
    for (;;) {
      const byte = data[at++]!;
      value |= BigInt(byte & 0x7f) << shift;
      if (byte < 0x80) break;
      shift += 7n;
    }
    out.push(Number(BigInt.asIntN(64, value)));
  }
  return out;
}

/** A TensorProto's dims, packed or not. */
function dimsOf(t: readonly Field[]): number[] {
  return t.filter((x) => x.num === TENSOR_DIMS).flatMap((x) => (x.int !== undefined ? [Number(x.int)] : x.data ? packedInts(x.data) : []));
}

function intAttribute(n: readonly Field[], name: string): number | undefined {
  for (const a of n) {
    if (a.num !== NODE_ATTRIBUTE) continue;
    const attr = parse(a.data!);
    if (str(attr.find((x) => x.num === ATTR_NAME)) !== name) continue;
    const i = attr.find((x) => x.num === ATTR_I);
    return i?.int !== undefined ? Number(BigInt.asIntN(64, i.int)) : undefined;
  }
  return undefined;
}

interface Table {
  /** The quantized table, its scales, its zero points (or ""). */
  weights: string;
  scales: string;
  zeros: string;
  /** The float type the dequantization produces (the scales' type). */
  floatType: number;
}

/** Calls `visit` on every node list: the graph's, then every subgraph's, depth first. */
function forEachNodeList(graph: Field[], visit: (nodes: Field[][]) => void): void {
  const nodes = graph.filter((f) => f.num === GRAPH_NODE).map((f) => parse(f.data!));
  visit(nodes);
  for (const n of nodes) {
    for (const a of n) {
      if (a.num !== NODE_ATTRIBUTE) continue;
      for (const x of parse(a.data!)) if ((x.num === ATTR_G || x.num === ATTR_GRAPHS) && x.data) forEachNodeList(parse(x.data), visit);
    }
  }
}

/**
 * The graph with each matching `Gather` rebuilt: `replace` maps a Gather's fields to the nodes
 * standing for it, or null to keep it. Subgraphs too.
 */
function rewriteGraph(graph: Field[], replace: (n: Field[]) => Field[] | null, drop: ReadonlySet<string>): Field[] {
  const out: Field[] = [];
  for (const f of graph) {
    if (f.num !== GRAPH_NODE) {
      out.push(f);
      continue;
    }
    const n = parse(f.data!);
    if (drop.has(str(n.find((x) => x.num === NODE_OUTPUT)))) continue;
    const replaced = replace(n);
    if (replaced) {
      out.push(...replaced);
      continue;
    }

    let changed = false;
    const fields = n.map((a) => {
      if (a.num !== NODE_ATTRIBUTE) return a;
      const attr = parse(a.data!);
      if (!attr.some((x) => (x.num === ATTR_G || x.num === ATTR_GRAPHS) && x.data)) return a;
      changed = true;
      const rebuilt = attr.map((x) => ((x.num === ATTR_G || x.num === ATTR_GRAPHS) && x.data ? { ...x, data: encode(rewriteGraph(parse(x.data), replace, drop)) } : x));
      return { ...a, data: encode(rebuilt) };
    });
    out.push(changed ? { ...f, data: encode(fields) } : f);
  }
  return out;
}

export function gatherBeforeDequantize(model: Uint8Array): { bytes: Uint8Array; rewritten: number } {
  const ops = operatorTypes(model);
  if (!ops.has("DequantizeLinear") || (!ops.has("Gather") && !ops.has("Transpose"))) return { bytes: model, rewritten: 0 };
  const fields = parse(model);
  const graphAt = fields.findIndex((f) => f.num === MODEL_GRAPH);
  if (graphAt < 0) return { bytes: model, rewritten: 0 };
  const graph = parse(fields[graphAt]!.data!);
  const opset = opsetOf(fields);

  const initializers = new Map<string, { dataType: number; rank: number; dims: number[]; raw: Uint8Array | null }>();
  for (const f of graph) {
    if (f.num !== GRAPH_INITIALIZER) continue;
    const t = parse(f.data!);
    const dims = dimsOf(t);
    initializers.set(str(t.find((x) => x.num === TENSOR_NAME)), {
      dataType: Number(t.find((x) => x.num === TENSOR_DATA_TYPE)?.int ?? 0n),
      rank: dims.length,
      dims,
      raw: t.find((x) => x.num === TENSOR_RAW)?.data ?? null,
    });
  }

  const tables = new Map<string, Table>();
  for (const f of graph) {
    if (f.num !== GRAPH_NODE) continue;
    const n = parse(f.data!);
    if (str(n.find((x) => x.num === NODE_OP_TYPE)) !== "DequantizeLinear" || str(n.find((x) => x.num === NODE_DOMAIN))) continue;
    const [weights = "", scales = "", zeros = ""] = n.filter((x) => x.num === NODE_INPUT).map(str);
    const w = initializers.get(weights);
    const s = initializers.get(scales);
    const z = zeros ? initializers.get(zeros) : undefined;
    if (!w || !s || w.rank !== 2 || s.rank !== 1 || (zeros && (!z || z.rank !== 1))) continue;
    if ((intAttribute(n, "axis") ?? 1) !== 0 || intAttribute(n, "block_size")) continue;
    tables.set(str(n.find((x) => x.num === NODE_OUTPUT)), { weights, scales, zeros, floatType: s.dataType });
  }
  if (tables.size === 0) return { bytes: model, rewritten: 0 };

  forEachNodeList(graph, (nodes) => {
    for (const n of nodes) {
      const op = str(n.find((x) => x.num === NODE_OP_TYPE));
      const inputs = n.filter((x) => x.num === NODE_INPUT).map(str);
      inputs.forEach((input, at) => {
        const table = tables.get(input);
        if (!table) return;
        const gather = op === "Gather" && at === 0 && (intAttribute(n, "axis") ?? 0) === 0;
        const w = initializers.get(table.weights)!;
        const perm = intsAttribute(n, "perm");
        const transpose = op === "Transpose" && (!perm || (perm.length === 2 && perm[0] === 1 && perm[1] === 0)) && w.raw !== null && (w.dataType === INT8 || w.dataType === UINT8);
        if (!gather && !transpose) tables.delete(input);
      });
    }
  });
  if (tables.size === 0) return { bytes: model, rewritten: 0 };

  const axes = `${SUFFIX}_axes`;
  let rewritten = 0;
  /** Each table's transposed weights, written once. */
  const transposed = new Map<string, Field>();
  const replace = (n: Field[]): Field[] | null => {
    const op = str(n.find((x) => x.num === NODE_OP_TYPE));
    if (op === "Transpose") {
      const table = tables.get(str(n.find((x) => x.num === NODE_INPUT)));
      if (!table) return null;
      const name = `${table.weights}${SUFFIX}_t`;
      if (!transposed.has(table.weights)) {
        const w = initializers.get(table.weights)!;
        const [rows, cols] = w.dims as [number, number];
        const out = new Uint8Array(rows * cols);
        for (let r = 0; r < rows; r += 1) for (let c = 0; c < cols; c += 1) out[c * rows + r] = w.raw![r * cols + c]!;
        transposed.set(table.weights, {
          num: GRAPH_INITIALIZER,
          wire: 2,
          data: encode([
            { num: TENSOR_DIMS, wire: 0, int: BigInt(cols) },
            { num: TENSOR_DIMS, wire: 0, int: BigInt(rows) },
            { num: TENSOR_DATA_TYPE, wire: 0, int: BigInt(w.dataType) },
            { num: TENSOR_NAME, wire: 2, data: bytesOf(name) },
            { num: TENSOR_RAW, wire: 2, data: out },
          ]),
        });
      }
      rewritten += 1;
      const y = str(n.find((x) => x.num === NODE_OUTPUT));
      return [node("DequantizeLinear", [name, table.scales, ...(table.zeros ? [table.zeros] : [])], y, [attribute("axis", 1)])];
    }
    if (op !== "Gather") return null;
    const [data = "", indices = ""] = n.filter((x) => x.num === NODE_INPUT).map(str);
    const table = tables.get(data);
    if (!table) return null;
    const y = str(n.find((x) => x.num === NODE_OUTPUT));
    const p = `${y}${SUFFIX}`;
    const unsqueeze = (input: string, output: string) => (opset >= 13 ? node("Unsqueeze", [input, axes], output) : node("Unsqueeze", [input], output, [attribute("axes", [-1])]));
    const out: Field[] = [
      node("Gather", [table.weights, indices], `${p}_q`, [attribute("axis", 0)]),
      node("Cast", [`${p}_q`], `${p}_f`, [attribute("to", table.floatType)]),
      node("Gather", [table.scales, indices], `${p}_s`, [attribute("axis", 0)]),
      unsqueeze(`${p}_s`, `${p}_s1`),
    ];
    if (table.zeros) {
      out.push(
        node("Gather", [table.zeros, indices], `${p}_z`, [attribute("axis", 0)]),
        node("Cast", [`${p}_z`], `${p}_zf`, [attribute("to", table.floatType)]),
        unsqueeze(`${p}_zf`, `${p}_z1`),
        node("Sub", [`${p}_f`, `${p}_z1`], `${p}_c`),
        node("Mul", [`${p}_c`, `${p}_s1`], y),
      );
    } else {
      out.push(node("Mul", [`${p}_f`, `${p}_s1`], y));
    }
    rewritten += 1;
    return out;
  };
  const out = rewriteGraph(graph, replace, new Set(tables.keys()));
  if (rewritten === 0) return { bytes: model, rewritten: 0 };
  out.push(...transposed.values());
  if (opset >= 13) {
    out.push({
      num: GRAPH_INITIALIZER,
      wire: 2,
      data: encode([
        { num: TENSOR_DIMS, wire: 0, int: 1n },
        { num: TENSOR_DATA_TYPE, wire: 0, int: BigInt(INT64) },
        { num: TENSOR_NAME, wire: 2, data: bytesOf(axes) },
        { num: TENSOR_RAW, wire: 2, data: new Uint8Array(new BigInt64Array([-1n]).buffer) },
      ]),
    });
  }
  const copy = [...fields];
  copy[graphAt] = { ...fields[graphAt]!, data: encode(out) };
  return { bytes: encode(copy), rewritten };
}
