/** `Gather` indices computed on the GPU, cast to int32 for WebGPU. */

import { deeper, encode, operatorTypes, parse, str, type Field } from "./protobuf.js";

const MODEL_GRAPH = 7;
const GRAPH_NODE = 1;
const GRAPH_INITIALIZER = 5;
const GRAPH_INPUT = 11;
const VALUE_NAME = 1;
const TENSOR_NAME = 8;
const NODE_INPUT = 1;
const NODE_OUTPUT = 2;
const NODE_NAME = 3;
const NODE_OP_TYPE = 4;
const NODE_ATTRIBUTE = 5;
const NODE_DOMAIN = 7;
const ATTR_NAME = 1;
const ATTR_I = 3;
const ATTR_G = 6;
const ATTR_GRAPHS = 11;
const ATTR_TYPE = 20;
const ATTR_TYPE_INT = 2;
const INT32 = 6;

/** The suffix of the int32 copies; enough to spot them in a dumped graph. */
export const INT32_SUFFIX = "__ynonnx_i32";

const bytesOf = (text: string) => new TextEncoder().encode(text);

function castNode(input: string, output: string): Field {
  const to = encode([
    { num: ATTR_NAME, wire: 2, data: bytesOf("to") },
    { num: ATTR_I, wire: 0, int: BigInt(INT32) },
    { num: ATTR_TYPE, wire: 0, int: BigInt(ATTR_TYPE_INT) },
  ]);
  return {
    num: GRAPH_NODE,
    wire: 2,
    data: encode([
      { num: NODE_INPUT, wire: 2, data: bytesOf(input) },
      { num: NODE_OUTPUT, wire: 2, data: bytesOf(output) },
      { num: NODE_NAME, wire: 2, data: bytesOf(`${output}_cast`) },
      { num: NODE_OP_TYPE, wire: 2, data: bytesOf("Cast") },
      { num: NODE_ATTRIBUTE, wire: 2, data: to },
    ]),
  };
}

function rewriteGraph(graphBytes: Uint8Array, outerFed: ReadonlySet<string>, counter: { casts: number }, depth = 0): Uint8Array {
  const fields = parse(graphBytes);

  const fed = new Set(outerFed);
  for (const f of fields) {
    if (f.num === GRAPH_INPUT) fed.add(str(parse(f.data!).find((x) => x.num === VALUE_NAME)));
    else if (f.num === GRAPH_INITIALIZER) fed.add(str(parse(f.data!).find((x) => x.num === TENSOR_NAME)));
  }
  const out: Field[] = [];
  const casted = new Map<string, string>();
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
        if ((a.num !== ATTR_G && a.num !== ATTR_GRAPHS) || !a.data) continue;
        const rewritten = rewriteGraph(a.data, fed, counter, deeper(depth));
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
    const isGather = str(node.find((x) => x.num === NODE_OP_TYPE)) === "Gather" && !str(node.find((x) => x.num === NODE_DOMAIN));
    if (isGather) {
      const inputs = node.map((x, i) => [x, i] as const).filter(([x]) => x.num === NODE_INPUT);
      const [indexField, indexAt] = inputs[1] ?? [];
      const index = str(indexField);
      if (indexField && index && !fed.has(index)) {
        let copy = casted.get(index);
        if (!copy) {
          copy = `${index}${INT32_SUFFIX}${counter.casts}`;
          casted.set(index, copy);
          out.push(castNode(index, copy));
          counter.casts += 1;
        }
        node[indexAt!] = { ...indexField, data: bytesOf(copy) };
        nodeChanged = true;
      }
    }
    if (nodeChanged) {
      out.push({ ...f, data: encode(node) });
      changed = true;
    } else {
      out.push(f);
    }
  }
  return changed ? encode(out) : graphBytes;
}

/**
 * The model with an int32 cast in front of every computed `Gather` index, and how many casts were
 * added. The same bytes come back when none was needed.
 */
export function castGatherIndices(model: Uint8Array): { bytes: Uint8Array; casts: number } {
  if (!operatorTypes(model).has("Gather")) return { bytes: model, casts: 0 };
  const fields = parse(model);
  const counter = { casts: 0 };
  let changed = false;
  for (const [i, f] of fields.entries()) {
    if (f.num !== MODEL_GRAPH) continue;
    const graph = rewriteGraph(f.data!, new Set(), counter);
    if (graph !== f.data) {
      fields[i] = { ...f, data: graph };
      changed = true;
    }
  }
  return changed ? { bytes: encode(fields), casts: counter.casts } : { bytes: model, casts: 0 };
}
