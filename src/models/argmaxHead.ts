/** An output of scores replaced by its best class and that class's score. */

import { ManifestError } from "../errors.js";
import { DEFAULT_LIMITS } from "./limits.js";
import { encode, operatorTypes, parse, str, type Field } from "./protobuf.js";

const MODEL_OPSET = 8;
const MODEL_GRAPH = 7;
const OPSET_DOMAIN = 1;
const OPSET_VERSION = 2;
const GRAPH_NODE = 1;
const GRAPH_INITIALIZER = 5;
const GRAPH_OUTPUT = 12;
const VALUE_NAME = 1;
const VALUE_TYPE = 2;
const NODE_INPUT = 1;
const NODE_OUTPUT = 2;
const NODE_NAME = 3;
const NODE_OP_TYPE = 4;
const NODE_ATTRIBUTE = 5;
const ATTR_NAME = 1;
const ATTR_I = 3;
const ATTR_INTS = 8;
const ATTR_TYPE = 20;
const ATTR_TYPE_INT = 2;
const ATTR_TYPE_INTS = 7;
const TENSOR_DIMS = 1;
const TENSOR_DATA_TYPE = 2;
const TENSOR_NAME = 8;
const TENSOR_RAW = 9;
const FLOAT = 1;
const INT64 = 7;

export const ARGMAX_SUFFIX = "__argmax";
export const MAX_SUFFIX = "__max";

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

function node(op: string, inputs: string[], output: string, attributes: Field[]): Field {
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

/** A graph output of this element type, shape left to inference. */
function valueInfo(name: string, elemType: number): Field {
  const tensor = encode([{ num: 1, wire: 0, int: BigInt(elemType) }]);
  const type = encode([{ num: 1, wire: 2, data: tensor }]);
  return { num: GRAPH_OUTPUT, wire: 2, data: encode([{ num: VALUE_NAME, wire: 2, data: bytesOf(name) }, { num: VALUE_TYPE, wire: 2, data: type }]) };
}

/** The default-domain opset the model declares. */
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

/** The element type a graph output declares (0 when it says none). */
function elemTypeOf(valueInfo: Uint8Array): number {
  const type = parse(valueInfo).find((x) => x.num === VALUE_TYPE);
  const tensor = type && parse(type.data!).find((x) => x.num === 1);
  const elem = tensor && parse(tensor.data!).find((x) => x.num === 1);
  return elem?.int !== undefined ? Number(elem.int) : 0;
}

/** The last dimension a graph output declares, when it is a number. */
function lastDim(valueInfo: Uint8Array): number | null {
  const type = parse(valueInfo).find((x) => x.num === VALUE_TYPE);
  const tensor = type && parse(type.data!).find((x) => x.num === 1);
  const shape = tensor && parse(tensor.data!).find((x) => x.num === 2);
  const dims = shape ? parse(shape.data!).filter((x) => x.num === 1) : [];
  const last = dims[dims.length - 1];
  const value = last && parse(last.data!).find((x) => x.num === 1);
  return value?.int !== undefined && value.int > 0n ? Number(value.int) : null;
}

/**
 * The class count a head is built for, refused when it cannot be one: the weights written below are
 * one float per class, and a graph declaring 2^62 classes would otherwise be taken at its word.
 */
function classCount(count: number, output: string): number {
  if (!Number.isSafeInteger(count) || count <= 0 || count * 4 > DEFAULT_LIMITS.maxTensorBytes) {
    throw new ManifestError(`${output}: ${count} classes is not a class count a head can be built for`);
  }
  return count;
}

function initializer(name: string, dataType: number, dims: number[], raw: Uint8Array): Field {
  return {
    num: GRAPH_INITIALIZER,
    wire: 2,
    data: encode([
      ...dims.map((d) => ({ num: TENSOR_DIMS, wire: 0, int: BigInt(d) })),
      { num: TENSOR_DATA_TYPE, wire: 0, int: BigInt(dataType) },
      { num: TENSOR_NAME, wire: 2, data: bytesOf(name) },
      { num: TENSOR_RAW, wire: 2, data: raw },
    ]),
  };
}

/**
 * The model with `output` (its last axis the classes) replaced by `output__argmax` (the first best
 * class, as float32) and `output__max` (its score), both `[…, 1]`. `rewritten: false`, the bytes as
 * they were, when there is no such output or its class count is not fixed (pass `classes` then).
 */
export function argmaxHead(model: Uint8Array, output: string, classes?: number): { bytes: Uint8Array; rewritten: boolean } {
  const fields = parse(model);
  const opset = opsetOf(fields);
  let rewritten = false;
  for (const [i, f] of fields.entries()) {
    if (f.num !== MODEL_GRAPH) continue;
    const graph = parse(f.data!);
    const at = graph.findIndex((g) => g.num === GRAPH_OUTPUT && str(parse(g.data!).find((x) => x.num === VALUE_NAME)) === output);
    if (at < 0) continue;
    const declared = classes ?? lastDim(graph[at]!.data!);
    if (!declared) continue;
    const count = classCount(declared, output);
    const name = (suffix: string) => `${output}__argmax_${suffix}`;
    const max = `${output}${MAX_SUFFIX}`;
    const reduce = (input: string, out: string): Field[] => {
      if (opset < 18) return [node("ReduceMax", [input], out, [attribute("axes", [-1]), attribute("keepdims", 1)])];

      return [node("ReduceMax", [input, name("axes")], out, [attribute("keepdims", 1)])];
    };
    const weights = new Float32Array(count).map((_, k) => count - k);
    if (opset >= 18) graph.push(initializer(name("axes"), INT64, [1], new Uint8Array(new BigInt64Array([-1n]).buffer)));
    graph.push(
      initializer(name("weights"), FLOAT, [count], new Uint8Array(weights.buffer)),
      initializer(name("count"), FLOAT, [], new Uint8Array(new Float32Array([count]).buffer)),
    );

    const scores = elemTypeOf(graph[at]!.data!) === FLOAT ? output : name("float");
    const added: Field[] = [
      ...(scores === output ? [] : [node("Cast", [output], scores, [attribute("to", FLOAT)])]),
      ...reduce(scores, max),
      node("Equal", [scores, max], name("is_best"), []),
      node("Cast", [name("is_best")], name("best_mask"), [attribute("to", FLOAT)]),
      node("Mul", [name("best_mask"), name("weights")], name("weighted"), []),
      ...reduce(name("weighted"), name("first")),
      node("Sub", [name("count"), name("first")], `${output}${ARGMAX_SUFFIX}`, []),
    ];

    graph.splice(at, 1, valueInfo(`${output}${ARGMAX_SUFFIX}`, FLOAT), valueInfo(max, FLOAT));
    fields[i] = { ...f, data: encode([...added, ...graph]) };
    rewritten = true;
  }
  return rewritten ? { bytes: encode(fields), rewritten } : { bytes: model, rewritten: false };
}

export const TOP_IDS_SUFFIX = "__top_ids";
export const TOP_VALUES_SUFFIX = "__top_values";
export const LOGSUMEXP_SUFFIX = "__logsumexp";

/**
 * Candidates a decoder's head keeps by default: enough for a ban or a penalty on the best two or
 * three.
 */
export const GREEDY_HEAD_CANDIDATES = 4;

/**
 * A decoder's logits with, beside them, what a greedy step reads: the `k` best tokens
 * (`<output>__top_ids`, as float32, and `<output>__top_values`, both `[…, k]`, best first, ties to
 * the lower id) and the log of the softmax's denominator (`<output>__logsumexp`, `[…, 1]`).
 */
export function greedyHead(
  model: Uint8Array,
  output: string,
  options: { candidates?: number; classes?: number } = {},
): { bytes: Uint8Array; rewritten: boolean } {
  const fields = parse(model);
  const opset = opsetOf(fields);
  let rewritten = false;
  for (const [i, f] of fields.entries()) {
    if (f.num !== MODEL_GRAPH) continue;
    const graph = parse(f.data!);
    const at = graph.findIndex((g) => g.num === GRAPH_OUTPUT && str(parse(g.data!).find((x) => x.num === VALUE_NAME)) === output);
    if (at < 0) continue;
    const declared = options.classes ?? lastDim(graph[at]!.data!);
    if (!declared) continue;
    const count = classCount(declared, output);
    const k = Math.max(1, Math.min(Math.trunc(options.candidates ?? GREEDY_HEAD_CANDIDATES), count));
    const p = (suffix: string) => `${output}__greedy_${suffix}`;
    const axesInput = (input: string) => [input, p("axes")];

    const reduceMax = (input: string, out: string): Field =>
      opset >= 18 ? node("ReduceMax", axesInput(input), out, [attribute("keepdims", 1)]) : node("ReduceMax", [input], out, [attribute("axes", [-1]), attribute("keepdims", 1)]);
    const reduceSum = (input: string, out: string): Field =>
      opset >= 13 ? node("ReduceSum", axesInput(input), out, [attribute("keepdims", 1)]) : node("ReduceSum", [input], out, [attribute("axes", [-1]), attribute("keepdims", 1)]);
    const weights = new Float32Array(count).map((_, c) => count - c);
    const constants: Field[] = [
      initializer(p("weights"), FLOAT, [count], new Uint8Array(weights.buffer)),
      initializer(p("count"), FLOAT, [], new Uint8Array(new Float32Array([count]).buffer)),
    ];
    if (k > 1) constants.push(initializer(p("masked"), FLOAT, [], new Uint8Array(new Float32Array([Number.NEGATIVE_INFINITY]).buffer)));
    if (opset >= 13) constants.push(initializer(p("axes"), INT64, [1], new Uint8Array(new BigInt64Array([-1n]).buffer)));
    const first = elemTypeOf(graph[at]!.data!) === FLOAT ? output : p("x0");
    const added: Field[] = first === output ? [] : [node("Cast", [output], first, [attribute("to", FLOAT)])];
    const ids: string[] = [];
    const values: string[] = [];
    let scores = first;
    for (let rank = 0; rank < k; rank += 1) {
      const r = (suffix: string) => p(`${suffix}${rank}`);
      added.push(
        reduceMax(scores, r("max")),
        node("Equal", [scores, r("max")], r("is_best"), []),
        node("Cast", [r("is_best")], r("mask"), [attribute("to", FLOAT)]),
        node("Mul", [r("mask"), p("weights")], r("weighted"), []),
        reduceMax(r("weighted"), r("first")),
        node("Sub", [p("count"), r("first")], r("id"), []),
      );
      ids.push(r("id"));
      values.push(r("max"));
      if (rank === k - 1) break;

      added.push(node("Equal", [p("weights"), r("first")], r("taken"), []), node("Where", [r("taken"), p("masked"), scores], r("rest"), []));
      scores = r("rest");
    }
    const join = (parts: string[], out: string): Field => (parts.length === 1 ? node("Identity", parts, out, []) : node("Concat", parts, out, [attribute("axis", -1)]));
    added.push(
      join(ids, `${output}${TOP_IDS_SUFFIX}`),
      join(values, `${output}${TOP_VALUES_SUFFIX}`),
      node("Sub", [first, values[0]!], p("shifted"), []),
      node("Exp", [p("shifted")], p("exp"), []),
      reduceSum(p("exp"), p("sum")),
      node("Log", [p("sum")], p("log_sum"), []),
      node("Add", [values[0]!, p("log_sum")], `${output}${LOGSUMEXP_SUFFIX}`, []),
    );
    graph.push(
      ...constants,
      valueInfo(`${output}${TOP_IDS_SUFFIX}`, FLOAT),
      valueInfo(`${output}${TOP_VALUES_SUFFIX}`, FLOAT),
      valueInfo(`${output}${LOGSUMEXP_SUFFIX}`, FLOAT),
    );
    fields[i] = { ...f, data: encode([...added, ...graph]) };
    rewritten = true;
  }
  return rewritten ? { bytes: encode(fields), rewritten } : { bytes: model, rewritten: false };
}

const NODE_DOMAIN = 7;
const SCALAR_SUFFIX = "__ynonnx_argmax";

/** A node's integer attributes. */
function intAttributes(node: readonly Field[]): Map<string, number> {
  const out = new Map<string, number>();
  for (const f of node) {
    if (f.num !== NODE_ATTRIBUTE) continue;
    const attr = parse(f.data!);
    const value = attr.find((x) => x.num === ATTR_I);
    if (value?.int !== undefined) out.set(str(attr.find((x) => x.num === ATTR_NAME)), Number(BigInt.asIntN(64, value.int)));
  }
  return out;
}

/** Every `ArgMax` over the last axis rewritten without `ArgMax`, for WebGPU. */
export function replaceArgMax(model: Uint8Array): { bytes: Uint8Array; replaced: number } {
  if (!operatorTypes(model).has("ArgMax")) return { bytes: model, replaced: 0 };
  const fields = parse(model);
  const opset = opsetOf(fields);
  let replaced = 0;
  for (const [i, f] of fields.entries()) {
    if (f.num !== MODEL_GRAPH) continue;
    const graph = parse(f.data!);
    const out: Field[] = [];
    let count = 0;
    for (const g of graph) {
      if (g.num !== GRAPH_NODE) {
        out.push(g);
        continue;
      }
      const n = parse(g.data!);
      const op = str(n.find((x) => x.num === NODE_OP_TYPE));
      const attrs = intAttributes(n);
      if (op !== "ArgMax" || str(n.find((x) => x.num === NODE_DOMAIN)) || (attrs.get("axis") ?? 0) !== -1 || (attrs.get("select_last_index") ?? 0) !== 0) {
        out.push(g);
        continue;
      }
      const x = str(n.find((y) => y.num === NODE_INPUT));
      const y = str(n.find((z) => z.num === NODE_OUTPUT));
      const keep = attrs.get("keepdims") ?? 1;
      const p = `${y}${SCALAR_SUFFIX}${count}_`;
      const reduce = (input: string, output: string, keepdims: number): Field =>
        opset >= 18
          ? node("ReduceMax", [input, `${SCALAR_SUFFIX}_axes`], output, [attribute("keepdims", keepdims)])
          : node("ReduceMax", [input], output, [attribute("axes", [-1]), attribute("keepdims", keepdims)]);
      out.push(
        reduce(x, `${p}max`, 1),
        node("Equal", [x, `${p}max`], `${p}is_best`, []),
        node("Cast", [`${p}is_best`], `${p}mask`, [attribute("to", FLOAT)]),
        node("Shape", [x], `${p}shape`, []),
        node("Gather", [`${p}shape`, `${SCALAR_SUFFIX}_last`], `${p}count`, [attribute("axis", 0)]),
        node("Cast", [`${p}count`], `${p}count_f`, [attribute("to", FLOAT)]),
        node("Range", [`${p}count_f`, `${SCALAR_SUFFIX}_zero`, `${SCALAR_SUFFIX}_minus_one`], `${p}weights`, []),
        node("Mul", [`${p}mask`, `${p}weights`], `${p}weighted`, []),
        reduce(`${p}weighted`, `${p}first`, keep),
        node("Sub", [`${p}count_f`, `${p}first`], `${p}index`, []),
        node("Cast", [`${p}index`], y, [attribute("to", INT64)]),
      );
      count += 1;
    }
    if (count === 0) continue;

    out.push(
      initializer(`${SCALAR_SUFFIX}_last`, INT64, [], new Uint8Array(new BigInt64Array([-1n]).buffer)),
      initializer(`${SCALAR_SUFFIX}_zero`, FLOAT, [], new Uint8Array(new Float32Array([0]).buffer)),
      initializer(`${SCALAR_SUFFIX}_minus_one`, FLOAT, [], new Uint8Array(new Float32Array([-1]).buffer)),
    );
    if (opset >= 18) out.push(initializer(`${SCALAR_SUFFIX}_axes`, INT64, [1], new Uint8Array(new BigInt64Array([-1n]).buffer)));
    fields[i] = { ...f, data: encode(out) };
    replaced += count;
  }
  return replaced > 0 ? { bytes: encode(fields), replaced } : { bytes: model, replaced: 0 };
}

/**
 * Every top-level `IsNaN(x)` rewritten as `Not(Equal(x, x))` — NaN is the one value not equal to
 * itself.
 */
export function replaceIsNaN(model: Uint8Array): { bytes: Uint8Array; replaced: number } {
  if (!operatorTypes(model).has("IsNaN")) return { bytes: model, replaced: 0 };
  const fields = parse(model);
  let replaced = 0;
  for (const [i, f] of fields.entries()) {
    if (f.num !== MODEL_GRAPH) continue;
    const out: Field[] = [];
    let count = 0;
    for (const g of parse(f.data!)) {
      const n = g.num === GRAPH_NODE ? parse(g.data!) : null;
      if (!n || str(n.find((x) => x.num === NODE_OP_TYPE)) !== "IsNaN" || str(n.find((x) => x.num === NODE_DOMAIN))) {
        out.push(g);
        continue;
      }
      const x = str(n.find((y) => y.num === NODE_INPUT));
      const y = str(n.find((z) => z.num === NODE_OUTPUT));
      out.push(node("Equal", [x, x], `${y}__self_equal`, []), node("Not", [`${y}__self_equal`], y, []));
      count += 1;
    }
    if (count === 0) continue;
    fields[i] = { ...f, data: encode(out) };
    replaced += count;
  }
  return replaced > 0 ? { bytes: encode(fields), replaced } : { bytes: model, replaced: 0 };
}
