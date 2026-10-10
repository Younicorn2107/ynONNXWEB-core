/** A merged decoder taken apart into its two branches, for WebGPU. */

import { encode, operatorTypes, parse, str, type Field } from "./protobuf.js";

const MODEL_GRAPH = 7;
const GRAPH_NODE = 1;
const GRAPH_NAME = 2;
const GRAPH_INITIALIZER = 5;
const GRAPH_INPUT = 11;
const GRAPH_OUTPUT = 12;
const GRAPH_SPARSE_INITIALIZER = 15;
const VALUE_NAME = 1;
const NODE_INPUT = 1;
const NODE_OUTPUT = 2;
const NODE_NAME = 3;
const NODE_OP_TYPE = 4;
const NODE_ATTRIBUTE = 5;
const ATTR_NAME = 1;
const ATTR_G = 6;
const ATTR_GRAPHS = 11;
const TENSOR_NAME = 8;

const bytesOf = (text: string) => new TextEncoder().encode(text);

const nameOf = (value: Field) => str(parse(value.data!).find((x) => x.num === VALUE_NAME));

/** Every name a node list reads, nested subgraphs included. */
function readNames(nodes: readonly Field[], into: Set<string>): void {
  for (const node of nodes) {
    for (const f of parse(node.data!)) {
      if (f.num === NODE_INPUT) into.add(str(f));
      else if (f.num === NODE_ATTRIBUTE) {
        for (const a of parse(f.data!)) {
          if ((a.num === ATTR_G || a.num === ATTR_GRAPHS) && a.data) readNames(parse(a.data).filter((g) => g.num === GRAPH_NODE), into);
        }
      }
    }
  }
}

function identity(input: string, output: string): Field {
  return {
    num: GRAPH_NODE,
    wire: 2,
    data: encode([
      { num: NODE_INPUT, wire: 2, data: bytesOf(input) },
      { num: NODE_OUTPUT, wire: 2, data: bytesOf(output) },
      { num: NODE_NAME, wire: 2, data: bytesOf(`${output}__branch_out`) },
      { num: NODE_OP_TYPE, wire: 2, data: bytesOf("Identity") },
    ]),
  };
}

/**
 * The two branches of a merged decoder as graphs of their own (`withPast`: `use_cache_branch` true;
 * `firstPass`: false), or null when `model` is not one (no top-level `If` on `use_cache_branch`).
 */
export function splitMergedDecoder(model: Uint8Array): { firstPass: Uint8Array; withPast: Uint8Array } | null {
  if (!operatorTypes(model).has("If")) return null;
  const fields = parse(model);
  const graphAt = fields.findIndex((f) => f.num === MODEL_GRAPH);
  if (graphAt < 0) return null;
  const graph = parse(fields[graphAt]!.data!);
  const nodes = graph.filter((f) => f.num === GRAPH_NODE);
  const ifNode = nodes.find((n) => {
    const node = parse(n.data!);
    return str(node.find((x) => x.num === NODE_OP_TYPE)) === "If" && str(node.find((x) => x.num === NODE_INPUT)) === "use_cache_branch";
  });
  if (!ifNode) return null;
  const node = parse(ifNode.data!);
  const ifOutputs = node.filter((x) => x.num === NODE_OUTPUT).map((x) => str(x));
  const branches = new Map<string, Field[]>();
  for (const a of node.filter((x) => x.num === NODE_ATTRIBUTE)) {
    const attr = parse(a.data!);
    const g = attr.find((x) => x.num === ATTR_G);
    if (g?.data) branches.set(str(attr.find((x) => x.num === ATTR_NAME)), parse(g.data));
  }
  const thenBranch = branches.get("then_branch");
  const elseBranch = branches.get("else_branch");
  if (!thenBranch || !elseBranch) return null;

  const otherNodes = nodes.filter((n) => n !== ifNode);
  const initializers = graph.filter((f) => f.num === GRAPH_INITIALIZER || f.num === GRAPH_SPARSE_INITIALIZER);
  const inputs = graph.filter((f) => f.num === GRAPH_INPUT);
  const outputs = graph.filter((f) => f.num === GRAPH_OUTPUT);
  const name = graph.find((f) => f.num === GRAPH_NAME);

  const build = (branch: Field[]): Uint8Array => {
    const branchNodes = branch.filter((f) => f.num === GRAPH_NODE);
    const branchInitializers = branch.filter((f) => f.num === GRAPH_INITIALIZER || f.num === GRAPH_SPARSE_INITIALIZER);
    const branchOutputs = branch.filter((f) => f.num === GRAPH_OUTPUT).map(nameOf);

    const links = branchOutputs.map((out, i) => (out === ifOutputs[i] ? null : identity(out, ifOutputs[i]!))).filter((f): f is Field => f !== null);
    const allNodes = [...otherNodes, ...branchNodes, ...links];
    const read = new Set<string>();
    readNames(allNodes, read);

    const keptInputs = inputs.filter((i) => nameOf(i) !== "use_cache_branch" && read.has(nameOf(i)));
    const keptInitializers = [...initializers, ...branchInitializers].filter((t) => {
      const tensorName = str(parse(t.data!).find((x) => x.num === TENSOR_NAME));
      return !tensorName || read.has(tensorName);
    });
    const out: Field[] = [...(name ? [name] : []), ...allNodes, ...keptInitializers, ...keptInputs, ...outputs];
    const copy = [...fields];
    copy[graphAt] = { ...fields[graphAt]!, data: encode(out) };
    return encode(copy);
  };
  return { firstPass: build(elseBranch), withPast: build(thenBranch) };
}
