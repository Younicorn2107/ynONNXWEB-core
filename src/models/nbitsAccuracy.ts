/** `MatMulNBits` told it may compute in int8 on the CPU. */

import { deeper, encode, operatorTypes, parse, str, type Field } from "./protobuf.js";

const MODEL_GRAPH = 7;
const GRAPH_NODE = 1;
const NODE_OP_TYPE = 4;
const NODE_ATTRIBUTE = 5;
const ATTR_NAME = 1;
const ATTR_I = 3;
const ATTR_G = 6;
const ATTR_GRAPHS = 11;
const ATTR_TYPE = 20;
const ATTR_TYPE_INT = 2;

export const CPU_NBITS_ACCURACY = 4;

function rewriteGraph(graph: Uint8Array, level: number, count: { n: number }, depth = 0): Uint8Array {
  const out = parse(graph).map((f): Field => {
    if (f.num !== GRAPH_NODE) return f;
    const node = parse(f.data!);
    let changed = false;

    const fields = node.map((a): Field => {
      if (a.num !== NODE_ATTRIBUTE) return a;
      const attr = parse(a.data!);
      if (!attr.some((x) => (x.num === ATTR_G || x.num === ATTR_GRAPHS) && x.data)) return a;
      changed = true;
      return { ...a, data: encode(attr.map((x) => ((x.num === ATTR_G || x.num === ATTR_GRAPHS) && x.data ? { ...x, data: rewriteGraph(x.data, level, count, deeper(depth)) } : x))) };
    });
    if (str(node.find((x) => x.num === NODE_OP_TYPE)) === "MatMulNBits") {
      const has = node.some((a) => a.num === NODE_ATTRIBUTE && str(parse(a.data!).find((x) => x.num === ATTR_NAME)) === "accuracy_level");
      if (!has) {
        fields.push({
          num: NODE_ATTRIBUTE,
          wire: 2,
          data: encode([
            { num: ATTR_NAME, wire: 2, data: new TextEncoder().encode("accuracy_level") },
            { num: ATTR_I, wire: 0, int: BigInt(level) },
            { num: ATTR_TYPE, wire: 0, int: BigInt(ATTR_TYPE_INT) },
          ]),
        });
        count.n += 1;
        changed = true;
      }
    }
    return changed ? { ...f, data: encode(fields) } : f;
  });
  return encode(out);
}

/** Every `MatMulNBits` without an `accuracy_level` given `level` (4: int8 compute). */
export function setNBitsAccuracy(model: Uint8Array, level = CPU_NBITS_ACCURACY): { bytes: Uint8Array; rewritten: number } {
  if (!operatorTypes(model).has("MatMulNBits")) return { bytes: model, rewritten: 0 };
  const fields = parse(model);
  const count = { n: 0 };
  const out = fields.map((f) => (f.num === MODEL_GRAPH && f.data ? { ...f, data: rewriteGraph(f.data, level, count) } : f));
  return count.n > 0 ? { bytes: encode(out), rewritten: count.n } : { bytes: model, rewritten: 0 };
}
