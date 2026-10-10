/** A graph's weights moved out of it, as if it had been published with external data. */

import { encode, parse, type Field } from "./protobuf.js";

const MODEL_GRAPH = 7;
const GRAPH_INITIALIZER = 5;
const TENSOR_RAW = 9;
const TENSOR_EXTERNAL = 13;
const TENSOR_LOCATION = 14;
const ENTRY_KEY = 1;
const ENTRY_VALUE = 2;
const EXTERNAL = 1n;

/** Tensors smaller than this stay in the graph (shapes, scalars, small tables). */
const MIN_BYTES = 1024;

const text = (value: string) => new TextEncoder().encode(value);

function entry(key: string, value: string): Field {
  return { num: TENSOR_EXTERNAL, wire: 2, data: encode([{ num: ENTRY_KEY, wire: 2, data: text(key) }, { num: ENTRY_VALUE, wire: 2, data: text(value) }]) };
}

/**
 * `model` with its large inline weights moved to `files` (named `<prefix>_<n>`), or null when it
 * holds less than `minTotal` bytes of them.
 */
export function externalizeWeights(
  model: Uint8Array,
  prefix: string,
  options: { chunkBytes?: number; minTotal?: number } = {},
): { bytes: Uint8Array; files: { path: string; data: Uint8Array }[] } | null {
  const chunkBytes = options.chunkBytes ?? 2 ** 30;
  const minTotal = options.minTotal ?? 512 * 2 ** 20;
  const fields = parse(model);
  const graphAt = fields.findIndex((f) => f.num === MODEL_GRAPH);
  if (graphAt < 0) return null;
  const graph = parse(fields[graphAt]!.data!);

  let total = 0;
  const tensors = graph.map((f) => (f.num === GRAPH_INITIALIZER ? parse(f.data!) : null));
  for (const t of tensors) {
    const raw = t?.find((x) => x.num === TENSOR_RAW)?.data;
    if (raw && raw.length >= MIN_BYTES) total += raw.length;
  }
  if (total < minTotal) return null;

  const chunks: { path: string; parts: Uint8Array[]; size: number }[] = [];
  const out = graph.map((f, i) => {
    const t = tensors[i];
    const raw = t?.find((x) => x.num === TENSOR_RAW)?.data;
    if (!t || !raw || raw.length < MIN_BYTES) return f;
    let chunk = chunks[chunks.length - 1];
    if (!chunk || (chunk.size > 0 && chunk.size + raw.length > chunkBytes)) {
      chunk = { path: `${prefix}_${chunks.length}`, parts: [], size: 0 };
      chunks.push(chunk);
    }
    const offset = chunk.size;
    chunk.parts.push(raw);
    chunk.size += raw.length;
    const kept = t.filter((x) => x.num !== TENSOR_RAW && x.num !== TENSOR_EXTERNAL && x.num !== TENSOR_LOCATION);
    return {
      ...f,
      data: encode([
        ...kept,
        entry("location", chunk.path),
        entry("offset", String(offset)),
        entry("length", String(raw.length)),
        { num: TENSOR_LOCATION, wire: 0, int: EXTERNAL },
      ]),
    };
  });

  const files = chunks.map((chunk) => {
    const data = new Uint8Array(chunk.size);
    let at = 0;
    for (const part of chunk.parts) {
      data.set(part, at);
      at += part.length;
    }
    return { path: chunk.path, data };
  });
  const copy = [...fields];
  copy[graphAt] = { ...fields[graphAt]!, data: encode(out) };
  return { bytes: encode(copy), files };
}

/**
 * A weight file too large for one buffer, cut into parts at tensor boundaries: the parts (`[start,
 * end)` in the file, each at most `partBytes` unless one tensor alone is larger) and the graph
 * pointing each tensor into its part (`<location>.part<n>`, offsets from the part's start).
 */
export function splitExternalData(
  model: Uint8Array,
  location: string,
  partBytes = 2 ** 30,
): { bytes: Uint8Array; parts: { name: string; start: number; end: number }[] } {
  const fields = parse(model);
  const graphAt = fields.findIndex((f) => f.num === MODEL_GRAPH);
  const graph = parse(fields[graphAt]!.data!);
  const entries = (t: readonly Field[]) =>
    new Map(
      t.filter((x) => x.num === TENSOR_EXTERNAL).map((x) => {
        const e = parse(x.data!);
        const text = (n: number) => new TextDecoder().decode(e.find((y) => y.num === n)?.data ?? new Uint8Array());
        return [text(ENTRY_KEY), text(ENTRY_VALUE)] as const;
      }),
    );

  const stored = graph
    .map((f, index) => {
      if (f.num !== GRAPH_INITIALIZER) return null;
      const t = parse(f.data!);
      const e = entries(t);
      if (e.get("location") !== location) return null;
      return { index, t, offset: Number(e.get("offset") ?? 0), length: Number(e.get("length") ?? 0) };
    })
    .filter((x): x is { index: number; t: Field[]; offset: number; length: number } => x !== null)
    .sort((a, b) => a.offset - b.offset);
  const parts: { name: string; start: number; end: number }[] = [];
  const out = [...graph];
  for (const tensor of stored) {
    let part = parts[parts.length - 1];
    const end = tensor.offset + tensor.length;
    if (!part || (end - part.start > partBytes && part.end > part.start)) {
      part = { name: `${location}.part${parts.length}`, start: tensor.offset, end: tensor.offset };
      parts.push(part);
    }
    part.end = Math.max(part.end, end);
    const kept = tensor.t.filter((x) => x.num !== TENSOR_EXTERNAL);
    out[tensor.index] = {
      ...graph[tensor.index]!,
      data: encode([...kept.filter((x) => x.num !== TENSOR_LOCATION), entry("location", part.name), entry("offset", String(tensor.offset - part.start)), entry("length", String(tensor.length)), { num: TENSOR_LOCATION, wire: 0, int: EXTERNAL }]),
    };
  }
  const copy = [...fields];
  copy[graphAt] = { ...fields[graphAt]!, data: encode(out) };
  return { bytes: encode(copy), parts };
}
