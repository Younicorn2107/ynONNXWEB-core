/** fastText supervised models, read and run in plain TypeScript. */

import { ConfigurationError } from "../errors.js";

const MAGIC = 793712314;
const EOS = "</s>";
const BOW = "<";
const EOW = ">";
const LABEL = "__label__";

const enum Loss {
  HierarchicalSoftmax = 1,
  NegativeSampling = 2,
  Softmax = 3,
  OneVsAll = 4,
}

interface Args {
  dim: number;
  wordNgrams: number;
  loss: number;
  model: number;
  bucket: number;
  minn: number;
  maxn: number;
}

class Reader {
  offset = 0;
  readonly #view: DataView;
  constructor(readonly bytes: Uint8Array) {
    this.#view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  int32(): number {
    const v = this.#view.getInt32(this.offset, true);
    this.offset += 4;
    return v;
  }
  int64(): number {
    const v = Number(this.#view.getBigInt64(this.offset, true));
    this.offset += 8;
    return v;
  }
  int8(): number {
    const v = this.#view.getInt8(this.offset);
    this.offset += 1;
    return v;
  }
  bool(): boolean {
    return this.bytes[this.offset++] !== 0;
  }
  float64(): number {
    const v = this.#view.getFloat64(this.offset, true);
    this.offset += 8;
    return v;
  }
  floats(count: number): Float32Array {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = this.#view.getFloat32(this.offset + i * 4, true);
    this.offset += count * 4;
    return out;
  }
  bytesOf(count: number): Uint8Array {
    const out = this.bytes.subarray(this.offset, this.offset + count);
    this.offset += count;
    return out;
  }
  cstring(): Uint8Array {
    const end = this.bytes.indexOf(0, this.offset);
    const out = this.bytes.subarray(this.offset, end);
    this.offset = end + 1;
    return out;
  }
}

/** fastText's FNV-1a: each byte sign-extended before the xor, as `int8_t`. */
export function fasttextHash(bytes: Uint8Array): number {
  let h = 2166136261;
  for (const byte of bytes) {
    h = (h ^ ((byte << 24) >> 24 >>> 0)) >>> 0;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

class ProductQuantizer {
  readonly ksub = 256;
  constructor(
    readonly dim: number,
    readonly nsubq: number,
    readonly dsub: number,
    readonly lastdsub: number,
    readonly centroids: Float32Array,
  ) {}
  static load(r: Reader): ProductQuantizer {
    const dim = r.int32();
    const nsubq = r.int32();
    const dsub = r.int32();
    const lastdsub = r.int32();
    return new ProductQuantizer(dim, nsubq, dsub, lastdsub, r.floats(dim * 256));
  }
  centroid(m: number, i: number): number {
    return m === this.nsubq - 1 ? m * this.ksub * this.dsub + i * this.lastdsub : (m * this.ksub + i) * this.dsub;
  }
  addCode(x: Float32Array, codes: Uint8Array, t: number, alpha: number): void {
    let d = this.dsub;
    const base = this.nsubq * t;
    for (let m = 0; m < this.nsubq; m += 1) {
      const c = this.centroid(m, codes[base + m]!);
      if (m === this.nsubq - 1) d = this.lastdsub;
      for (let n = 0; n < d; n += 1) x[m * this.dsub + n] = Math.fround(x[m * this.dsub + n]! + Math.fround(alpha * this.centroids[c + n]!));
    }
  }
}

interface Matrix {
  rows: number;
  cols: number;
  addRowToVector(x: Float32Array, row: number): void;
  dotRow(x: Float32Array, row: number): number;
}

function loadDense(r: Reader): Matrix {
  const rows = r.int64();
  const cols = r.int64();
  const data = r.floats(rows * cols);
  return {
    rows,
    cols,
    addRowToVector(x, row) {
      for (let j = 0; j < cols; j += 1) x[j] = Math.fround(x[j]! + data[row * cols + j]!);
    },
    dotRow(x, row) {
      let s = 0;
      for (let j = 0; j < cols; j += 1) s = Math.fround(s + Math.fround(x[j]! * data[row * cols + j]!));
      return s;
    },
  };
}

function loadQuantized(r: Reader): Matrix {
  const qnorm = r.bool();
  const rows = r.int64();
  const cols = r.int64();
  const codesize = r.int32();
  const codes = r.bytesOf(codesize);
  const pq = ProductQuantizer.load(r);
  let normCodes: Uint8Array | null = null;
  let npq: ProductQuantizer | null = null;
  if (qnorm) {
    normCodes = r.bytesOf(rows);
    npq = ProductQuantizer.load(r);
  }
  const norm = (row: number) => (normCodes && npq ? npq.centroids[npq.centroid(0, normCodes[row]!)]! : 1);
  return {
    rows,
    cols,
    addRowToVector(x, row) {
      pq.addCode(x, codes, row, norm(row));
    },
    dotRow(x, row) {
      const v = new Float32Array(cols);
      pq.addCode(v, codes, row, norm(row));
      let s = 0;
      for (let j = 0; j < cols; j += 1) s = Math.fround(s + Math.fround(x[j]! * v[j]!));
      return s;
    },
  };
}

interface Node {
  parent: number;
  left: number;
  right: number;
  count: number;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const stdLog = (x: number) => Math.log(x + 1e-5);

export interface FastTextPrediction {
  label: string;
  probability: number;
}

export class FastTextModel {
  readonly #wordIds = new Map<string, number>();
  readonly #words: string[];
  readonly #subwords: number[][];
  readonly #prune: Map<number, number> | null;
  readonly #pruneSize: number;
  readonly #tree: Node[] = [];

  private constructor(
    readonly args: Args,
    words: string[],
    types: number[],
    counts: number[],
    readonly nwords: number,
    readonly nlabels: number,
    prune: Map<number, number> | null,
    pruneSize: number,
    private readonly input: Matrix,
    private readonly output: Matrix,
  ) {
    this.#words = words;
    words.forEach((w, i) => this.#wordIds.set(w, i));
    this.#prune = prune;
    this.#pruneSize = pruneSize;

    this.#subwords = words.map((w, i) => {
      const ids = [i];
      if (types[i] === 0 && w !== EOS) this.#computeSubwords(encoder.encode(`${BOW}${w}${EOW}`), ids);
      return ids;
    });
    if (args.loss === Loss.HierarchicalSoftmax) this.#buildTree(counts.slice(nwords));
  }

  /** Read a `.bin` or `.ftz` file. */
  static load(bytes: ArrayBuffer | Uint8Array): FastTextModel {
    const r = new Reader(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
    if (r.int32() !== MAGIC) throw new ConfigurationError("not a fastText model (bad magic number)");
    const version = r.int32();
    if (version > 12) throw new ConfigurationError(`fastText model version ${version} is newer than this reader (12)`);
    const raw = Array.from({ length: 12 }, () => r.int32());
    r.float64();
    const args: Args = { dim: raw[0]!, wordNgrams: raw[5]!, loss: raw[6]!, model: raw[7]!, bucket: raw[8]!, minn: raw[9]!, maxn: raw[10]! };
    const size = r.int32();
    const nwords = r.int32();
    const nlabels = r.int32();
    r.int64();
    const pruneSize = r.int64();
    const words: string[] = [];
    const counts: number[] = [];
    const types: number[] = [];
    for (let i = 0; i < size; i += 1) {
      words.push(decoder.decode(r.cstring()));
      counts.push(r.int64());
      types.push(r.int8());
    }
    let prune: Map<number, number> | null = null;
    if (pruneSize > 0) {
      prune = new Map();
      for (let i = 0; i < pruneSize; i += 1) {
        const first = r.int32();
        prune.set(first, r.int32());
      }
    }
    const quantInput = r.bool();
    const input = quantInput ? loadQuantized(r) : loadDense(r);
    const quantOutput = r.bool();
    const output = quantInput && quantOutput ? loadQuantized(r) : loadDense(r);
    if (args.model !== 3) throw new ConfigurationError("only supervised fastText models predict labels");
    return new FastTextModel(args, words, types, counts, nwords, nlabels, prune, pruneSize, input, output);
  }

  get labels(): string[] {
    return this.#words.slice(this.nwords);
  }

  #pushHash(out: number[], id: number): void {
    if (this.#pruneSize === 0 || id < 0) return;
    if (this.#pruneSize > 0) {
      const mapped = this.#prune!.get(id);
      if (mapped === undefined) return;
      id = mapped;
    }
    out.push(this.nwords + id);
  }

  /** Character n-grams of `<word>`, over UTF-8 bytes, as Dictionary::computeSubwords. */
  #computeSubwords(word: Uint8Array, out: number[]): void {
    const { minn, maxn, bucket } = this.args;
    for (let i = 0; i < word.length; i += 1) {
      if ((word[i]! & 0xc0) === 0x80) continue;
      let j = i;
      for (let n = 1; j < word.length && n <= maxn; n += 1) {
        j += 1;
        while (j < word.length && (word[j]! & 0xc0) === 0x80) j += 1;
        if (n >= minn && !(n === 1 && (i === 0 || j === word.length))) {
          this.#pushHash(out, fasttextHash(word.subarray(i, j)) % bucket);
        }
      }
    }
  }

  /** The ids a line feeds the model: words (or their n-grams when unknown), then word n-grams. */
  #lineIds(text: string): number[] {
    const ids: number[] = [];
    const hashes: number[] = [];

    const tokens = text.split(/[ \n\r\t\v\f\0]+/).filter(Boolean);
    tokens.push(EOS);
    for (const token of tokens) {
      const bytes = encoder.encode(token);
      const wid = this.#wordIds.get(token);
      const isLabel = wid === undefined ? token.startsWith(LABEL) : wid >= this.nwords;
      if (isLabel) continue;
      if (wid === undefined) {
        if (token !== EOS) this.#computeSubwords(encoder.encode(`${BOW}${token}${EOW}`), ids);
      } else if (this.args.maxn <= 0) {
        ids.push(wid);
      } else {
        ids.push(...this.#subwords[wid]!);
      }
      hashes.push(fasttextHash(bytes));
      if (token === EOS) break;
    }
    for (let i = 0; i < hashes.length; i += 1) {
      let h = BigInt(hashes[i]!);
      for (let j = i + 1; j < hashes.length && j < i + this.args.wordNgrams; j += 1) {
        h = BigInt.asUintN(64, h * 116049371n + BigInt(hashes[j]!));
        this.#pushHash(ids, Number(h % BigInt(this.args.bucket)));
      }
    }
    return ids;
  }

  #buildTree(counts: number[]): void {
    const osz = counts.length;
    for (let i = 0; i < 2 * osz - 1; i += 1) this.#tree.push({ parent: -1, left: -1, right: -1, count: 1e15 });
    for (let i = 0; i < osz; i += 1) this.#tree[i]!.count = counts[i]!;
    let leaf = osz - 1;
    let node = osz;
    for (let i = osz; i < 2 * osz - 1; i += 1) {
      const mini = [0, 0];
      for (let j = 0; j < 2; j += 1) {
        if (leaf >= 0 && this.#tree[leaf]!.count < this.#tree[node]!.count) mini[j] = leaf--;
        else mini[j] = node++;
      }
      const t = this.#tree[i]!;
      t.left = mini[0]!;
      t.right = mini[1]!;
      t.count = this.#tree[mini[0]!]!.count + this.#tree[mini[1]!]!.count;
      this.#tree[mini[0]!]!.parent = i;
      this.#tree[mini[1]!]!.parent = i;
    }
  }

  /** The `k` likeliest labels with their probabilities, above `threshold`. */
  predict(text: string, k = 1, threshold = 0): FastTextPrediction[] {
    const ids = this.#lineIds(text);
    if (ids.length === 0) return [];
    const hidden = new Float32Array(this.args.dim);
    for (const id of ids) this.input.addRowToVector(hidden, id);
    const scale = Math.fround(1 / ids.length);
    for (let j = 0; j < hidden.length; j += 1) hidden[j] = Math.fround(hidden[j]! * scale);
    const found: [number, number][] = [];
    if (this.args.loss === Loss.HierarchicalSoftmax) {
      const osz = this.nlabels;
      const logThreshold = stdLog(threshold);
      const dfs = (node: number, score: number) => {
        if (score < logThreshold) return;
        if (found.length === k && score < Math.min(...found.map((f) => f[0]))) return;
        const t = this.#tree[node]!;
        if (t.left === -1 && t.right === -1) {
          found.push([score, node]);
          found.sort((a, b) => b[0] - a[0]);
          if (found.length > k) found.pop();
          return;
        }
        const f = Math.fround(1 / (1 + Math.exp(-this.output.dotRow(hidden, node - osz))));
        dfs(t.left, score + stdLog(1 - f));
        dfs(t.right, score + stdLog(f));
      };
      dfs(2 * osz - 2, 0);
    } else {
      const logits = Array.from({ length: this.nlabels }, (_, i) => this.output.dotRow(hidden, i));
      let probabilities: number[];
      if (this.args.loss === Loss.OneVsAll) {
        probabilities = logits.map((v) => 1 / (1 + Math.exp(-v)));
      } else {
        const max = Math.max(...logits);
        const exps = logits.map((v) => Math.exp(v - max));
        const sum = exps.reduce((s, v) => s + v, 0);
        probabilities = exps.map((v) => v / sum);
      }
      probabilities.forEach((p, i) => {
        const score = stdLog(p);
        if (score >= stdLog(threshold)) found.push([score, i]);
      });
      found.sort((a, b) => b[0] - a[0]);
      found.splice(k);
    }
    return found.map(([score, i]) => ({ label: this.#words[this.nwords + i]!, probability: Math.min(Math.exp(score), 1) }));
  }
}
