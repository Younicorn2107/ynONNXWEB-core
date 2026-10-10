/** Picking the next token(s) from the logits. */

import { GenerationError } from "../errors.js";
import { NEG_INF, type Logits, type Sequences } from "./processors.js";

export type FinishReason = "eos" | "max_length" | "stop_string" | "constraint";

export interface StepDecision {
  /** One id per row, to feed at the next step. */
  nextTokens: number[];
  /** Which row each new row continues (cache reordering), or null without a beam. */
  beamIndices: number[] | null;
  /** Per sequence. */
  finished: boolean[];
  allDone: boolean;
}

export interface GenerationOutput {
  sequences: number[][];
  scores: number[];
  sumLogprob: number[];
  avgLogprob: number[];
  finishReasons: FinishReason[];
  /** Probabilities of chosen ids at the first step (a no-speech probe, say). */
  firstStepProbs: Record<number, number>;
  /** The same probe, one record per input sequence (a batch of windows). */
  firstStepProbsPerRow: Record<number, number>[];
  steps: number;
  generatedTokens: number;
  prefillMs: number;
  decodeMs: number;
  temperature: number;
}

/** Numerically stable log-softmax of one row, in place into `out`. */
export function logSoftmax(row: Float32Array, out: Float32Array = new Float32Array(row.length)): Float32Array {
  let max = NEG_INF;
  for (const v of row) if (v > max) max = v;
  if (max === NEG_INF) {
    out.fill(NEG_INF);
    return out;
  }
  let sum = 0;
  for (const v of row) sum += Math.exp(v - max);
  const log = Math.log(sum);
  for (let i = 0; i < row.length; i += 1) out[i] = row[i]! - max - log;
  return out;
}

/**
 * The best entry of `row` and its log-probability, without writing the whole log-softmax out:
 * greedy decoding reads one value of it, and a 50 000-word vocabulary written out at every step was
 * most of a step's JavaScript.
 */
function bestOf(row: Float32Array): { token: number; logprob: number } {
  let token = 0;
  let max = NEG_INF;
  for (let i = 0; i < row.length; i += 1) {
    if (row[i]! > max) {
      max = row[i]!;
      token = i;
    }
  }
  if (max === NEG_INF) return { token: 0, logprob: NEG_INF };
  let sum = 0;
  for (let i = 0; i < row.length; i += 1) sum += Math.exp(row[i]! - max);
  return { token, logprob: -Math.log(sum) };
}

function rowOf(logits: Logits, index: number): Float32Array {
  return logits.data.subarray(index * logits.vocab, (index + 1) * logits.vocab);
}

/** Cut at the first EOS and drop trailing padding. */
export function trimSequence(tokens: readonly number[], eosIds: ReadonlySet<number>, padId: number): number[] {
  const out: number[] = [];
  for (const token of tokens) {
    if (eosIds.has(token)) return out;
    out.push(token);
  }
  while (out.length > 0 && out[out.length - 1] === padId) out.pop();
  return out;
}

function idSet(value: number | readonly number[] | null | undefined): Set<number> {
  if (value === null || value === undefined) return new Set();
  return new Set(typeof value === "number" ? [value] : value);
}

function emptyOutput(): Omit<GenerationOutput, "sequences" | "scores" | "sumLogprob" | "avgLogprob" | "finishReasons"> {
  return { firstStepProbs: {}, firstStepProbsPerRow: [], steps: 0, generatedTokens: 0, prefillMs: 0, decodeMs: 0, temperature: 0 };
}

export interface SearchStrategy {
  /** Decoder rows run per input sequence. */
  readonly rowsPerSequence: number;
  /** Tokens generated so far, per row. */
  readonly sequences: Sequences;
  init(batchSize: number): void;
  step(logits: Logits): StepDecision;
  /** External criteria closing ROWS (time, stop string). */
  markStopped(rows: readonly boolean[], reason: FinishReason): void;
  finalize(): GenerationOutput;
}

interface CommonOptions {
  eosTokenId?: number | readonly number[] | null;
  padTokenId?: number;
  maxLength?: number;
}

/** argmax. One hypothesis per sequence, no reordering. */
export class GreedySearch implements SearchStrategy {
  readonly rowsPerSequence = 1;
  readonly eosIds: Set<number>;
  readonly padId: number;
  readonly maxLength: number;
  #sequences: number[][] = [];
  #scores: number[] = [];
  #lengths: number[] = [];
  #finished: boolean[] = [];
  #reasons: FinishReason[] = [];

  constructor(options: CommonOptions = {}) {
    this.eosIds = idSet(options.eosTokenId);
    this.padId = options.padTokenId ?? 0;
    this.maxLength = options.maxLength ?? 448;
  }

  get sequences(): Sequences {
    return this.#sequences;
  }

  init(batchSize: number): void {
    this.#sequences = Array.from({ length: batchSize }, () => []);
    this.#scores = new Array<number>(batchSize).fill(0);
    this.#lengths = new Array<number>(batchSize).fill(0);
    this.#finished = new Array<boolean>(batchSize).fill(false);
    this.#reasons = new Array<FinishReason>(batchSize).fill("max_length");
  }

  step(logits: Logits): StepDecision {
    const tokens: number[] = [];
    for (let r = 0; r < logits.rows; r += 1) {
      const best = bestOf(rowOf(logits, r));
      let token = best.token;
      if (this.#finished[r]) {
        token = this.padId;
      } else {
        this.#scores[r]! += best.logprob;
        this.#lengths[r]! += 1;
        if (this.eosIds.has(token)) {
          this.#reasons[r] = "eos";
          this.#finished[r] = true;
        }
      }
      tokens.push(token);
      this.#sequences[r]!.push(token);
    }
    if ((this.#sequences[0]?.length ?? 0) >= this.maxLength) this.#finished.fill(true);
    return { nextTokens: tokens, beamIndices: null, finished: [...this.#finished], allDone: this.#finished.every(Boolean) };
  }

  markStopped(rows: readonly boolean[], reason: FinishReason): void {
    rows.forEach((stop, r) => {
      if (stop && !this.#finished[r]) {
        this.#reasons[r] = reason;
        this.#finished[r] = true;
      }
    });
  }

  finalize(): GenerationOutput {
    return {
      ...emptyOutput(),
      sequences: this.#sequences.map((s) => trimSequence(s, this.eosIds, this.padId)),
      scores: [...this.#scores],
      sumLogprob: [...this.#scores],
      avgLogprob: this.#scores.map((s, i) => s / Math.max(this.#lengths[i]!, 1)),
      finishReasons: [...this.#reasons],
    };
  }
}

export interface BeamOptions extends CommonOptions {
  beamSize?: number;
  lengthPenalty?: number;
  /** faster-whisper semantics: keep going while finished < beam × patience. */
  patience?: number;
  earlyStopping?: boolean;
}

type Hypothesis = { score: number; sum: number; tokens: number[] };

/**
 * Full beam search: cumulative log-prob, length penalty, patience. Final score: sum(log p) / length
 * ** lengthPenalty.
 */
export class BeamSearch implements SearchStrategy {
  readonly beamSize: number;
  readonly eosIds: Set<number>;
  readonly padId: number;
  readonly maxLength: number;
  readonly lengthPenalty: number;
  readonly patience: number;
  readonly earlyStopping: boolean;
  #batch = 0;
  #sequences: number[][] = [];
  #beamScores: number[][] = [];
  #done: boolean[] = [];
  #hypotheses: Hypothesis[][] = [];
  #reasons: FinishReason[] = [];

  constructor(options: BeamOptions = {}) {
    const beamSize = options.beamSize ?? 5;
    if (beamSize < 1) throw new GenerationError(`beam_size must be >= 1, got ${beamSize}`);
    this.beamSize = beamSize;
    this.eosIds = idSet(options.eosTokenId);
    this.padId = options.padTokenId ?? 0;
    this.maxLength = options.maxLength ?? 448;
    this.lengthPenalty = options.lengthPenalty ?? 1.0;
    this.patience = options.patience ?? 1.0;
    this.earlyStopping = options.earlyStopping ?? true;
  }

  get rowsPerSequence(): number {
    return this.beamSize;
  }

  get sequences(): Sequences {
    return this.#sequences;
  }

  init(batchSize: number): void {
    this.#batch = batchSize;
    this.#sequences = Array.from({ length: batchSize * this.beamSize }, () => []);

    this.#beamScores = Array.from({ length: batchSize }, () =>
      Array.from({ length: this.beamSize }, (_, i) => (i === 0 ? 0 : NEG_INF)),
    );
    this.#done = new Array<boolean>(batchSize).fill(false);
    this.#hypotheses = Array.from({ length: batchSize }, () => []);
    this.#reasons = new Array<FinishReason>(batchSize).fill("max_length");
  }

  step(logits: Logits): StepDecision {
    const batch = this.#batch;
    const beam = this.beamSize;
    const vocab = logits.vocab;
    const nextTokens = new Array<number>(batch * beam).fill(this.padId);
    const nextIndices = new Array<number>(batch * beam).fill(0);
    const nextScores = Array.from({ length: batch }, () => new Array<number>(beam).fill(NEG_INF));
    const generated = (this.#sequences[0]?.length ?? 0) + 1;

    const take = Math.min(2 * beam, beam * vocab);
    const probs = new Float32Array(vocab);

    for (let b = 0; b < batch; b += 1) {
      if (this.#done[b]) {
        for (let k = 0; k < beam; k += 1) {
          nextIndices[b * beam + k] = b * beam + k;
          nextScores[b]![k] = this.#beamScores[b]![k]!;
        }
        continue;
      }

      const best: { score: number; origin: number; token: number }[] = [];
      for (let k = 0; k < beam; k += 1) {
        const base = this.#beamScores[b]![k]!;
        if (base === NEG_INF) continue;
        logSoftmax(rowOf(logits, b * beam + k), probs);
        for (let t = 0; t < vocab; t += 1) {
          const score = base + probs[t]!;
          if (score === NEG_INF || Number.isNaN(score)) continue;
          if (best.length < take) {
            best.push({ score, origin: k, token: t });
            if (best.length === take) best.sort((x, y) => y.score - x.score);
          } else if (score > best[take - 1]!.score) {
            best[take - 1] = { score, origin: k, token: t };
            for (let i = take - 1; i > 0 && best[i]!.score > best[i - 1]!.score; i -= 1) {
              [best[i], best[i - 1]] = [best[i - 1]!, best[i]!];
            }
          }
        }
      }
      if (best.length < take) best.sort((x, y) => y.score - x.score);

      let filled = 0;
      for (const candidate of best) {
        if (this.eosIds.has(candidate.token)) {
          this.#hypotheses[b]!.push({
            score: candidate.score / generated ** this.lengthPenalty,
            sum: candidate.score,
            tokens: [...this.#sequences[b * beam + candidate.origin]!],
          });
          continue;
        }
        const slot = b * beam + filled;
        nextTokens[slot] = candidate.token;
        nextIndices[slot] = b * beam + candidate.origin;
        nextScores[b]![filled] = candidate.score;
        filled += 1;
        if (filled === beam) break;
      }
      if (filled === 0) {
        this.#done[b] = true;
        this.#reasons[b] = "eos";
        for (let k = 0; k < beam; k += 1) nextIndices[b * beam + k] = b * beam + k;
        continue;
      }
      for (let slot = filled; slot < beam; slot += 1) nextIndices[b * beam + slot] = b * beam + slot;
      if (this.#isDone(b, nextScores[b]![0]!, generated)) {
        this.#done[b] = true;
        this.#reasons[b] = "eos";
      }
    }

    this.#beamScores = nextScores;
    this.#sequences = nextIndices.map((origin, slot) => [...this.#sequences[origin]!, nextTokens[slot]!]);
    if ((this.#sequences[0]?.length ?? 0) >= this.maxLength) this.#done.fill(true);
    return { nextTokens, beamIndices: nextIndices, finished: [...this.#done], allDone: this.#done.every(Boolean) };
  }

  #isDone(b: number, bestRunning: number, generated: number): boolean {
    const required = Math.max(1, Math.round(this.beamSize * this.patience));
    const pool = this.#hypotheses[b]!;
    if (pool.length < required) return false;
    if (this.earlyStopping) return true;

    const bestPossible = bestRunning / generated ** this.lengthPenalty;
    return bestPossible <= Math.min(...pool.map((h) => h.score));
  }

  /**
   * The mask arrives per ROW (batch × beam); a sequence is only over once every one of its beams is
   * — stopping it while a live beam could still win would truncate the answer.
   */
  markStopped(rows: readonly boolean[], reason: FinishReason): void {
    for (let b = 0; b < this.#batch; b += 1) {
      let all = true;
      for (let k = 0; k < this.beamSize; k += 1) all &&= rows[b * this.beamSize + k] === true;
      if (all && !this.#done[b]) {
        this.#reasons[b] = reason;
        this.#done[b] = true;
      }
    }
  }

  finalize(): GenerationOutput {
    const sequences: number[][] = [];
    const scores: number[] = [];
    const sums: number[] = [];
    const avg: number[] = [];
    const generated = Math.max(this.#sequences[0]?.length ?? 0, 1);
    for (let b = 0; b < this.#batch; b += 1) {
      let pool = this.#hypotheses[b]!;
      if (pool.length === 0) {
        const row = this.#beamScores[b]!;
        const k = row.indexOf(Math.max(...row));
        pool = [{ score: row[k]! / generated ** this.lengthPenalty, sum: row[k]!, tokens: this.#sequences[b * this.beamSize + k]! }];
      }
      const winner = pool.reduce((best, h) => (h.score > best.score ? h : best));
      const trimmed = trimSequence(winner.tokens, this.eosIds, this.padId);
      sequences.push(trimmed);
      scores.push(winner.score);
      sums.push(winner.sum);
      avg.push(winner.sum / Math.max(trimmed.length, 1));
    }
    return { ...emptyOutput(), sequences, scores, sumLogprob: sums, avgLogprob: avg, finishReasons: [...this.#reasons] };
  }
}

/** A small seedable generator (sfc32), so sampling is reproducible in tests. */
export function seededRandom(seed: number): () => number {
  let a = 0x9e3779b9 ^ seed;
  let b = 0x243f6a88 ^ (seed * 31);
  let c = 0xb7e15162 ^ (seed * 17);
  let d = 0xdeadbeef ^ (seed * 7);
  const next = (): number => {
    a >>>= 0;
    b >>>= 0;
    c >>>= 0;
    d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 12; i += 1) next();
  return next;
}

export interface SamplingOptions extends CommonOptions {
  /** Independent samples per sequence; the best average log-prob wins. */
  bestOf?: number;
  seed?: number | null;
}

/** Multinomial sampling after the warpers. */
export class SamplingSearch implements SearchStrategy {
  readonly bestOf: number;
  readonly eosIds: Set<number>;
  readonly padId: number;
  readonly maxLength: number;
  readonly #random: () => number;
  #batch = 0;
  #sequences: number[][] = [];
  #scores: number[] = [];
  #lengths: number[] = [];
  #finished: boolean[] = [];
  #reasons: FinishReason[] = [];

  constructor(options: SamplingOptions = {}) {
    this.bestOf = Math.max(1, options.bestOf ?? 1);
    this.eosIds = idSet(options.eosTokenId);
    this.padId = options.padTokenId ?? 0;
    this.maxLength = options.maxLength ?? 448;
    this.#random = options.seed === null || options.seed === undefined ? Math.random : seededRandom(options.seed);
  }

  get rowsPerSequence(): number {
    return this.bestOf;
  }

  get sequences(): Sequences {
    return this.#sequences;
  }

  init(batchSize: number): void {
    this.#batch = batchSize;
    const rows = batchSize * this.bestOf;
    this.#sequences = Array.from({ length: rows }, () => []);
    this.#scores = new Array<number>(rows).fill(0);
    this.#lengths = new Array<number>(rows).fill(0);
    this.#finished = new Array<boolean>(rows).fill(false);
    this.#reasons = new Array<FinishReason>(batchSize).fill("max_length");
  }

  step(logits: Logits): StepDecision {
    const probs = new Float32Array(logits.vocab);
    const tokens: number[] = [];
    for (let r = 0; r < logits.rows; r += 1) {
      logSoftmax(rowOf(logits, r), probs);
      let token = this.padId;
      if (!this.#finished[r]) {
        let total = 0;
        for (const p of probs) total += Math.exp(p);
        if (total > 0) {
          let draw = this.#random() * total;
          token = probs.length - 1;
          for (let i = 0; i < probs.length; i += 1) {
            draw -= Math.exp(probs[i]!);
            if (draw <= 0) {
              token = i;
              break;
            }
          }
          this.#scores[r]! += probs[token]!;
          this.#lengths[r]! += 1;
          if (this.eosIds.has(token)) this.#finished[r] = true;
        }
      }
      tokens.push(token);
      this.#sequences[r]!.push(token);
    }
    if ((this.#sequences[0]?.length ?? 0) >= this.maxLength) this.#finished.fill(true);
    return { nextTokens: tokens, beamIndices: null, finished: [...this.#finished], allDone: this.#finished.every(Boolean) };
  }

  markStopped(rows: readonly boolean[], reason: FinishReason): void {
    rows.forEach((stop, r) => {
      if (stop && !this.#finished[r]) {
        this.#finished[r] = true;
        const b = Math.floor(r / this.bestOf);
        if (this.#reasons[b] === "max_length") this.#reasons[b] = reason;
      }
    });
  }

  finalize(): GenerationOutput {
    const sequences: number[][] = [];
    const scores: number[] = [];
    const avg: number[] = [];
    for (let b = 0; b < this.#batch; b += 1) {
      let best = b * this.bestOf;
      let bestAvg = NEG_INF;
      for (let k = 0; k < this.bestOf; k += 1) {
        const r = b * this.bestOf + k;
        const value = this.#scores[r]! / Math.max(this.#lengths[r]!, 1);
        if (value > bestAvg) {
          bestAvg = value;
          best = r;
        }
      }
      sequences.push(trimSequence(this.#sequences[best]!, this.eosIds, this.padId));
      scores.push(this.#scores[best]!);
      avg.push(bestAvg);
      if (this.#sequences[best]!.some((t) => this.eosIds.has(t))) this.#reasons[b] = "eos";
    }
    return { ...emptyOutput(), sequences, scores, sumLogprob: [...scores], avgLogprob: avg, finishReasons: [...this.#reasons] };
  }
}

/**
 * Temperature escalation at the WINDOW level: decode at T=0; if the result fails the caller's
 * quality test, decode again at the next temperature. What counts as degenerate is the library's
 * call, passed as `accept`.
 */
export async function withTemperatureLadder(
  generate: (temperature: number) => Promise<GenerationOutput>,
  temperatures: readonly number[] = [0, 0.2, 0.4, 0.6, 0.8, 1.0],
  accept?: (output: GenerationOutput) => boolean,
): Promise<GenerationOutput> {
  const ladder = temperatures.length > 0 ? temperatures : [0];
  let result: GenerationOutput | null = null;
  for (const temperature of ladder) {
    result = await generate(temperature);
    result.temperature = temperature;
    if (!accept || accept(result)) return result;
  }
  return result!;
}
