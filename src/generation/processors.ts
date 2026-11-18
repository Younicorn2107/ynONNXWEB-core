/**
 * Transformations applied to the logits before the next token is picked, and the conditions that
 * end a sequence.
 */

import type { Tokenizer } from "../text/tokenizer.js";

/** A block of logits: `rows` rows of `vocab` values, row-major. */
export interface Logits {
  data: Float32Array;
  rows: number;
  vocab: number;
}

/** The ids generated so far, one array per row. */
export type Sequences = readonly (readonly number[])[];

export const NEG_INF = Number.NEGATIVE_INFINITY;

export interface LogitsProcessor {
  process(logits: Logits, sequences: Sequences): void;
  /**
   * True when the processor only ever lowers logits (a ban, a penalty) and reads no more of a row
   * than the entries it changes. A greedy step can then be decided from the few best candidates the
   * graph computed (`greedyHead`): the best after the rule is among them unless the rule pushed
   * them all below the next one, which the engine sees, and then reads the whole row. Absent means
   * no: a rule of its own (Whisper's timestamps) reads the whole distribution.
   */
  readonly lowersOnly?: boolean;
}

/** Applies a chain of processors in declaration order. */
export function applyProcessors(processors: readonly LogitsProcessor[], logits: Logits, sequences: Sequences): void {
  for (const processor of processors) processor.process(logits, sequences);
}

function row(logits: Logits, index: number): Float32Array {
  return logits.data.subarray(index * logits.vocab, (index + 1) * logits.vocab);
}

/** logits / T. T <= 0 is a no-op: selection is argmax anyway. */
export class TemperatureWarper implements LogitsProcessor {
  constructor(readonly temperature = 1.0) {}
  process(logits: Logits): void {
    const t = this.temperature;
    if (t <= 0 || t === 1) return;
    const data = logits.data;
    for (let i = 0; i < data.length; i += 1) data[i] = data[i]! / t;
  }
}

/** Keep the k highest logits per row, push the rest to -inf. */
export class TopKWarper implements LogitsProcessor {
  constructor(readonly topK = 0) {}
  process(logits: Logits): void {
    const k = this.topK;
    if (k <= 0 || k >= logits.vocab) return;
    for (let r = 0; r < logits.rows; r += 1) {
      const values = row(logits, r);
      const threshold = kthLargest(values, k);
      for (let i = 0; i < values.length; i += 1) if (values[i]! < threshold) values[i] = NEG_INF;
    }
  }
}

/** The k-th largest value of a row (k >= 1), by partial selection. */
export function kthLargest(values: Float32Array, k: number): number {
  const copy = Float32Array.from(values);
  let lo = 0;
  let hi = copy.length - 1;
  const target = k - 1;
  while (lo < hi) {
    const pivot = copy[(lo + hi) >> 1]!;
    let i = lo;
    let j = hi;
    while (i <= j) {
      while (copy[i]! > pivot) i += 1;
      while (copy[j]! < pivot) j -= 1;
      if (i <= j) {
        const swap = copy[i]!;
        copy[i] = copy[j]!;
        copy[j] = swap;
        i += 1;
        j -= 1;
      }
    }
    if (target <= j) hi = j;
    else if (target >= i) lo = i;
    else break;
  }
  return copy[target]!;
}

/** Nucleus sampling: the smallest set whose cumulative mass reaches p. */
export class TopPWarper implements LogitsProcessor {
  constructor(
    readonly topP = 1.0,
    readonly minTokensToKeep = 1,
  ) {}
  process(logits: Logits): void {
    if (this.topP >= 1) return;
    const keep = Math.max(1, this.minTokensToKeep);
    for (let r = 0; r < logits.rows; r += 1) {
      const values = row(logits, r);
      const order = Array.from(values.keys()).sort((a, b) => values[b]! - values[a]!);
      const max = values[order[0]!]!;
      let total = 0;
      const probs = new Float64Array(order.length);
      order.forEach((index, rank) => {
        probs[rank] = Math.exp(values[index]! - max);
        total += probs[rank]!;
      });
      let cumulative = 0;
      for (let rank = 0; rank < order.length; rank += 1) {
        const p = probs[rank]! / total;

        if (rank >= keep && cumulative > this.topP) values[order[rank]!] = NEG_INF;
        cumulative += p;
      }
    }
  }
}

/** Keep tokens whose probability is at least `minP` times the best one. */
export class MinPWarper implements LogitsProcessor {
  constructor(readonly minP = 0) {}
  process(logits: Logits): void {
    if (this.minP <= 0) return;
    const cut = Math.log(this.minP);
    for (let r = 0; r < logits.rows; r += 1) {
      const values = row(logits, r);
      let max = NEG_INF;
      for (const v of values) if (v > max) max = v;
      for (let i = 0; i < values.length; i += 1) if (values[i]! - max < cut) values[i] = NEG_INF;
    }
  }
}

/** Divide (logit > 0) or multiply (logit < 0) the tokens already seen. */
export class RepetitionPenaltyProcessor implements LogitsProcessor {
  constructor(readonly penalty = 1.0) {}
  /** Below 1 it raises the tokens already seen. */
  get lowersOnly(): boolean {
    return this.penalty >= 1;
  }
  process(logits: Logits, sequences: Sequences): void {
    if (this.penalty === 1) return;
    for (let r = 0; r < logits.rows; r += 1) {
      const values = row(logits, r);
      for (const token of new Set(sequences[r] ?? [])) {
        if (token < 0 || token >= values.length) continue;
        const v = values[token]!;
        values[token] = v > 0 ? v / this.penalty : v * this.penalty;
      }
    }
  }
}

/**
 * Presence and frequency penalties, as the OpenAI API defines them: subtract `presence` once from
 * every token seen, and `frequency` per occurrence.
 */
export class PresenceFrequencyPenaltyProcessor implements LogitsProcessor {
  constructor(
    readonly presence = 0,
    readonly frequency = 0,
  ) {}
  /** A negative penalty raises the tokens already seen. */
  get lowersOnly(): boolean {
    return this.presence >= 0 && this.frequency >= 0;
  }
  process(logits: Logits, sequences: Sequences): void {
    if (this.presence === 0 && this.frequency === 0) return;
    for (let r = 0; r < logits.rows; r += 1) {
      const values = row(logits, r);
      const counts = new Map<number, number>();
      for (const token of sequences[r] ?? []) counts.set(token, (counts.get(token) ?? 0) + 1);
      for (const [token, count] of counts) {
        if (token < 0 || token >= values.length) continue;
        values[token] = values[token]! - this.presence - this.frequency * count;
      }
    }
  }
}

/** Ban any n-gram already emitted within the same sequence. */
export class NoRepeatNGramProcessor implements LogitsProcessor {
  readonly lowersOnly = true;
  constructor(readonly ngramSize = 0) {}
  process(logits: Logits, sequences: Sequences): void {
    const n = this.ngramSize;
    if (n <= 0) return;
    for (let r = 0; r < logits.rows; r += 1) {
      const tokens = sequences[r] ?? [];
      if (tokens.length < n) continue;
      const values = row(logits, r);
      const prefix = n > 1 ? tokens.slice(tokens.length - (n - 1)) : [];
      for (let start = 0; start + n <= tokens.length; start += 1) {
        let same = true;
        for (let i = 0; i < n - 1; i += 1) {
          if (tokens[start + i] !== prefix[i]) {
            same = false;
            break;
          }
        }
        if (!same) continue;
        const banned = tokens[start + n - 1]!;
        if (banned >= 0 && banned < values.length) values[banned] = NEG_INF;
      }
    }
  }
}

/** Hold EOS at -inf until `minLength` tokens have been generated. */
export class MinLengthProcessor implements LogitsProcessor {
  readonly lowersOnly = true;
  readonly eosIds: number[];
  constructor(
    readonly minLength: number,
    eos: number | readonly number[],
  ) {
    this.eosIds = typeof eos === "number" ? [eos] : [...eos];
  }
  process(logits: Logits, sequences: Sequences): void {
    for (let r = 0; r < logits.rows; r += 1) {
      if ((sequences[r]?.length ?? 0) >= this.minLength) continue;
      const values = row(logits, r);
      for (const eos of this.eosIds) if (eos >= 0 && eos < values.length) values[eos] = NEG_INF;
    }
  }
}

/** Hold a fixed list of ids at -inf (markers, non-textual tokens). */
export class SuppressTokensProcessor implements LogitsProcessor {
  readonly lowersOnly = true;
  readonly ids: number[];
  constructor(ids: readonly number[]) {
    this.ids = [...ids];
  }
  process(logits: Logits): void {
    for (let r = 0; r < logits.rows; r += 1) {
      const values = row(logits, r);
      for (const id of this.ids) if (id >= 0 && id < values.length) values[id] = NEG_INF;
    }
  }
}

/** Ban a leading space and EOS at the very first generated position only. */
export class SuppressBlankProcessor implements LogitsProcessor {
  readonly lowersOnly = true;
  constructor(
    readonly spaceTokenId: number | null,
    readonly eosTokenId: number | null,
  ) {}
  process(logits: Logits, sequences: Sequences): void {
    for (let r = 0; r < logits.rows; r += 1) {
      if ((sequences[r]?.length ?? 0) !== 0) continue;
      const values = row(logits, r);
      for (const token of [this.spaceTokenId, this.eosTokenId]) {
        if (token !== null && token >= 0 && token < values.length) values[token] = NEG_INF;
      }
    }
  }
}

/** Force given ids at given generated positions (`{position: id}`). */
export class ForcedTokensProcessor implements LogitsProcessor {
  readonly forced: ReadonlyMap<number, number>;
  constructor(forced: Readonly<Record<number, number>> | ReadonlyMap<number, number>) {
    this.forced = forced instanceof Map ? forced : new Map(Object.entries(forced).map(([k, v]) => [Number(k), v]));
  }
  process(logits: Logits, sequences: Sequences): void {
    for (let r = 0; r < logits.rows; r += 1) {
      const token = this.forced.get(sequences[r]?.length ?? 0);
      if (token === undefined || token < 0 || token >= logits.vocab) continue;
      const values = row(logits, r);
      values.fill(NEG_INF);
      values[token] = 0;
    }
  }
}

/** Ban whole token sequences, not just single ids. */
export class BadWordsProcessor implements LogitsProcessor {
  readonly lowersOnly = true;
  readonly words: number[][];
  constructor(words: readonly (readonly number[])[]) {
    this.words = words.filter((w) => w.length > 0).map((w) => [...w]);
  }
  process(logits: Logits, sequences: Sequences): void {
    for (let r = 0; r < logits.rows; r += 1) {
      const tokens = sequences[r] ?? [];
      const values = row(logits, r);
      for (const word of this.words) {
        const last = word[word.length - 1]!;
        if (last < 0 || last >= values.length) continue;
        if (word.length === 1) {
          values[last] = NEG_INF;
          continue;
        }
        const head = word.slice(0, -1);
        if (tokens.length < head.length) continue;
        const tail = tokens.slice(tokens.length - head.length);
        if (tail.every((t, i) => t === head[i])) values[last] = NEG_INF;
      }
    }
  }
}

export interface StoppingCriterion {
  /** One flag per row: stop it now. */
  check(sequences: Sequences, step: number): boolean[];
  /** Forget per-run state; called before each generation. */
  reset?(): void;
  /** The finish reason a stop by this criterion records. */
  readonly reason: string;
}

/** Stop once the generated part reaches `maxLength`. */
export class MaxLengthCriteria implements StoppingCriterion {
  readonly reason = "max_length";
  constructor(readonly maxLength: number) {}
  check(sequences: Sequences): boolean[] {
    return sequences.map((s) => s.length >= this.maxLength);
  }
}

/** Stop a row the moment it emits one of the end ids. */
export class EOSTokenCriteria implements StoppingCriterion {
  readonly reason = "eos";
  readonly eosIds: ReadonlySet<number>;
  constructor(eos: number | readonly number[]) {
    this.eosIds = new Set(typeof eos === "number" ? [eos] : eos);
  }
  check(sequences: Sequences): boolean[] {
    return sequences.map((s) => s.length > 0 && this.eosIds.has(s[s.length - 1]!));
  }
}

/** Stop on a literal string, decoded from the last tokens of each row. */
export class StopStringCriteria implements StoppingCriterion {
  readonly reason = "stop_string";
  readonly stops: string[];
  constructor(
    stops: readonly string[],
    readonly tokenizer: Tokenizer,
    readonly lookbackTokens = 16,
  ) {
    this.stops = stops.filter((s) => s.length > 0);
  }
  check(sequences: Sequences): boolean[] {
    return sequences.map((s) => {
      if (this.stops.length === 0 || s.length === 0) return false;
      const text = this.tokenizer.decode(s.slice(-this.lookbackTokens));
      return this.stops.some((stop) => text.includes(stop));
    });
  }
}

/** A wall-clock bound on one generation: a degenerate loop cannot hold the device forever. */
export class MaxTimeCriteria implements StoppingCriterion {
  readonly reason = "max_length";
  #started: number | null = null;
  constructor(readonly maxTimeMs: number) {}
  reset(): void {
    this.#started = null;
  }
  check(sequences: Sequences): boolean[] {
    this.#started ??= performance.now();
    const expired = performance.now() - this.#started >= this.maxTimeMs;
    return sequences.map(() => expired);
  }
}
