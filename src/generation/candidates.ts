/** A greedy step decided from the few best tokens the graph computed. */

import { NEG_INF, applyProcessors, type LogitsProcessor, type Sequences } from "./processors.js";
import type { TokenPick } from "./search.js";

/** Each row's best tokens, as `greedyHead` gives them (last position only). */
export interface Candidates {
  rows: number;
  /** Candidates per row. */
  k: number;
  /** `rows × k` token ids, best first, ties to the lower id. */
  ids: ArrayLike<number>;
  /** Their raw scores. */
  values: ArrayLike<number>;
  /** Per row, log Σ exp(logit) over the whole vocabulary, before any rule. */
  logSumExp: ArrayLike<number>;
}

/**
 * Each row's greedy pick, or null when one row cannot be decided from its candidates (every
 * candidate banned, or pushed below one it does not see).
 */
export function pickFromCandidates(
  candidates: Candidates,
  processors: readonly LogitsProcessor[],
  sequences: Sequences,
  vocab: number,
  scratch?: Float32Array,
): TokenPick[] | null {
  const { rows, k, ids, values, logSumExp } = candidates;
  let processed: Float32Array | null = null;
  if (processors.length > 0) {
    const data = scratch ?? new Float32Array(rows * vocab).fill(NEG_INF);
    for (let r = 0; r < rows; r += 1) {
      for (let i = 0; i < k; i += 1) data[r * vocab + Math.round(ids[r * k + i]!)] = values[r * k + i]!;
    }
    applyProcessors(processors, { data, rows, vocab }, sequences);
    processed = new Float32Array(rows * k);
    for (let r = 0; r < rows; r += 1) {
      for (let i = 0; i < k; i += 1) processed[r * k + i] = data[r * vocab + Math.round(ids[r * k + i]!)]!;
    }
    for (let r = 0; r < rows; r += 1) {
      for (let i = 0; i < k; i += 1) data[r * vocab + Math.round(ids[r * k + i]!)] = NEG_INF;
    }
  }

  const picks: TokenPick[] = [];
  for (let r = 0; r < rows; r += 1) {
    let best = -1;
    let bestId = 0;
    let bestValue = NEG_INF;
    for (let i = 0; i < k; i += 1) {
      const value = processed ? processed[r * k + i]! : values[r * k + i]!;
      const id = Math.round(ids[r * k + i]!);
      if (value > bestValue || (value === bestValue && best >= 0 && id < bestId)) {
        best = i;
        bestId = id;
        bestValue = value;
      }
    }
    if (best < 0 || bestValue === NEG_INF) return null;

    const last = values[r * k + k - 1]!;
    const raw = values[r * k + best]!;
    if (!(bestValue > last || (bestValue === last && bestValue === raw))) return null;
    const lse = logSumExp[r]!;
    let mass = 1;
    if (processed) {
      for (let i = 0; i < k; i += 1) mass += Math.exp(processed[r * k + i]! - lse) - Math.exp(values[r * k + i]! - lse);
    }

    mass = Math.max(mass, Math.exp(bestValue - lse));
    picks.push({ token: bestId, logprob: bestValue - lse - Math.log(mass) });
  }
  return picks;
}
