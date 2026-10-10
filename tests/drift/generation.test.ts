/** The search strategies and logits processors decide exactly as Python's. */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  BadWordsProcessor,
  ForcedTokensProcessor,
  MinLengthProcessor,
  NoRepeatNGramProcessor,
  RepetitionPenaltyProcessor,
  SuppressBlankProcessor,
  SuppressTokensProcessor,
  TemperatureWarper,
  TopKWarper,
  TopPWarper,
  type LogitsProcessor,
} from "../../src/generation/processors.js";
import { BeamSearch, GreedySearch, type SearchStrategy } from "../../src/generation/search.js";

const fixture = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "generation.json"), "utf-8"),
) as {
  vocab: number;
  eos: number;
  starts: number[];
  table: number[][];
  cases: Record<string, { sequences: number[][]; scores: number[]; sum_logprob: number[]; avg_logprob: number[]; finish_reasons: string[] }>;
  logits: number[][];
  processors: Record<string, { sequences: number[][]; expected: (number | null)[][] }>;
};

function run(strategy: SearchStrategy) {
  const { vocab, starts } = fixture;
  const table = fixture.table.map((row) => Float32Array.from(row));
  strategy.init(starts.length);
  const per = strategy.rowsPerSequence;
  let last = starts.flatMap((s) => new Array<number>(per).fill(s));
  for (let step = 0; step < 12; step += 1) {
    const data = new Float32Array(last.length * vocab);
    last.forEach((token, r) => {
      const row = table[token]!;

      for (let v = 0; v < vocab; v += 1) data[r * vocab + v] = Math.fround(row[v]! + Math.fround(0.1 * step));
    });
    const decision = strategy.step({ data, rows: last.length, vocab });
    last = decision.nextTokens;
    if (decision.allDone) break;
  }
  return strategy.finalize();
}

describe("search strategies agree with ynonnx_core.generation.search", () => {
  const strategies: Record<string, () => SearchStrategy> = {
    greedy: () => new GreedySearch({ eosTokenId: fixture.eos, padTokenId: 0, maxLength: 10 }),
    beam3: () => new BeamSearch({ beamSize: 3, eosTokenId: fixture.eos, padTokenId: 0, maxLength: 10 }),
    beam3_lp_patience: () =>
      new BeamSearch({ beamSize: 3, eosTokenId: fixture.eos, padTokenId: 0, lengthPenalty: 0.6, patience: 2, maxLength: 10 }),
    beam4_no_early: () =>
      new BeamSearch({ beamSize: 4, eosTokenId: fixture.eos, padTokenId: 0, earlyStopping: false, maxLength: 10 }),
  };

  it.each(Object.keys(strategies))("%s", (name) => {
    const expected = fixture.cases[name]!;
    const out = run(strategies[name]!());
    expect(out.sequences).toEqual(expected.sequences);
    expect(out.finishReasons).toEqual(expected.finish_reasons);
    out.scores.forEach((s, i) => expect(s).toBeCloseTo(expected.scores[i]!, 4));
    out.sumLogprob.forEach((s, i) => expect(s).toBeCloseTo(expected.sum_logprob[i]!, 4));
    out.avgLogprob.forEach((s, i) => expect(s).toBeCloseTo(expected.avg_logprob[i]!, 4));
  });
});

describe("logits processors agree with ynonnx_core.generation.logits_processors", () => {
  const processors: Record<string, LogitsProcessor> = {
    temperature: new TemperatureWarper(0.7),
    top_k: new TopKWarper(4),
    top_p: new TopPWarper(0.6),
    repetition: new RepetitionPenaltyProcessor(1.3),
    no_repeat_ngram: new NoRepeatNGramProcessor(2),
    min_length: new MinLengthProcessor(6, [3, 4]),
    suppress: new SuppressTokensProcessor([0, 15, 99]),
    suppress_blank_first: new SuppressBlankProcessor(2, 3),
    forced: new ForcedTokensProcessor({ 5: 11 }),
    bad_words: new BadWordsProcessor([[6], [2, 4, 9], [7, 7, 10]]),
  };

  it.each(Object.keys(processors))("%s", (name) => {
    const { sequences, expected } = fixture.processors[name]!;
    const data = Float32Array.from(fixture.logits.flat());
    processors[name]!.process({ data, rows: fixture.logits.length, vocab: fixture.vocab }, sequences);
    expected.flat().forEach((want, i) => {
      if (want === null) expect(data[i], `index ${i}`).toBe(Number.NEGATIVE_INFINITY);
      else expect(data[i], `index ${i}`).toBeCloseTo(want, 5);
    });
  });
});
