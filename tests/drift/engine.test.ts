/** The decode loop, end to end on the real onnxruntime-web, against Python's. */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import { GenerationEngine, type GenerateOptions, type TensorFactory } from "../../src/generation/engine.js";
import { ModelBundle } from "../../src/session/bundle.js";
import type { OrtLike } from "../../src/session/create.js";
import { parseManifest } from "../../src/types/manifest.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "models");
const fixture = JSON.parse(readFileSync(join(MODELS, "table_decoder.json"), "utf-8")) as {
  manifest: Record<string, unknown>;
  prompts: number[][];
  cases: Record<string, { sequences: number[][]; sum_logprob: number[]; finish_reasons: string[] }>;
};
const bytes = readFileSync(join(MODELS, "table_decoder.onnx"));

async function engine(): Promise<GenerationEngine> {
  const bundle = await ModelBundle.create({
    ort: ort as unknown as OrtLike,
    manifest: parseManifest(fixture.manifest),
    key: { name: "table-decoder", computeType: "float32" },
    files: new Map([["model.onnx", bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer]]),
    plan: { order: ["wasm"], dropped: [], threads: 1 },
  });
  return new GenerationEngine(bundle, ort.Tensor as unknown as TensorFactory);
}

const settings: Record<string, Partial<GenerateOptions>> = {
  greedy: { strategy: "greedy" },
  beam3: { strategy: "beam", beamSize: 3 },
  no_repeat_2: { strategy: "greedy", noRepeatNgramSize: 2 },
};

describe("GenerationEngine agrees with ynonnx_core's on a decoder that runs", () => {
  it.each(Object.keys(settings))("%s", async (name) => {
    const generator = await engine();
    try {
      const expected = fixture.cases[name]!;
      const out = await generator.generate({
        prompts: fixture.prompts,
        maxNewTokens: 9,
        eosTokenId: 5,
        padTokenId: 0,
        ...settings[name],
      });
      expect(out.sequences).toEqual(expected.sequences);
      expect(out.finishReasons).toEqual(expected.finish_reasons);
      out.sumLogprob.forEach((s, i) => expect(s).toBeCloseTo(expected.sum_logprob[i]!, 3));
    } finally {
      await generator.bundle.close();
    }
  }, 60000);

  it("streams every step's tokens and can be cancelled", async () => {
    const generator = await engine();
    try {
      const seen: number[][] = [];
      const controller = new AbortController();
      await expect(
        generator.generate({
          prompts: fixture.prompts,
          maxNewTokens: 9,
          onStep: (tokens) => {
            seen.push([...tokens]);
            if (seen.length === 3) controller.abort();
          },
          signal: controller.signal,
        }),
      ).rejects.toThrow(/cancelled/);
      expect(seen).toHaveLength(3);
      expect(seen[0]).toEqual(fixture.cases["greedy"]!.sequences.map((s) => s[0]));
    } finally {
      await generator.bundle.close();
    }
  }, 60000);

  it("refuses prompts of different lengths", async () => {
    const generator = await engine();
    try {
      await expect(generator.generate({ prompts: [[1, 2], [3]] })).rejects.toThrow(/mixed lengths/);
    } finally {
      await generator.bundle.close();
    }
  }, 60000);
});
