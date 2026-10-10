import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { camelCase, fromPythonNames, snakeCase, toPythonNames } from "../../src/tasks/wire.js";
import { parseManifest } from "../../src/types/manifest.js";

describe("field names across the Python boundary", () => {
  it("converts both ways", () => {
    expect(snakeCase("avgLogprob")).toBe("avg_logprob");
    expect(snakeCase("logitsVocabSize")).toBe("logits_vocab_size");
    expect(snakeCase("sizeMb")).toBe("size_mb");
    expect(camelCase("first_step_probs")).toBe("firstStepProbs");
    expect(camelCase("exposes_cross_qk")).toBe("exposesCrossQk");
  });

  it("gives a generation output the names of Python's GenerationOutput", () => {
    const wire = toPythonNames(
      { sequences: [[1, 2]], sumLogprob: [-1.5], avgLogprob: [-0.75], firstStepProbs: { 50362: 0.9 }, finishReasons: ["eos"] },
      ["first_step_probs"],
    );
    expect(wire).toEqual({
      sequences: [[1, 2]],
      sum_logprob: [-1.5],
      avg_logprob: [-0.75],
      first_step_probs: { 50362: 0.9 },
      finish_reasons: ["eos"],
    });
  });

  it("renames every field of a Python manifest to the JS one and back", () => {
    const raw = JSON.parse(
      readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "manifests", "published_merged.json"), "utf-8"),
    ) as Record<string, unknown>;
    const js = fromPythonNames(raw, ["extra"]) as Record<string, unknown>;

    expect(Object.keys(js).sort()).toEqual(Object.keys(parseManifest(raw)).sort());
    expect(toPythonNames(js, ["extra"])).toEqual(raw);
  });

  it("leaves typed arrays and data-keyed records alone", () => {
    const vectors = new Float32Array([1, 2]);
    const wire = toPythonNames({ denseVectors: vectors, languageProbs: { pt_BR: 0.1 } }, ["language_probs"]) as Record<string, unknown>;
    expect(wire["dense_vectors"]).toBe(vectors);
    expect(wire["language_probs"]).toEqual({ pt_BR: 0.1 });
  });
});
