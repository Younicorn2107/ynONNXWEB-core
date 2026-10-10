import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { assemblePrompt, quantizeContext } from "../../src/text/prompt.js";

const fixture = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "prompt.json"), "utf-8"),
) as { cases: { length: number; quantized: number[]; built: number[]; built_small: number[] }[] };

describe("prompt plumbing agrees with ynonnx_core.text.prompt", () => {
  it.each(fixture.cases.map((c) => [c.length, c] as const))("a context of %i tokens", (length, c) => {
    const context = Array.from({ length }, (_, i) => 1000 + i);
    const quantized = quantizeContext(context);
    expect(quantized).toEqual(c.quantized);
    expect(assemblePrompt([7], quantized, [1, 2, 3], 224)).toEqual(c.built);
    expect(assemblePrompt([7], context, [1, 2, 3], 50)).toEqual(c.built_small);
  });
});
