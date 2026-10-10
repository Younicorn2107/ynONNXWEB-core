/** The tokenizer reproduces the Rust `tokenizers` library, family by family. */

import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { Tokenizer } from "../../src/text/tokenizer.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const COMMITTED = join(HERE, "..", "fixtures", "tokenizers");
const CACHED = join(HERE, "..", ".cache", "tokenizers");

interface Case {
  text: string;
  ids: number[];
  tokens: string[];
  offsets: [number, number][];
  ids_no_specials: number[];
  decoded: string;
  decoded_with_specials: string;
  pieces: string[];
}

interface Expected {
  vocab_size: number;
  pad_id: number;
  special_ids: Record<string, number>;
  committed: boolean;
  cases: Case[];
}

/** Code-point offsets (Python) → UTF-16 offsets (JS). */
function toUtf16(text: string, codePoint: number): number {
  let units = 0;
  let points = 0;
  for (const char of text) {
    if (points === codePoint) break;
    units += char.length;
    points += 1;
  }
  return units;
}

function visible(text: string, [start, end]: [number, number]): [number, number] {
  let from = toUtf16(text, start);
  let to = toUtf16(text, end);
  while (from < to && /\s/u.test(text[from]!)) from += 1;
  while (to > from && /\s/u.test(text[to - 1]!)) to -= 1;
  return [from, to];
}

for (const name of readdirSync(COMMITTED)) {
  const expected = JSON.parse(readFileSync(join(COMMITTED, name, "expected.json"), "utf-8")) as Expected;
  const path = join(expected.committed ? COMMITTED : CACHED, name, "tokenizer.json");
  const available = existsSync(path);

  describe.skipIf(!available)(`${name} tokenizer`, () => {
    const tokenizer = available ? Tokenizer.fromJson(readFileSync(path, "utf-8")) : null!;

    it("has the same vocabulary size, pad id and specials", () => {
      expect(tokenizer.vocabSize).toBe(expected.vocab_size);
      expect(tokenizer.resolvePadId()).toBe(expected.pad_id);
      expect(tokenizer.specialIds()).toEqual(expected.special_ids);
    });

    it.each(expected.cases.map((c) => [JSON.stringify(c.text).slice(0, 40), c] as const))(
      "encodes and decodes %s",
      (_label, c) => {
        expect(tokenizer.encode(c.text)).toEqual(c.ids);
        expect(tokenizer.encode(c.text, { addSpecialTokens: false })).toEqual(c.ids_no_specials);
        expect(tokenizer.decode(c.ids)).toBe(c.decoded);
        expect(tokenizer.decode(c.ids, { skipSpecialTokens: false })).toBe(c.decoded_with_specials);
        expect(tokenizer.decodeTokens(c.ids_no_specials)).toEqual(c.pieces);
      },
    );

    it.each(expected.cases.map((c) => [JSON.stringify(c.text).slice(0, 40), c] as const))(
      "places every token of %s where Rust does",
      (_label, c) => {
        const encoded = tokenizer.encodeWithOffsets(c.text);
        expect(encoded.ids).toEqual(c.ids);
        const mismatches: string[] = [];
        c.offsets.forEach((span, i) => {
          const want = visible(c.text, span);
          const got = encoded.offsets[i]!;

          if (want[0] === want[1] && got[0] === got[1]) return;
          if (want[0] !== got[0] || want[1] !== got[1]) {
            mismatches.push(`${c.tokens[i]} rust=${JSON.stringify(want)} ours=${JSON.stringify(got)}`);
          }
        });
        expect(mismatches).toEqual([]);
      },
    );
  });
}
