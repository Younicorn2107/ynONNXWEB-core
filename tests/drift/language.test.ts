import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { FastTextModel, fasttextHash } from "../../src/text/fasttext.js";
import { LanguageDetector } from "../../src/text/language.js";

const here = dirname(fileURLToPath(import.meta.url));
const MODEL = join(here, "..", ".cache", "lid.176.ftz");
const f = JSON.parse(readFileSync(join(here, "..", "fixtures", "language.json"), "utf-8")) as {
  scores: { text: string; answer: [string, number] | null }[];
  segments: { text: string; lang: string; start: number; end: number }[];
  mixed: string;
};

describe("fastText", () => {
  it("hashes bytes as fastText does (signed bytes)", () => {
    expect(fasttextHash(new TextEncoder().encode("a"))).toBe(3826002220);
    expect(fasttextHash(new Uint8Array([0xc3, 0xa9]))).toBe(fasttextHash(new TextEncoder().encode("é")));
  });
});

describe.skipIf(!existsSync(MODEL))("language detection, against ynlibs (lid.176.ftz)", () => {
  const detector = existsSync(MODEL) ? new LanguageDetector(FastTextModel.load(readFileSync(MODEL))) : null;

  it("names the same language with the same score", () => {
    for (const { text, answer } of f.scores) {
      const got = detector!.languageAndScore(text);
      expect(got?.[0], text).toBe(answer?.[0]);
      expect(got?.[1], text).toBeCloseTo(answer![1], 4);
    }
  });

  it("cuts a mixed text into the same runs", () => {
    expect(detector!.detectSegments(f.mixed)).toEqual(f.segments);
  });
});

const SPM = join(here, "..", ".cache", "opus-mt-fr-en.source.spm");
const spm = JSON.parse(readFileSync(join(here, "..", "fixtures", "sentencepiece.json"), "utf-8")) as { pieces: { text: string; pieces: string[]; ids: number[] }[]; special: Record<"unk" | "bos" | "eos" | "pad", number> };

describe.skipIf(!existsSync(SPM))("SentencePiece, against sentencepiece (opus-mt-fr-en source.spm)", () => {
  it("cuts text into the same pieces", async () => {
    const { SentencePieceProcessor } = await import("../../src/text/sentencepiece.js");
    const processor = new SentencePieceProcessor(new Uint8Array(readFileSync(SPM)));
    for (const k of spm.pieces) {
      expect(processor.encodePieces(k.text), JSON.stringify(k.text)).toEqual(k.pieces);
      expect(processor.encodeIds(k.text), JSON.stringify(k.text)).toEqual(k.ids);
    }
    const { unkId, bosId, eosId, padId } = processor.model;
    expect({ unk: unkId, bos: bosId, eos: eosId, pad: padId }).toEqual(spm.special);
  });
});
