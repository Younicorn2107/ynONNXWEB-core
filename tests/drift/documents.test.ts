import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { mapCues, splitFrontMatter, splitLines, splitLinesKeepEnds } from "../../src/text/documents.js";

const here = dirname(fileURLToPath(import.meta.url));
const f = JSON.parse(readFileSync(join(here, "..", "fixtures", "documents.json"), "utf-8")) as {
  front: { text: string; body: string; metadata: Record<string, string> }[];
  cues: { content: string; texts: string[]; text: string; count: number }[];
  lines: { text: string; lines: string[]; keep: string[] }[];
};

describe("documents, against ynlibs.text", () => {
  it("splits a front matter as split_front_matter does", () => {
    for (const { text, body, metadata } of f.front) expect(splitFrontMatter(text)).toEqual([body, metadata]);
  });

  it("maps cues as map_cues does", async () => {
    for (const { content, texts, text, count } of f.cues) {
      let seen: string[] = [];
      const got = await mapCues(content, (all) => {
        seen = all;
        return all.map((t, i) => (i % 2 ? t.toUpperCase().replaceAll("\n", " / ") : `${t}\nextra`));
      });
      expect(seen).toEqual(texts);
      expect(got).toEqual([text, count]);
    }
  });

  it("splits lines as str.splitlines does", () => {
    for (const { text, lines, keep } of f.lines) {
      expect(splitLines(text)).toEqual(lines);
      expect(splitLinesKeepEnds(text)).toEqual(keep);
    }
  });
});
