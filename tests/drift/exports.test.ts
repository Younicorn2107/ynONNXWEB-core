import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  exportBlob,
  exportString,
  formatTimestamp,
  splitLongLines,
  toCsv,
  toJson,
  toMarkdown,
  toSrt,
  toTxt,
  toVtt,
  toXml,
} from "../../src/export/index.js";
import { fromPythonNames } from "../../src/tasks/wire.js";

const f = JSON.parse(
  readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "exports.json"), "utf-8"),
) as { [format: string]: string } & { result: Record<string, unknown>; records: Record<string, unknown>[]; json: unknown };

/**
 * The Python side wrote text files in text mode on Windows: its newlines are `\r\n` there and `\n`
 * in truth.
 */
const lf = (text: string | undefined) => (text ?? "").replace(/\r\n/g, "\n");

const result = fromPythonNames(f.result);

describe("exports, against ynlibs.export", () => {
  it("JSON carries Python's field names", () => {
    expect(JSON.parse(toJson(result))).toEqual(f.json);
  });
  it("TXT", () => expect(toTxt(result)).toBe(lf(f.txt)));
  it("CSV of segments, with Python's quoting and \\r\\n", () => expect(toCsv((result as { segments: unknown[] }).segments)).toBe(f.csv_segments ?? ""));
  it("CSV of mixed records", () => expect(toCsv(f.records)).toBe(f.csv_records ?? ""));
  it("XML", () => expect(toXml(result)).toBe(lf(f.xml)));
  it("Markdown with front matter and title", () => expect(toMarkdown(result, { title: "Transcript", frontMatter: true })).toBe(lf(f.md)));
  it("Markdown table", () => expect(toMarkdown(f.records)).toBe(lf(f.md_records)));
  it("Markdown text", () => expect(toMarkdown({ text: "Just text." })).toBe(lf(f.md_text)));
  it("SRT", () => expect(toSrt(result)).toBe(lf(f.srt)));
  it("VTT", () => expect(toVtt(result)).toBe(lf(f.vtt)));
  it("SRT of a plain text", () => expect(toSrt("No segments here.")).toBe(lf(f.srt_plain)));
});

describe("export helpers", () => {
  it("rounds milliseconds like Python and never shows 1000 ms", () => {
    expect(formatTimestamp(3725.0625)).toBe("01:02:05,062");
    expect(formatTimestamp(1.9995, ".")).toBe("00:00:02.000");
    expect(formatTimestamp(-3)).toBe("00:00:00,000");
  });

  it("picks the format by name and refuses unknown ones", () => {
    expect(exportString({ text: "hi" }, ".TXT")).toBe("hi\n");
    expect(() => exportString({}, "pdf")).toThrow(/Unknown export format 'pdf'/);
    expect(exportBlob({ text: "hi" }, "vtt").type).toBe("text/vtt;charset=utf-8");
  });

  it("keeps typed arrays out of tables and as shapes in XML", () => {
    const masked = { name: "m", mask: new Float32Array(4) };
    expect(toCsv([masked])).toBe("name\r\nm\r\n");
    expect(toXml(masked)).toContain("<dtype>Float32Array</dtype>");
  });

  it("splits long subtitle lines and their time", () => {
    const out = splitLongLines([{ start: 0, end: 4, text: "one two three four five six seven eight" }], 10, 1);
    expect(out.map((s) => s.text)).toEqual(["one two", "three four", "five six", "seven", "eight"]);
    expect(out[1]).toMatchObject({ start: 0.8, end: 1.6 });
  });
});
