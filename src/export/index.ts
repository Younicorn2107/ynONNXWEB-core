/** Results written out: JSON, TXT, Markdown, CSV, XML, SRT and VTT. */

import { ExportError } from "../errors.js";
import { toPythonNames } from "../tasks/wire.js";

export type ArrayPolicy = "list" | "shape" | "skip";

export interface TimedSegment {
  start: number;
  end: number;
  text: string;
  speaker?: string | null;
}

/** A JSON-ready value, recursively, with Python field names. */
export function toSerializable(result: unknown, policy: ArrayPolicy = "list", pythonNames = true): unknown {
  const walk = (value: unknown): unknown => {
    if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
      const array = value as unknown as ArrayLike<number | bigint> & { constructor: { name: string } };
      if (policy === "list") return Array.from(array, (v) => (typeof v === "bigint" ? Number(v) : v));
      if (policy === "shape") return { __ndarray__: { shape: [array.length], dtype: array.constructor.name } };
      return undefined;
    }
    if (typeof value === "bigint") return Number(value);
    if (value instanceof Map) return walk(Object.fromEntries(value));
    if (Array.isArray(value)) return value.map(walk);
    if (value !== null && typeof value === "object") {
      const out: Record<string, unknown> = {};
      for (const [key, item] of Object.entries(value)) {
        if (typeof item === "function") continue;
        const converted = walk(item);
        if (converted === undefined) continue;
        out[key] = converted;
      }
      return out;
    }
    return value;
  };
  const plain = walk(result);
  return pythonNames ? toPythonNames(plain) : plain;
}

/** The primary text of a result, a record or a string. */
export function extractText(result: unknown): string {
  if (typeof result === "string") return result;
  if (result !== null && typeof result === "object" && "text" in result) return String((result as { text: unknown }).text);
  return String(result);
}

function flattenRow(row: Record<string, unknown>): Record<string, unknown> {
  const flat: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(row)) {
    flat[key] = value === null || ["string", "number", "boolean"].includes(typeof value) ? value : JSON.stringify(value);
  }
  return flat;
}

/** Rows for a table: `toRecords()`, a list of records, or one record. Arrays are skipped. */
export function extractRecords(result: unknown, policy: ArrayPolicy = "skip"): Record<string, unknown>[] {
  const withRecords = result as { toRecords?: () => unknown[] } | null;
  const raw =
    withRecords !== null && typeof withRecords === "object" && typeof withRecords.toRecords === "function"
      ? withRecords.toRecords()
      : Array.isArray(result)
        ? result
        : [result];
  return raw.map((item) => {
    const converted = toSerializable(item, policy);
    return converted !== null && typeof converted === "object" && !Array.isArray(converted)
      ? flattenRow(converted as Record<string, unknown>)
      : { value: converted };
  });
}

/**
 * `[{start, end, text, speaker}]` from a result's `segments`, or a list given directly (records or
 * `[start, end, text]`). Empty texts are dropped. With `defaultDuration`, a result without segments
 * becomes one cue of its text.
 */
export function extractSegments(result: unknown, defaultDuration?: number): TimedSegment[] {
  const list = Array.isArray(result)
    ? result
    : result !== null && typeof result === "object"
      ? (result as { segments?: unknown[] }).segments
      : undefined;
  if (list === undefined) {
    const text = defaultDuration !== undefined ? extractText(result).trim() : "";
    return text ? [{ start: 0, end: defaultDuration!, text, speaker: null }] : [];
  }
  const out: TimedSegment[] = [];
  for (const item of list) {
    const row = Array.isArray(item)
      ? { start: item[0], end: item[1], text: item[2] }
      : (item as { start?: unknown; end?: unknown; text?: unknown; speaker?: unknown });
    const text = String(row.text ?? "").trim();
    if (!text) continue;
    const speaker = (row as { speaker?: unknown }).speaker;
    out.push({
      start: Number(row.start ?? 0),
      end: Number(row.end ?? 0),
      text,
      speaker: typeof speaker === "string" ? speaker : null,
    });
  }
  return out;
}

/** Python's `round()` to an integer: a half goes to the even neighbour. */
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff === 0.5) return floor % 2 === 0 ? floor : floor + 1;
  return Math.round(value);
}

/** `HH:MM:SS,mmm` (SRT) or `HH:MM:SS.mmm` (VTT), in whole milliseconds; negative times are zero. */
export function formatTimestamp(seconds: number, separator = ","): string {
  const total = roundHalfEven(Math.max(0, seconds) * 1000);
  const pad = (n: number, width = 2) => String(n).padStart(width, "0");
  const hours = Math.floor(total / 3_600_000);
  const minutes = Math.floor((total % 3_600_000) / 60_000);
  const secs = Math.floor((total % 60_000) / 1000);
  return `${pad(hours)}:${pad(minutes)}:${pad(secs)}${separator}${pad(total % 1000, 3)}`;
}

export function toJson(result: unknown, options: { policy?: ArrayPolicy; indent?: boolean } = {}): string {
  try {
    return JSON.stringify(toSerializable(result, options.policy ?? "list"), null, options.indent === false ? undefined : 2);
  } catch (cause) {
    throw new ExportError(`JSON export failed: ${String(cause)}`, { cause });
  }
}

export function toTxt(result: unknown): string {
  const text = extractText(result);
  return text && !text.endsWith("\n") ? `${text}\n` : text;
}

function csvCell(value: unknown, delimiter: string): string {
  if (value === null || value === undefined) return "";
  const text = typeof value === "boolean" ? (value ? "True" : "False") : String(value);
  return /["\r\n]/.test(text) || text.includes(delimiter) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** One row per record, columns in first-seen order, `\r\n` line ends (Python's `csv`). */
export function toCsv(result: unknown, options: { delimiter?: string } = {}): string {
  const delimiter = options.delimiter ?? ",";
  const records = extractRecords(result);
  const columns: string[] = [];
  for (const record of records) for (const key of Object.keys(record)) if (!columns.includes(key)) columns.push(key);
  const lines = [columns.map((c) => csvCell(c, delimiter)).join(delimiter)];
  for (const record of records) lines.push(columns.map((c) => csvCell(record[c], delimiter)).join(delimiter));
  return `${lines.join("\r\n")}\r\n`;
}

function xmlEscape(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function safeTag(tag: string): string {
  let cleaned = Array.from(tag, (c) => (/[\p{L}\p{N}._-]/u.test(c) ? c : "_")).join("");
  if (!cleaned || /^\d/.test(cleaned)) cleaned = `_${cleaned}`;
  return cleaned;
}

/**
 * Keys become elements, lists repeat `<item>`, scalars become text; indented by two spaces, with
 * the declaration ElementTree writes.
 */
export function toXml(result: unknown, options: { rootTag?: string; policy?: ArrayPolicy } = {}): string {
  const lines: string[] = ["<?xml version='1.0' encoding='utf-8'?>"];
  const element = (tag: string, value: unknown, depth: number) => {
    const indent = "  ".repeat(depth);
    const children: [string, unknown][] = Array.isArray(value)
      ? value.map((item) => ["item", item])
      : value !== null && typeof value === "object"
        ? Object.entries(value).map(([k, v]) => [safeTag(k), v])
        : [];
    if (children.length === 0) {
      const text = value === null || value === undefined || typeof value === "object" ? "" : xmlEscape(pyString(value));
      lines.push(text ? `${indent}<${tag}>${text}</${tag}>` : `${indent}<${tag} />`);
      return;
    }
    lines.push(`${indent}<${tag}>`);
    for (const [childTag, child] of children) element(childTag, child, depth + 1);
    lines.push(`${indent}</${tag}>`);
  };
  element(options.rootTag ?? "result", toSerializable(result, options.policy ?? "shape"), 0);
  return lines.join("\n");
}

/** Python's `str()` of a scalar: booleans capitalized. */
function pyString(value: unknown): string {
  if (typeof value === "boolean") return value ? "True" : "False";
  return String(value);
}

const FRONT_MATTER_FIELDS = ["model_name", "model", "language", "source_lang", "target_lang", "strategy"];

/**
 * Timestamped segments (speaker in bold), else the text, else a table; optional front matter and
 * title.
 */
export function toMarkdown(
  result: unknown,
  options: { title?: string; frontMatter?: boolean; metadata?: Record<string, unknown>; maxTableRows?: number } = {},
): string {
  const parts: string[] = [];
  if (options.metadata !== undefined || options.frontMatter) {
    let meta = options.metadata;
    if (meta === undefined) {
      meta = {};
      const named = toSerializable(result) as Record<string, unknown> | null;
      for (const key of FRONT_MATTER_FIELDS) {
        const value = named && typeof named === "object" ? named[key] : undefined;
        if (["string", "number", "boolean"].includes(typeof value)) meta[key] = value;
      }
    }
    const entries = Object.entries(meta);
    if (entries.length > 0) parts.push(`${["---", ...entries.map(([k, v]) => `${k}: ${pyString(v)}`), "---"].join("\n")}\n`);
  }
  if (options.title) parts.push(`# ${options.title}\n`);
  const segments = extractSegments(result);
  if (segments.length > 0) {
    for (const s of segments) {
      const stamp = formatTimestamp(s.start, ".").slice(0, -4);
      parts.push(`\`${stamp}\` ${s.speaker ? `**${s.speaker}** ` : ""}${s.text}\n`);
    }
    return parts.join("\n");
  }
  const carriesText =
    typeof result === "string" ? result.trim().length > 0 : result !== null && typeof result === "object" && (result as { text?: unknown }).text !== null && (result as { text?: unknown }).text !== undefined;
  if (carriesText) {
    parts.push(`${extractText(result).trim()}\n`);
    return parts.join("\n");
  }
  const records = extractRecords(result);
  if (records.length > 0) {
    const max = Math.max(1, options.maxTableRows ?? 500);
    const columns: string[] = [];
    for (const record of records) for (const key of Object.keys(record)) if (!columns.includes(key)) columns.push(key);
    if (columns.length > 0) {
      const cell = (v: unknown) => (v === null || v === undefined ? "" : pyString(v).replace(/\|/g, "\\|").replace(/\n/g, " "));
      const rows = [`| ${columns.join(" | ")} |`, `| ${columns.map(() => "---").join(" | ")} |`];
      for (const record of records.slice(0, max)) rows.push(`| ${columns.map((c) => cell(record[c])).join(" | ")} |`);
      if (records.length > max) rows.push(`\n_${records.length - max} more rows omitted._`);
      parts.push(rows.join("\n"));
    }
  }
  return parts.join("\n");
}

/** SubRip; a speaker is kept as `[name]`. */
export function toSrt(result: unknown, options: { defaultDuration?: number } = {}): string {
  return extractSegments(result, options.defaultDuration ?? 5)
    .map((s, i) => `${i + 1}\n${formatTimestamp(s.start, ",")} --> ${formatTimestamp(s.end, ",")}\n${s.speaker ? `[${s.speaker}] ` : ""}${s.text}\n`)
    .join("\n");
}

/** WebVTT; a speaker is kept as a `<v name>` voice. */
export function toVtt(result: unknown, options: { defaultDuration?: number } = {}): string {
  const lines = ["WEBVTT", ""];
  for (const s of extractSegments(result, options.defaultDuration ?? 5)) {
    lines.push(`${formatTimestamp(s.start, ".")} --> ${formatTimestamp(s.end, ".")}`);
    lines.push(s.speaker ? `<v ${s.speaker}>${s.text}` : s.text);
    lines.push("");
  }
  return lines.join("\n");
}

type Renderer = (result: unknown, options?: Record<string, unknown>) => string;

/** Renderers by format name, with the usual aliases. */
export const EXPORTERS: Readonly<Record<string, Renderer>> = {
  json: toJson as Renderer,
  txt: toTxt as Renderer,
  text: toTxt as Renderer,
  md: toMarkdown as Renderer,
  markdown: toMarkdown as Renderer,
  csv: toCsv as Renderer,
  xml: toXml as Renderer,
  srt: toSrt as Renderer,
  vtt: toVtt as Renderer,
};

const MIME: Readonly<Record<string, string>> = {
  json: "application/json",
  txt: "text/plain",
  text: "text/plain",
  md: "text/markdown",
  markdown: "text/markdown",
  csv: "text/csv",
  xml: "application/xml",
  srt: "application/x-subrip",
  vtt: "text/vtt",
};

function rendererFor(format: string): [string, Renderer] {
  const name = format.toLowerCase().replace(/^\./, "");
  const renderer = EXPORTERS[name];
  if (!renderer) {
    throw new ExportError(`Unknown export format '${format}'. Known: ${Object.keys(EXPORTERS).sort().join(", ")}`);
  }
  return [name, renderer];
}

/** A result in one format, as a string. */
export function exportString(result: unknown, format: string, options: Record<string, unknown> = {}): string {
  const [, renderer] = rendererFor(format);
  return renderer(result, options);
}

/** A result in one format, as a Blob ready for a download link. */
export function exportBlob(result: unknown, format: string, options: Record<string, unknown> = {}): Blob {
  const [name, renderer] = rendererFor(format);
  return new Blob([renderer(result, options)], { type: `${MIME[name] ?? "text/plain"};charset=utf-8` });
}

/**
 * Break subtitle text onto lines of at most `maxWidth` characters, `maxLines` per cue, splitting
 * the cue's time in proportion: two halves sharing the parent's full span overlap, and players then
 * show only one.
 */
export function splitLongLines<S extends TimedSegment>(segments: readonly S[], maxWidth: number, maxLines = 2): S[] {
  if (maxWidth <= 0) return [...segments];
  const out: S[] = [];
  for (const segment of segments) {
    const words = segment.text.split(/\s+/).filter(Boolean);
    if (words.length === 0) continue;
    const lines: string[] = [];
    let current = "";
    for (const word of words) {
      const candidate = `${current} ${word}`.trim();
      if (candidate.length > maxWidth && current) {
        lines.push(current);
        current = word;
      } else {
        current = candidate;
      }
    }
    if (current) lines.push(current);
    const chunks: string[][] = [];
    for (let i = 0; i < lines.length; i += maxLines) chunks.push(lines.slice(i, i + maxLines));
    const span = (segment.end - segment.start) / Math.max(chunks.length, 1);
    chunks.forEach((chunk, index) => {
      const start = segment.start + index * span;
      out.push({
        ...segment,
        id: out.length,
        text: chunk.join("\n"),
        start: Math.round(start * 1000) / 1000,
        end: Math.round(Math.min(start + span, segment.end) * 1000) / 1000,
      });
    });
  }
  return out;
}
