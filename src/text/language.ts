/**
 * The language of a text: the port of `ynlibs.text.language` over the same model Python uses
 * (fast-langdetect's `lite`, fastText `lid.176.ftz`, CC BY-SA 3.0), run by `FastTextModel`.
 */

import type { AssetStore } from "../models/store.js";
import { FastTextModel } from "./fasttext.js";

/** Characters the detector reads; more changes nothing on a text of one language. */
export const PREFIX_CHARS = 2000;
/** Below this score the answer is `und`, not a guess. */
export const MIN_CONFIDENCE = 0.5;
/** A sentence shorter than this takes a neighbour's language. */
export const MIN_SEGMENT_CHARS = 15;

/** Where the model comes from by default: a copy of fastText's `lid.176.ftz` on the Hub. */
export const LID_MODEL_URL = "https://huggingface.co/julien-c/fasttext-language-id/resolve/main/lid.176.ftz";

export interface LanguageSegment {
  text: string;
  lang: string;
  start: number;
  end: number;
}

/** fast-langdetect's normalization: an all-caps text is lowercased (else it reads as Japanese). */
function normalize(text: string): string {
  const letters = text.match(/[A-Za-z]/g)?.length ?? 0;
  const capitals = text.match(/[A-Z]/g)?.length ?? 0;
  const cased = [...text].filter((c) => c.toLowerCase() !== c.toUpperCase());
  const isUpper = cased.length > 0 && cased.every((c) => c === c.toUpperCase());
  if (isUpper || (capitals > 0.8 * letters && text.length > 5)) return text.toLowerCase();
  return text;
}

export class LanguageDetector {
  readonly #memo = new Map<string, [string, number] | null>();

  constructor(readonly model: FastTextModel) {}

  /** Load the model from a URL (the Hub's copy by default), kept in `store` when given. */
  static async load(options: { url?: string; store?: AssetStore; signal?: AbortSignal } = {}): Promise<LanguageDetector> {
    const url = options.url ?? LID_MODEL_URL;
    const key = "language/lid.176.ftz";
    let bytes = (await options.store?.read(key)) ?? null;
    if (!bytes) {
      const response = await fetch(url, options.signal ? { signal: options.signal } : {});
      if (!response.ok) throw new Error(`the language model could not be fetched from ${url}: HTTP ${response.status}`);
      bytes = await response.arrayBuffer();
      await options.store?.write(key, new Response(bytes));
    }
    return new LanguageDetector(FastTextModel.load(bytes));
  }

  /** `(ISO 639-1 code, score)` however unsure, or null for no text. */
  languageAndScore(text: string): [string, number] | null {
    const prefix = (text ?? "").slice(0, PREFIX_CHARS).split(/\s+/).filter(Boolean).join(" ");
    if (!prefix) return null;
    if (this.#memo.has(prefix)) return this.#memo.get(prefix)!;
    const best = this.model.predict(normalize(prefix), 1)[0];
    const answer: [string, number] | null = best ? [best.label.replace("__label__", "").toLowerCase(), Math.min(best.probability, 1)] : null;
    if (this.#memo.size > 1024) this.#memo.clear();
    this.#memo.set(prefix, answer);
    return answer;
  }

  /** The code; `und` when unsure; null when there is no text. */
  detect(text: string, minConfidence = MIN_CONFIDENCE): string | null {
    const found = this.languageAndScore(text);
    if (!found) return null;
    return found[1] >= minConfidence ? found[0] : "und";
  }

  /**
   * Runs of one language, in order: each long-enough sentence detected on its own, a short one
   * takes a neighbour's language (else the whole text's), consecutive sentences of one language
   * merged.
   */
  detectSegments(text: string, minConfidence = 0): LanguageSegment[] {
    if (!text.trim()) return [];
    const code = (t: string) => this.detect(t, minConfidence) ?? "und";
    const sentences = sentencesWithOffsets(text);
    if (sentences.length === 0) return [{ text, lang: code(text), start: 0, end: text.length }];
    let labels = sentences.map(([s]) => (s.trim().length >= MIN_SEGMENT_CHARS ? code(s) : ""));
    labels = labels.map((label, i) => {
      if (label && label !== "und") return label;
      const left = i > 0 ? labels[i - 1]! : "";
      const right = i + 1 < labels.length ? labels[i + 1]! : "";
      return [left, right].find((n) => n && n !== "und") ?? "und";
    });
    if (labels.includes("und")) {
      const whole = code(text);
      labels = labels.map((l) => (l === "und" ? whole : l));
    }
    const segments: LanguageSegment[] = [];
    sentences.forEach(([, start, end], i) => {
      const label = labels[i]!;
      const last = segments[segments.length - 1];
      if (last && last.lang === label) segments[segments.length - 1] = { text: text.slice(last.start, end), lang: label, start: last.start, end };
      else segments.push({ text: text.slice(start, end), lang: label, start, end });
    });
    return segments;
  }
}

function sentencesWithOffsets(text: string): [string, number, number][] {
  const spans: [string, number, number][] = [];
  let start = 0;
  for (const match of text.matchAll(/(?<=[.!?…。！？])\s+/g)) {
    if (text.slice(start, match.index).trim()) spans.push([text.slice(start, match.index), start, match.index]);
    start = match.index + match[0].length;
  }
  if (text.slice(start).trim()) spans.push([text.slice(start), start, text.length]);
  return spans;
}
