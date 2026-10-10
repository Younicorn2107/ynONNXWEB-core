/** Encoding, decoding, padding and special-token lookup over one vocabulary. */

import { Tokenizer as HFTokenizer } from "@huggingface/tokenizers";

import { TokenizerError } from "../errors.js";
import type { Logger } from "../logger.js";

export interface TokenizerOptions {
  /** Add the model's special tokens (`[CLS]`, `</s>`…). Default true. */
  addSpecialTokens?: boolean;
  /** Cut every encoding to this many ids. */
  truncationMaxLength?: number | null;
  padTokenId?: number | null;
  paddingSide?: "left" | "right";
}

export interface Encoding {
  ids: number[];
  tokens: string[];
  /** `[start, end)` per token, UTF-16 indices; `[0, 0]` for added specials. */
  offsets: [number, number][];
}

export interface PaddedBatch {
  ids: BigInt64Array;
  attentionMask: BigInt64Array;
  dims: [number, number];
}

type Json = Record<string, unknown>;

function asObject(value: unknown): Json | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Json) : null;
}

/** Every node of a component tree (`Sequence` nests), depth first. */
function nodes(value: unknown): Json[] {
  const node = asObject(value);
  if (!node) return [];
  const out = [node];
  for (const key of ["normalizers", "pretokenizers", "decoders", "processors"]) {
    const list = node[key];
    if (Array.isArray(list)) for (const child of list) out.push(...nodes(child));
  }
  return out;
}

/** How tokens spell the text they stand for. */
interface Spelling {
  byteLevel: boolean;
  /** `##` for WordPiece. */
  continuation: string | null;
  /** `▁` for Metaspace. */
  spaceMarker: string | null;
  /**
   * Whether the marker stands for a space the pre-tokenizer already removed (T5: `WhitespaceSplit`
   * then `Metaspace`), rather than for a character of the text (XLM-R, NLLB: the space, or what the
   * normalizer turned into one).
   */
  virtualMarker: boolean;
  /** SentencePiece `<0xAB>` tokens. */
  byteFallback: boolean;
  unk: string | null;
}

function spellingOf(json: Json): Spelling {
  const model = asObject(json["model"]) ?? {};
  const parts = [...nodes(json["pre_tokenizer"]), ...nodes(json["decoder"])];
  const byteLevel = parts.some((n) => n["type"] === "ByteLevel");
  const meta = parts.find((n) => n["type"] === "Metaspace");
  const replace = parts.find(
    (n) => n["type"] === "Replace" && asObject(n["pattern"])?.["String"] === " ",
  );
  const marker =
    typeof meta?.["replacement"] === "string"
      ? (meta["replacement"] as string)
      : typeof replace?.["content"] === "string"
        ? (replace["content"] as string)
        : null;

  const wordPiece = parts.find((n) => n["type"] === "WordPiece");
  const isWordPiece =
    model["type"] === "WordPiece" ||
    wordPiece !== undefined ||
    (model["type"] === undefined && typeof model["continuing_subword_prefix"] === "string");
  return {
    byteLevel,
    continuation: isWordPiece
      ? ((wordPiece?.["prefix"] as string | undefined) ??
        (model["continuing_subword_prefix"] as string | undefined) ??
        "##")
      : null,
    spaceMarker: byteLevel ? null : marker,
    virtualMarker: parts.some((n) => n["type"] === "WhitespaceSplit" || n["type"] === "Whitespace"),
    byteFallback: model["byte_fallback"] === true,
    unk: typeof model["unk_token"] === "string" ? (model["unk_token"] as string) : null,
  };
}

/** GPT-2's reversible byte ↔ printable-character table. */
const BYTE_OF: ReadonlyMap<string, number> = (() => {
  const bytes: number[] = [];
  for (let b = 0x21; b <= 0x7e; b += 1) bytes.push(b);
  for (let b = 0xa1; b <= 0xac; b += 1) bytes.push(b);
  for (let b = 0xae; b <= 0xff; b += 1) bytes.push(b);
  const chars = [...bytes];
  let extra = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      chars.push(256 + extra);
      extra += 1;
    }
  }
  return new Map(bytes.map((byte, index) => [String.fromCodePoint(chars[index]!), byte]));
})();

/** What a character becomes for matching: no case, no accents, compatibility forms. */
function fold(text: string): string {
  return text.normalize("NFKD").replace(/[\p{M}\p{Cf}]/gu, "").toLowerCase();
}

const WHITESPACE = /\s/u;
/** BERT's notion of punctuation: Unicode P*, and every ASCII symbol. */
const PUNCTUATION = /[\p{P}!-/:-@[-`{-~]/u;
const CJK = /[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]/u;

/** Align tokens with the text they were encoded from. */
export function alignOffsets(
  text: string,
  tokens: readonly string[],
  spelling: Spelling,
  added: ReadonlySet<string>,
): [number, number][] {
  return spelling.byteLevel
    ? alignBytes(text, tokens, added)
    : alignPieces(text, tokens, spelling, added);
}

function alignBytes(text: string, tokens: readonly string[], added: ReadonlySet<string>): [number, number][] {
  const encoder = new TextEncoder();
  const byteToChar: number[] = [];
  let index = 0;
  for (const char of text) {
    const width = encoder.encode(char).length;
    for (let i = 0; i < width; i += 1) byteToChar.push(index);
    index += char.length;
  }
  byteToChar.push(text.length);
  const charEnd = (byte: number): number => {
    if (byte <= 0) return 0;
    const start = byteToChar[byte - 1]!;
    const cp = text.codePointAt(start)!;
    return start + (cp > 0xffff ? 2 : 1);
  };

  const out: [number, number][] = [];
  const isSpaceByte = (value: number): boolean => value === 0x20 || value === 0x09 || value === 0x0a || value === 0x0d;
  let byte = 0;
  for (const token of tokens) {
    if (added.has(token)) {
      const at = byteToChar[byte] ?? text.length;
      if (text.startsWith(token, at)) {
        out.push([at, at + token.length]);
        byte += encoder.encode(token).length;
      } else {
        out.push([0, 0]);
      }
      continue;
    }
    const values: number[] = [];
    for (const char of token) {
      const value = BYTE_OF.get(char);
      if (value !== undefined) values.push(value);
    }
    const end = byte + values.length;

    let first = 0;
    while (first < values.length && isSpaceByte(values[first]!)) first += 1;
    if (first === values.length) {
      out.push([charEnd(end), charEnd(end)]);
    } else {
      out.push([byteToChar[byte + first] ?? text.length, charEnd(end)]);
    }
    byte = end;
  }
  return out;
}

function alignPieces(
  text: string,
  tokens: readonly string[],
  spelling: Spelling,
  added: ReadonlySet<string>,
): [number, number][] {
  const out: [number, number][] = [];
  let cursor = 0;
  let pendingBytes: number[] = [];
  let pendingStart = 0;
  const pendingIndices: number[] = [];

  const skipSpace = (): void => {
    while (cursor < text.length && WHITESPACE.test(text[cursor]!)) cursor += 1;
  };
  const nextChar = (at: number): string => String.fromCodePoint(text.codePointAt(at) ?? 0);

  for (const raw of tokens) {
    if (added.has(raw) && raw !== spelling.unk) {
      const probe = text.indexOf(raw, cursor);
      const between = probe >= 0 ? text.slice(cursor, probe) : "";
      if (probe >= 0 && between.trim() === "") {
        out.push([probe, probe + raw.length]);
        cursor = probe + raw.length;
      } else {
        out.push([0, 0]);
      }
      continue;
    }

    const fallback = spelling.byteFallback ? /^<0x([0-9A-Fa-f]{2})>$/.exec(raw) : null;
    if (fallback) {
      if (pendingBytes.length === 0) {
        skipSpace();
        pendingStart = cursor;
      }
      pendingBytes.push(parseInt(fallback[1]!, 16));
      pendingIndices.push(out.length);
      out.push([pendingStart, pendingStart]);
      const decoded = new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(pendingBytes));
      if (!decoded.includes("�")) {
        const width = nextChar(pendingStart).length;
        for (const i of pendingIndices) out[i] = [pendingStart, pendingStart + width];
        cursor = pendingStart + width;
        pendingBytes = [];
        pendingIndices.length = 0;
      }
      continue;
    }

    let piece = raw;
    if (spelling.continuation && piece.startsWith(spelling.continuation) && piece !== spelling.continuation) {
      piece = piece.slice(spelling.continuation.length);
    } else if (spelling.continuation) {
      skipSpace();
    }
    if (spelling.spaceMarker) piece = piece.split(spelling.spaceMarker).join(" ");
    const visible = piece.replace(/^\s+/u, "");

    if (raw === spelling.unk) {
      skipSpace();
      const start = cursor;
      if (cursor < text.length) {
        const first = nextChar(cursor);
        cursor += first.length;
        if (spelling.continuation && !CJK.test(first) && !PUNCTUATION.test(first)) {
          while (cursor < text.length) {
            const char = nextChar(cursor);
            if (WHITESPACE.test(char) || PUNCTUATION.test(char) || CJK.test(char)) break;
            cursor += char.length;
          }
        }
      }
      out.push([start, cursor]);
      continue;
    }

    if (visible.length === 0) {
      if (spelling.virtualMarker) {
        while (cursor < text.length && fold(nextChar(cursor)).trim().length === 0) {
          cursor += nextChar(cursor).length;
        }
        const width = cursor < text.length ? nextChar(cursor).length : 0;
        out.push([cursor, cursor + width]);
      } else if (cursor < text.length) {
        const char = nextChar(cursor);
        if (WHITESPACE.test(char)) {
          out.push([cursor + char.length, cursor + char.length]);
          cursor += char.length;
        } else if (fold(char).trim().length === 0) {
          out.push([cursor, cursor + char.length]);
          cursor += char.length;
        } else {
          out.push([cursor, cursor + char.length]);
        }
      } else {
        out.push([cursor, cursor]);
      }
      continue;
    }

    const target = fold(visible).replace(/\s+/gu, "");
    skipSpace();
    const from = cursor;
    let matched = 0;
    while (cursor < text.length && matched < target.length) {
      const char = nextChar(cursor);
      const folded = fold(char).replace(/\s+/gu, "");
      if (folded.length === 0) {
        cursor += char.length;
        continue;
      }
      if (!target.startsWith(folded, matched)) break;
      matched += folded.length;
      cursor += char.length;
    }
    out.push([from, cursor]);
  }
  return out;
}

export class Tokenizer {
  readonly #impl: HFTokenizer;
  readonly #json: Json;
  readonly #spelling: Spelling;
  readonly #added: Map<string, number>;
  readonly #specials: Set<number>;
  readonly #pieceCache = new Map<number, string>();
  readonly options: Required<Omit<TokenizerOptions, "truncationMaxLength" | "padTokenId">> &
    Pick<TokenizerOptions, "truncationMaxLength" | "padTokenId">;

  private constructor(json: Json, config: Json, options: TokenizerOptions, logger?: Logger) {
    this.#json = Tokenizer.#repaired(json, logger);
    try {
      this.#impl = new HFTokenizer(this.#json, config);
    } catch (cause) {
      throw new TokenizerError(
        `the tokenizer could not be read: ${cause instanceof Error ? cause.message : String(cause)}`,
        { cause },
      );
    }
    this.#spelling = spellingOf(this.#json);
    this.#added = new Map();
    this.#specials = new Set();
    for (const entry of (this.#json["added_tokens"] as Json[] | undefined) ?? []) {
      if (typeof entry["content"] === "string" && typeof entry["id"] === "number") {
        this.#added.set(entry["content"], entry["id"]);
        if (entry["special"] === true) this.#specials.add(entry["id"]);
      }
    }
    this.options = {
      addSpecialTokens: options.addSpecialTokens ?? true,
      paddingSide: options.paddingSide ?? "right",
      truncationMaxLength: options.truncationMaxLength ?? null,
      padTokenId: options.padTokenId ?? null,
    };
  }

  /**
   * From the text (or parsed object) of a `tokenizer.json`, and optionally its
   * `tokenizer_config.json`.
   */
  static fromJson(
    tokenizer: string | object,
    config?: string | object | null,
    options: TokenizerOptions = {},
    logger?: Logger,
  ): Tokenizer {
    const parse = (value: string | object | null | undefined, what: string): Json => {
      if (value === null || value === undefined) return {};
      if (typeof value !== "string") return value as Json;
      try {
        return JSON.parse(value) as Json;
      } catch (cause) {
        throw new TokenizerError(`${what} is not valid JSON`, { cause });
      }
    };
    return new Tokenizer(parse(tokenizer, "tokenizer.json"), parse(config, "tokenizer_config.json"), options, logger);
  }

  /** The same repairs the Python side makes before reading a file. */
  static #repaired(json: Json, logger?: Logger): Json {
    const added = json["added_tokens"];
    if (!Array.isArray(added)) return json;
    const sentinels = added.filter((e) => typeof (e as Json)["id"] === "number" && ((e as Json)["id"] as number) < 0);
    if (sentinels.length === 0) return json;
    logger?.debug(
      `dropped ${sentinels.length} placeholder(s) with no vocabulary row: ` +
        sentinels.map((e) => `${String((e as Json)["content"])} (id ${String((e as Json)["id"])})`).join(", "),
    );
    return { ...json, added_tokens: added.filter((e) => !sentinels.includes(e)) };
  }

  #specialsFlag(override: boolean | undefined): boolean {
    return override ?? this.options.addSpecialTokens;
  }

  #truncate(ids: number[]): number[] {
    const limit = this.options.truncationMaxLength;
    return limit ? ids.slice(0, limit) : ids;
  }

  encode(text: string, options: { addSpecialTokens?: boolean; pair?: string } = {}): number[] {
    return this.#truncate(
      this.#impl.encode(text, {
        add_special_tokens: this.#specialsFlag(options.addSpecialTokens),
        ...(options.pair !== undefined ? { text_pair: options.pair } : {}),
      }).ids,
    );
  }

  encodeBatch(texts: readonly string[], options: { addSpecialTokens?: boolean } = {}): number[][] {
    return texts.map((text) => this.encode(text, options));
  }

  /** The token type ids of a pair (`0` for the first text, `1` for the second). */
  encodePair(first: string, second: string, options: { addSpecialTokens?: boolean } = {}): { ids: number[]; typeIds: number[] } {
    const encoded = this.#impl.encode(first, {
      text_pair: second,
      add_special_tokens: this.#specialsFlag(options.addSpecialTokens),
      return_token_type_ids: true,
    });
    const ids = this.#truncate(encoded.ids);
    return { ids, typeIds: encoded.token_type_ids.slice(0, ids.length) };
  }

  /** Ids, tokens, and the visible span of each token in `text`. */
  encodeWithOffsets(text: string, options: { addSpecialTokens?: boolean } = {}): Encoding {
    const encoded = this.#impl.encode(text, { add_special_tokens: this.#specialsFlag(options.addSpecialTokens) });
    const offsets = alignOffsets(text, encoded.tokens, this.#spelling, new Set(this.#added.keys()));
    const ids = this.#truncate(encoded.ids);
    return { ids, tokens: encoded.tokens.slice(0, ids.length), offsets: offsets.slice(0, ids.length) };
  }

  decode(ids: readonly number[], options: { skipSpecialTokens?: boolean } = {}): string {
    if (ids.length === 0) return "";
    return this.#impl.decode([...ids], {
      skip_special_tokens: options.skipSpecialTokens ?? true,
      clean_up_tokenization_spaces: false,
    });
  }

  decodeBatch(rows: readonly (readonly number[])[], options: { skipSpecialTokens?: boolean } = {}): string[] {
    return rows.map((row) => this.decode(row, options));
  }

  /**
   * One text fragment per token, uncollapsed — to attach a timestamp or a probability to each
   * token, which joining first would make impossible.
   */
  decodeTokens(ids: readonly number[]): string[] {
    return ids.map((id) => {
      let piece = this.#pieceCache.get(id);
      if (piece === undefined) {
        piece = this.#impl.decode([id], { skip_special_tokens: false, clean_up_tokenization_spaces: false });
        this.#pieceCache.set(id, piece);
      }
      return piece;
    });
  }

  /** Pad to a rectangle: `input_ids` and `attention_mask`, int64. */
  padBatch(
    sequences: readonly (readonly number[])[],
    options: { padId?: number; side?: "left" | "right"; toMultipleOf?: number } = {},
  ): PaddedBatch {
    const side = options.side ?? this.options.paddingSide;
    const padId = BigInt(this.resolvePadId(options.padId));
    let width = sequences.reduce((max, seq) => Math.max(max, seq.length), 0);
    if (options.toMultipleOf && options.toMultipleOf > 1) {
      width = Math.ceil(width / options.toMultipleOf) * options.toMultipleOf;
    }
    const ids = new BigInt64Array(sequences.length * width).fill(padId);
    const mask = new BigInt64Array(sequences.length * width);
    sequences.forEach((seq, row) => {
      const offset = row * width + (side === "left" ? width - seq.length : 0);
      seq.forEach((id, i) => {
        ids[offset + i] = BigInt(id);
        mask[offset + i] = 1n;
      });
    });
    return { ids, attentionMask: mask, dims: [sequences.length, width] };
  }

  /** The pad id: explicit, configured, or the first conventional pad token present. */
  resolvePadId(padId?: number | null): number {
    if (padId !== undefined && padId !== null) return padId;
    if (this.options.padTokenId !== null && this.options.padTokenId !== undefined) return this.options.padTokenId;
    for (const name of ["<pad>", "[PAD]", "<|endoftext|>", "</s>", "[SEP]"]) {
      const found = this.tokenToId(name);
      if (found !== null) return found;
    }
    return 0;
  }

  get vocabSize(): number {
    const model = asObject(this.#json["model"]) ?? {};
    const vocab = model["vocab"];
    let size = 0;
    if (Array.isArray(vocab)) size = vocab.length;
    else if (asObject(vocab)) size = Object.keys(vocab as Json).length;

    let top = size - 1;
    for (const id of this.#added.values()) top = Math.max(top, id);
    const extra = [...this.#added.values()].filter((id) => id >= size).length;
    return Math.max(size + extra, top + 1);
  }

  tokenToId(token: string): number | null {
    return this.#added.get(token) ?? this.#impl.token_to_id(token) ?? null;
  }

  idToToken(id: number): string | null {
    return this.#impl.id_to_token(id) ?? null;
  }

  /** Every added token, as `{literal: id}`. */
  specialIds(): Record<string, number> {
    return Object.fromEntries(this.#added);
  }

  isSpecial(id: number): boolean {
    return this.#specials.has(id);
  }

  /**
   * Every id whose literal token matches a pattern — Whisper's 1501 timestamp tokens, without
   * hardcoding one of them.
   */
  idsMatching(pattern: RegExp): number[] {
    const out: number[] = [];
    const model = asObject(this.#json["model"]) ?? {};
    const vocab = model["vocab"];
    if (Array.isArray(vocab)) {
      vocab.forEach((entry, id) => {
        const token = Array.isArray(entry) ? entry[0] : entry;
        if (typeof token === "string" && pattern.test(token)) out.push(id);
      });
    } else if (asObject(vocab)) {
      for (const [token, id] of Object.entries(vocab as Record<string, number>)) {
        if (pattern.test(token)) out.push(id);
      }
    }
    for (const [token, id] of this.#added) if (pattern.test(token) && !out.includes(id)) out.push(id);
    return out.sort((a, b) => a - b);
  }

  /** Refuse a vocabulary wider than the logits it is meant to read. */
  validateAgainst(logitsVocabSize: number | null, options: { strict?: boolean; modelKey?: string; logger?: Logger } = {}): void {
    if (logitsVocabSize === null) return;
    const actual = this.vocabSize;
    if (actual <= logitsVocabSize) return;
    const message =
      `tokenizer vocabulary (${actual}) exceeds the logits width of ` +
      `'${options.modelKey ?? "the model"}' (${logitsVocabSize}). The tokenizer does not belong ` +
      `to this graph; decoding would silently shift every special token.`;
    if (options.strict === false) {
      options.logger?.warn(`${message} Accepted: this caller resolves its special tokens by name.`);
      return;
    }
    throw new TokenizerError(message);
  }
}
