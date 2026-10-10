/**
 * A SentencePiece model (`.spm` / `.model`) turned into an equivalent `tokenizer.json`, as
 * transformers' `SpmConverter` does it:
 */

import { ConfigurationError } from "../errors.js";
import { Tokenizer } from "./tokenizer.js";

interface Piece {
  piece: string;
  score: number;
  /** 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte. */
  type: number;
}

export interface SentencePieceModel {
  pieces: Piece[];
  /** 1 unigram, 2 BPE, 3 word, 4 char. */
  modelType: number;
  unkId: number;
  byteFallback: boolean;
  precompiledCharsmap: Uint8Array;
  addDummyPrefix: boolean;
  removeExtraWhitespaces: boolean;
  /** The trainer's special ids; -1 when the model has none. */
  bosId: number;
  eosId: number;
  padId: number;
}

class Proto {
  offset = 0;
  constructor(readonly bytes: Uint8Array, readonly end = bytes.length) {}
  get done(): boolean {
    return this.offset >= this.end;
  }
  /** A varint read as a proto int32: its low 32 bits, signed (-1 travels as ten bytes). */
  int32(): number {
    let result = 0;
    let shift = 0;
    for (;;) {
      const byte = this.bytes[this.offset++]!;
      if (shift < 32) result |= (byte & 0x7f) << shift;
      if (byte < 0x80) return result | 0;
      shift += 7;
    }
  }
  varint(): number {
    let result = 0;
    let shift = 0;
    for (;;) {
      const byte = this.bytes[this.offset++]!;
      result += (byte & 0x7f) * 2 ** shift;
      if (byte < 0x80) return result;
      shift += 7;
    }
  }
  key(): [field: number, wire: number] {
    const k = this.varint();
    return [Math.floor(k / 8), k % 8];
  }
  sub(): Proto {
    const length = this.varint();
    const p = new Proto(this.bytes, this.offset + length);
    p.offset = this.offset;
    this.offset += length;
    return p;
  }
  bytesField(): Uint8Array {
    const length = this.varint();
    const out = this.bytes.subarray(this.offset, this.offset + length);
    this.offset += length;
    return out;
  }
  float32(): number {
    const v = new DataView(this.bytes.buffer, this.bytes.byteOffset + this.offset, 4).getFloat32(0, true);
    this.offset += 4;
    return v;
  }
  skip(wire: number): void {
    if (wire === 0) this.varint();
    else if (wire === 1) this.offset += 8;
    else if (wire === 2) {
      const length = this.varint();
      this.offset += length;
    }
    else if (wire === 5) this.offset += 4;
    else throw new ConfigurationError(`unsupported protobuf wire type ${wire} in a SentencePiece model`);
  }
}

const decoder = new TextDecoder();

/** The fields of a SentencePiece `ModelProto` a tokenizer needs. */
export function readSentencePiece(bytes: Uint8Array): SentencePieceModel {
  const model: SentencePieceModel = { pieces: [], modelType: 1, unkId: 0, bosId: 1, eosId: 2, padId: -1, byteFallback: false, precompiledCharsmap: new Uint8Array(0), addDummyPrefix: true, removeExtraWhitespaces: true };

  const names = { unk: "<unk>", bos: "<s>", eos: "</s>", pad: "<pad>" };
  const root = new Proto(bytes);
  while (!root.done) {
    const [field, wire] = root.key();
    if (field === 1 && wire === 2) {
      const p = root.sub();
      const piece: Piece = { piece: "", score: 0, type: 1 };
      while (!p.done) {
        const [f, w] = p.key();
        if (f === 1 && w === 2) piece.piece = decoder.decode(p.bytesField());
        else if (f === 2 && w === 5) piece.score = p.float32();
        else if (f === 3 && w === 0) piece.type = p.varint();
        else p.skip(w);
      }
      model.pieces.push(piece);
    } else if (field === 2 && wire === 2) {
      const t = root.sub();
      while (!t.done) {
        const [f, w] = t.key();
        if (f === 3 && w === 0) model.modelType = t.varint();
        else if (f === 35 && w === 0) model.byteFallback = t.varint() !== 0;
        else if (f === 40 && w === 0) model.unkId = t.int32();
        else if (f === 45 && w === 2) names.unk = decoder.decode(t.bytesField());
        else if (f === 46 && w === 2) names.bos = decoder.decode(t.bytesField());
        else if (f === 47 && w === 2) names.eos = decoder.decode(t.bytesField());
        else if (f === 48 && w === 2) names.pad = decoder.decode(t.bytesField());
        else t.skip(w);
      }
    } else if (field === 3 && wire === 2) {
      const n = root.sub();
      while (!n.done) {
        const [f, w] = n.key();
        if (f === 2 && w === 2) model.precompiledCharsmap = n.bytesField().slice();
        else if (f === 3 && w === 0) model.addDummyPrefix = n.varint() !== 0;
        else if (f === 4 && w === 0) model.removeExtraWhitespaces = n.varint() !== 0;
        else n.skip(w);
      }
    } else {
      root.skip(wire);
    }
  }
  if (model.pieces.length === 0) throw new ConfigurationError("this file holds no SentencePiece pieces");
  const control = (name: string) => model.pieces.findIndex((p) => p.piece === name && p.type === 3);
  const unknown = model.pieces.findIndex((p) => p.piece === names.unk && p.type === 2);
  if (unknown >= 0) model.unkId = unknown;
  model.bosId = control(names.bos);
  model.eosId = control(names.eos);
  model.padId = control(names.pad);
  return model;
}

function base64(bytes: Uint8Array): string {
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary);
}

/** The `tokenizer.json` transformers' SpmConverter would write for this model. */
export function sentencePieceToTokenizerJson(model: SentencePieceModel): string {
  const normalizers: unknown[] = [
    { type: "Strip", strip_left: false, strip_right: true },
    { type: "Replace", pattern: { Regex: " {2,}" }, content: "▁" },
  ];
  if (model.precompiledCharsmap.length > 0) normalizers.unshift({ type: "Precompiled", precompiled_charsmap: base64(model.precompiledCharsmap) });
  const prepend = model.addDummyPrefix ? "always" : "never";
  const added = model.pieces
    .map((p, id) => ({ p, id }))
    .filter(({ p }) => p.type === 2 || p.type === 3)
    .map(({ p, id }) => ({ id, content: p.piece, single_word: false, lstrip: false, rstrip: false, normalized: false, special: true }));
  let tokenizerModel: unknown;
  if (model.modelType === 1) {
    tokenizerModel = { type: "Unigram", unk_id: model.unkId, vocab: model.pieces.map((p) => [p.piece, p.score]), byte_fallback: model.byteFallback };
  } else {
    throw new ConfigurationError(`SentencePiece model type ${model.modelType} is not supported: only unigram models convert without their merges`);
  }
  return JSON.stringify({
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: added,
    normalizer: { type: "Sequence", normalizers },
    pre_tokenizer: { type: "Metaspace", replacement: "▁", prepend_scheme: prepend, split: true },
    post_processor: null,
    decoder: { type: "Metaspace", replacement: "▁", prepend_scheme: prepend, split: true },
    model: tokenizerModel,
  });
}

/** A tokenizer that cuts text exactly as this SentencePiece model does. */
export function sentencePieceTokenizer(bytes: Uint8Array | ArrayBuffer): Tokenizer {
  return Tokenizer.fromJson(sentencePieceToTokenizerJson(readSentencePiece(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes))));
}

/**
 * Pieces as SentencePiece's `encode(text, out_type=str)` gives them: an unknown character comes
 * back as ITSELF, not as `<unk>` — what a caller mapping pieces onto another vocabulary (opus-mt's
 * shared one) needs.
 */
export class SentencePieceProcessor {
  readonly model: SentencePieceModel;
  readonly tokenizer: Tokenizer;

  constructor(bytes: Uint8Array | ArrayBuffer) {
    this.model = readSentencePiece(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
    this.tokenizer = Tokenizer.fromJson(sentencePieceToTokenizerJson(this.model));
  }

  /** The sequence opener (`<s>`), null when the model has none. */
  get bosId(): number | null {
    return this.model.bosId >= 0 ? this.model.bosId : null;
  }

  get eosId(): number | null {
    return this.model.eosId >= 0 ? this.model.eosId : null;
  }

  /** The padding id; 0 when the model names none (the attention mask hides it). */
  get padId(): number {
    return this.model.padId >= 0 ? this.model.padId : 0;
  }

  /** The ids in this model's own vocabulary. */
  encodeIds(text: string): number[] {
    return this.tokenizer.encode(text, { addSpecialTokens: false });
  }

  encodePieces(text: string): string[] {
    const encoded = this.tokenizer.encodeWithOffsets(text, { addSpecialTokens: false });
    return encoded.ids.map((id, k) => {
      if (id !== this.model.unkId) return this.model.pieces[id]!.piece;
      const [start, end] = encoded.offsets[k]!;
      return text.slice(start, end);
    });
  }
}
