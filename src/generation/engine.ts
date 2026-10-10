/**
 * The decode loop: an encoder once (when there is one), the decoder once per token, the KV cache
 * carried from one step to the next.
 */

import { GenerationError, CancelledError } from "../errors.js";
import type { Logger } from "../logger.js";
import type { ModelBundle, ModelSession } from "../session/bundle.js";
import type { GraphRole } from "../types/manifest.js";
import {
  BadWordsProcessor,
  EOSTokenCriteria,
  MaxLengthCriteria,
  MinLengthProcessor,
  MinPWarper,
  NoRepeatNGramProcessor,
  PresenceFrequencyPenaltyProcessor,
  RepetitionPenaltyProcessor,
  SuppressTokensProcessor,
  TemperatureWarper,
  TopKWarper,
  TopPWarper,
  applyProcessors,
  type Logits,
  type LogitsProcessor,
  type StoppingCriterion,
} from "./processors.js";
import {
  BeamSearch,
  GreedySearch,
  SamplingSearch,
  logSoftmax,
  type GenerationOutput,
  type SearchStrategy,
} from "./search.js";

/** The slice of an onnxruntime-web tensor this loop touches. */
export interface TensorLike {
  readonly data: unknown;
  readonly dims: readonly number[];
  readonly type: string;
  readonly location?: string;
  getData?(release?: boolean): Promise<unknown>;
  dispose?(): void;
}

export interface TensorFactory {
  new (type: string, data: unknown, dims: readonly number[]): TensorLike;
}

export interface GenerateOptions {
  /** One prompt per sequence; all of one length (bucket them first). */
  prompts: readonly (readonly number[])[];
  strategy?: "greedy" | "beam" | "sampling";
  beamSize?: number;
  bestOf?: number;
  patience?: number;
  lengthPenalty?: number;
  earlyStopping?: boolean;
  /** Above zero, sampling is used whatever the strategy. */
  temperature?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  repetitionPenalty?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  noRepeatNgramSize?: number;
  /** Tokens to generate at most, prompt excluded. */
  maxNewTokens?: number;
  minNewTokens?: number;
  /** The model's positional limit, prompt included (Whisper: 448). */
  maxPositions?: number;
  suppressTokens?: readonly number[];
  badWords?: readonly (readonly number[])[];
  eosTokenId?: number | readonly number[] | null;
  padTokenId?: number | null;
  seed?: number | null;
  /** Domain rules, applied before the warpers (a warper rescales what they shape). */
  processors?: readonly LogitsProcessor[];
  stopping?: readonly StoppingCriterion[];
  /** The encoder's output, or its inputs to run it here. */
  encoderOutput?: TensorLike;
  encoderFeeds?: Record<string, TensorLike>;
  /** The cross-attention mask of a padded text batch; all ones otherwise. */
  encoderAttentionMask?: BigInt64Array;
  /** Softmax probabilities of these ids on the prefill logits, before any processor. */
  probeFirstStep?: readonly number[];
  signal?: AbortSignal;
  /** Every step's chosen ids, one per row, as they are picked. */
  onStep?: (tokens: readonly number[], step: number) => void;
}

const INPUT_ID_NAMES = ["input_ids", "decoder_input_ids", "tokens"] as const;
const ENCODER_STATE_NAMES = ["encoder_hidden_states", "encoder_outputs"] as const;
const PAST = "past_key_values";
const PRESENT = "present";

function isCross(name: string): boolean {
  return name.includes(".encoder.") || name.endsWith(".encoder");
}

function float32Of(tensor: TensorLike): Float32Array {
  const data = tensor.data;
  if (data instanceof Float32Array) return data;
  if (tensor.type === "float16") {
    const half = data as Uint16Array | { length: number; [i: number]: number };
    const out = new Float32Array(half.length);

    const Float16 = (globalThis as { Float16Array?: new (...args: never[]) => unknown }).Float16Array;
    if (Float16 && data instanceof Float16) {
      out.set(data as unknown as ArrayLike<number>);
      return out;
    }
    for (let i = 0; i < half.length; i += 1) out[i] = halfToFloat((half as Uint16Array)[i]!);
    return out;
  }
  if (ArrayBuffer.isView(data)) return Float32Array.from(data as unknown as ArrayLike<number>);
  throw new GenerationError(`cannot read a ${tensor.type} tensor as logits`);
}

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** Bring a tensor to the CPU, whatever its location. */
async function onHost(tensor: TensorLike): Promise<TensorLike> {
  if (tensor.location && tensor.location !== "cpu" && tensor.getData) {
    const data = await tensor.getData();
    return { data, dims: tensor.dims, type: tensor.type };
  }
  return tensor;
}

type TypedArray = Float32Array | Uint16Array | BigInt64Array | Int32Array | Uint8Array;

function emptyOf(type: string): TypedArray {
  switch (type) {
    case "float16":
      return new Uint16Array(0);
    case "int64":
      return new BigInt64Array(0);
    case "int32":
      return new Int32Array(0);
    case "bool":
      return new Uint8Array(0);
    default:
      return new Float32Array(0);
  }
}

/** Gather rows of a [rows, ...] tensor on the CPU: `out[i] = in[order[i]]`. */
function gatherRows(Tensor: TensorFactory, tensor: TensorLike, order: readonly number[]): TensorLike {
  const data = tensor.data as TypedArray;
  const rows = tensor.dims[0] ?? 1;
  const stride = rows === 0 ? 0 : data.length / rows;
  const Ctor = data.constructor as new (length: number) => TypedArray;
  const out = new Ctor(order.length * stride);
  order.forEach((from, to) => {
    (out as Float32Array).set((data as Float32Array).subarray(from * stride, (from + 1) * stride), to * stride);
  });
  return new Tensor(tensor.type, out, [order.length, ...tensor.dims.slice(1)]);
}

/** Let the event loop run once. */
export function yieldToEventLoop(): Promise<void> {
  const immediate = (globalThis as { setImmediate?: (fn: () => void) => void }).setImmediate;
  if (typeof immediate === "function") return new Promise((resolve) => immediate(() => resolve()));
  if (typeof MessageChannel !== "undefined") {
    return new Promise((resolve) => {
      const channel = new MessageChannel();
      channel.port1.onmessage = () => {
        channel.port1.close();
        resolve();
      };
      channel.port2.postMessage(null);
    });
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export class GenerationEngine {
  readonly encoder: ModelSession | null;
  readonly prefill: ModelSession;
  readonly decode: ModelSession;
  readonly #pairs = new Map<string, string>();
  readonly #pastNames: string[];
  readonly #logger: Logger | undefined;

  constructor(
    readonly bundle: ModelBundle,
    readonly Tensor: TensorFactory,
    options: { logger?: Logger } = {},
  ) {
    this.#logger = options.logger;
    const roles = bundle.roles;
    const has = (role: GraphRole) => roles.includes(role);
    this.encoder = has("encoder") ? bundle.require("encoder") : null;
    let prefillRole: GraphRole;
    let decodeRole: GraphRole;
    if (has("decoder_merged")) {
      prefillRole = decodeRole = "decoder_merged";
    } else if (has("decoder") || has("decoder_with_past")) {
      prefillRole = has("decoder") ? "decoder" : "decoder_with_past";
      decodeRole = has("decoder_with_past") ? "decoder_with_past" : prefillRole;
    } else if (has("single")) {
      prefillRole = decodeRole = "single";
    } else {
      throw new GenerationError(
        `'${bundle.manifest.modelKey}' has no decoder to step (roles: ${roles.join(", ") || "none"})`,
      );
    }
    this.prefill = bundle.require(prefillRole);
    this.decode = bundle.require(decodeRole);

    this.#pastNames = this.decode.inputs.map((s) => s.name).filter((n) => n.startsWith(PAST));
    const hasBranch = this.decode.inputs.some((s) => s.name === "use_cache_branch");
    if (this.#pastNames.length === 0 && !hasBranch) {
      throw new GenerationError(
        `the ${decodeRole} graph of '${bundle.manifest.modelKey}' has neither past_key_values.* inputs ` +
          `nor use_cache_branch: it cannot be driven step by step. Re-export it with use_cache=True.`,
      );
    }

    const bySuffix = new Map(this.#pastNames.map((name) => [name.slice(PAST.length), name]));
    for (const spec of [...this.prefill.outputs, ...this.decode.outputs]) {
      if (!spec.name.startsWith(PRESENT)) continue;
      const past = bySuffix.get(spec.name.slice(PRESENT.length));
      if (past) this.#pairs.set(spec.name, past);
    }
  }

  get isEncoderDecoder(): boolean {
    return this.encoder !== null;
  }

  /** Run the encoder; its main output. */
  async encode(feeds: Record<string, TensorLike>): Promise<TensorLike> {
    if (!this.encoder) throw new GenerationError(`'${this.bundle.manifest.modelKey}' has no encoder`);
    const produced = await this.encoder.run(feeds as Record<string, unknown>);
    const name =
      this.encoder.outputs.find((s) => s.name === "last_hidden_state" || s.name.includes("hidden"))?.name ??
      this.encoder.outputs[0]?.name;
    const value = name ? (produced[name] as TensorLike | undefined) : undefined;
    if (!value) throw new GenerationError("the encoder produced no output");
    return value;
  }

  #strategy(options: GenerateOptions, maxLength: number): SearchStrategy {
    const eos = options.eosTokenId ?? null;
    const firstEos = typeof eos === "number" ? eos : (eos?.[0] ?? 0);
    const pad = options.padTokenId ?? firstEos;
    const common = { eosTokenId: eos, padTokenId: pad, maxLength };
    let strategy = options.strategy ?? "greedy";

    if ((options.temperature ?? 0) > 0) strategy = "sampling";
    if (strategy === "sampling") {
      return new SamplingSearch({ ...common, bestOf: options.bestOf ?? 1, seed: options.seed ?? null });
    }
    if (strategy === "beam" && (options.beamSize ?? 1) > 1) {
      return new BeamSearch({
        ...common,
        beamSize: options.beamSize!,
        lengthPenalty: options.lengthPenalty ?? 1,
        patience: options.patience ?? 1,
        earlyStopping: options.earlyStopping ?? true,
      });
    }
    return new GreedySearch(common);
  }

  #processors(options: GenerateOptions): LogitsProcessor[] {
    const chain: LogitsProcessor[] = [];
    const eos = options.eosTokenId;
    if (options.suppressTokens?.length) chain.push(new SuppressTokensProcessor(options.suppressTokens));
    if ((options.minNewTokens ?? 0) > 0 && eos !== null && eos !== undefined) {
      chain.push(new MinLengthProcessor(options.minNewTokens!, eos));
    }
    if ((options.repetitionPenalty ?? 1) !== 1) chain.push(new RepetitionPenaltyProcessor(options.repetitionPenalty));
    if (options.presencePenalty || options.frequencyPenalty) {
      chain.push(new PresenceFrequencyPenaltyProcessor(options.presencePenalty ?? 0, options.frequencyPenalty ?? 0));
    }
    if ((options.noRepeatNgramSize ?? 0) > 0) chain.push(new NoRepeatNGramProcessor(options.noRepeatNgramSize));
    if (options.badWords?.length) chain.push(new BadWordsProcessor(options.badWords));
    chain.push(...(options.processors ?? []));
    if ((options.temperature ?? 0) > 0) chain.push(new TemperatureWarper(options.temperature));
    if ((options.topK ?? 0) > 0) chain.push(new TopKWarper(options.topK));
    if ((options.topP ?? 1) < 1) chain.push(new TopPWarper(options.topP));
    if ((options.minP ?? 0) > 0) chain.push(new MinPWarper(options.minP));
    return chain;
  }

  async generate(options: GenerateOptions): Promise<GenerationOutput> {
    const { Tensor } = this;
    const prompts = options.prompts;
    if (prompts.length === 0) {
      return { sequences: [], scores: [], sumLogprob: [], avgLogprob: [], finishReasons: [], firstStepProbs: {}, firstStepProbsPerRow: [], steps: 0, generatedTokens: 0, prefillMs: 0, decodeMs: 0, temperature: 0 };
    }
    const promptLength = prompts[0]!.length;
    if (prompts.some((p) => p.length !== promptLength)) {
      throw new GenerationError(
        `prompts have mixed lengths ${[...new Set(prompts.map((p) => p.length))].join(", ")}; ` +
          `bucket them by length before generating`,
      );
    }
    let encoderOutput = options.encoderOutput ?? null;
    if (!encoderOutput && options.encoderFeeds) encoderOutput = await this.encode(options.encoderFeeds);
    if (this.encoder && !encoderOutput) {
      throw new GenerationError("an encoder-decoder model needs encoderOutput or encoderFeeds");
    }

    const budget = options.maxPositions ? Math.max(1, options.maxPositions - promptLength) : Number.POSITIVE_INFINITY;
    const maxLength = Math.min(options.maxNewTokens ?? 448, budget);
    const batch = prompts.length;
    const strategy = this.#strategy(options, maxLength);
    const per = strategy.rowsPerSequence;
    const processors = this.#processors(options);
    const stopping: StoppingCriterion[] = [new MaxLengthCriteria(maxLength)];
    if (options.eosTokenId !== null && options.eosTokenId !== undefined) {
      stopping.push(new EOSTokenCriteria(options.eosTokenId));
    }
    stopping.push(...(options.stopping ?? []));
    for (const criterion of stopping) criterion.reset?.();

    const self = new Map<string, TensorLike>();
    const cross = new Map<string, TensorLike>();
    const owned = new Set<TensorLike>();
    const release = (tensor: TensorLike | undefined): void => {
      if (tensor && owned.delete(tensor)) tensor.dispose?.();
    };
    const capture = (produced: Record<string, unknown>): void => {
      for (const [present, past] of this.#pairs) {
        const value = produced[present] as TensorLike | undefined;
        if (!value) continue;
        owned.add(value);
        if (isCross(past)) {
          if (!cross.has(past)) cross.set(past, value);
          else release(value);
        } else {
          release(self.get(past));
          self.set(past, value);
        }
      }

      for (const [name, value] of Object.entries(produced)) {
        const tensor = value as TensorLike;
        if (name.startsWith(PRESENT) && !this.#pairs.has(name)) tensor.dispose?.();
      }
    };

    let encoderMask = options.encoderAttentionMask ?? null;
    const encoderLength = encoderOutput?.dims[1] ?? 0;
    if (!encoderMask && encoderLength > 0) encoderMask = new BigInt64Array(batch * encoderLength).fill(1n);

    const prefillNames = new Set(this.prefill.inputs.map((s) => s.name));
    const decodeNames = new Set(this.decode.inputs.map((s) => s.name));
    let pastLength = 0;

    const feedCommon = (names: ReadonlySet<string>, ids: BigInt64Array, rows: number, length: number, states: TensorLike | null) => {
      const feeds: Record<string, TensorLike> = {};
      const idName = INPUT_ID_NAMES.find((n) => names.has(n));
      if (idName) feeds[idName] = new Tensor("int64", ids, [rows, length]);
      const stateName = ENCODER_STATE_NAMES.find((n) => names.has(n));
      if (stateName && states) feeds[stateName] = states;
      if (names.has("attention_mask")) {
        feeds["attention_mask"] = new Tensor("int64", new BigInt64Array(rows * (pastLength + length)).fill(1n), [rows, pastLength + length]);
      }
      if (names.has("position_ids")) {
        const positions = new BigInt64Array(rows * length);
        for (let r = 0; r < rows; r += 1) for (let i = 0; i < length; i += 1) positions[r * length + i] = BigInt(pastLength + i);
        feeds["position_ids"] = new Tensor("int64", positions, [rows, length]);
      }
      if (names.has("encoder_attention_mask") && encoderMask) {
        const factor = rows / batch;
        const mask = factor === 1 ? encoderMask : new BigInt64Array(rows * encoderLength).map((_, i) => encoderMask![Math.floor(Math.floor(i / encoderLength) / factor) * encoderLength + (i % encoderLength)]!);
        feeds["encoder_attention_mask"] = new Tensor("int64", mask, [rows, encoderLength]);
      }
      return feeds;
    };

    const checkCancelled = (): void => {
      if (options.signal?.aborted) throw new CancelledError("generation was cancelled");
    };

    const logitsOf = async (session: ModelSession, produced: Record<string, unknown>): Promise<Logits> => {
      const name = session.outputs.find((s) => s.name === "logits")?.name ?? session.outputs.find((s) => !s.name.startsWith(PRESENT))?.name;
      const tensor = name ? (produced[name] as TensorLike | undefined) : undefined;
      if (!tensor) throw new GenerationError(`the decoder produced no logits (outputs: ${session.outputs.map((s) => s.name).join(", ")})`);
      const host = await onHost(tensor);
      const data = float32Of(host);
      const dims = tensor.dims;
      const vocab = dims[dims.length - 1]!;
      const positions = dims.length >= 3 ? dims[dims.length - 2]! : 1;
      const rows = data.length / (positions * vocab);

      const out = new Float32Array(rows * vocab);
      for (let r = 0; r < rows; r += 1) {
        const end = (r + 1) * positions * vocab;
        out.set(data.subarray(end - vocab, end), r * vocab);
      }
      tensor.dispose?.();
      return { data: out, rows, vocab };
    };

    checkCancelled();
    const started = performance.now();
    const ids = new BigInt64Array(batch * promptLength);
    prompts.forEach((prompt, r) => prompt.forEach((id, i) => (ids[r * promptLength + i] = BigInt(id))));
    const prefillFeeds = feedCommon(prefillNames, ids, batch, promptLength, encoderOutput);
    if (prefillNames.has("use_cache_branch")) prefillFeeds["use_cache_branch"] = new Tensor("bool", new Uint8Array([0]), [1]);
    for (const spec of this.prefill.inputs) {
      if (!spec.name.startsWith(PAST) || prefillFeeds[spec.name]) continue;

      const kv = this.bundle.manifest.kv;
      const dims = (spec.dims ?? []).map((d, axis) => {
        if (typeof d === "number" && d >= 0) return d;
        if (axis === 0) return batch;
        if (axis === 1) return kv?.numHeads ?? 1;
        if (axis === 3) return kv?.headDim ?? 1;
        return 0;
      });
      if (dims.length !== 4) dims.splice(0, dims.length, batch, kv?.numHeads ?? 1, 0, kv?.headDim ?? 1);
      dims[2] = 0;
      const type = spec.type?.replace(/^tensor\(|\)$/g, "") ?? kv?.dtype ?? "float32";
      prefillFeeds[spec.name] = new Tensor(type === "float" ? "float32" : type, emptyOf(type === "float" ? "float32" : type), dims);
    }
    const missing = [...prefillNames].filter((n) => !(n in prefillFeeds));
    if (missing.length > 0) throw new GenerationError(`the prefill graph requires ${missing.join(", ")}, which nothing supplies`);
    let produced = await this.prefill.run(prefillFeeds as Record<string, unknown>);
    let logits = await logitsOf(this.prefill, produced);
    capture(produced);
    pastLength = promptLength;
    const prefillMs = performance.now() - started;

    const firstStepProbsPerRow: Record<number, number>[] = [];
    if (options.probeFirstStep?.length) {
      for (let r = 0; r < logits.rows; r += 1) {
        const probs = logSoftmax(logits.data.subarray(r * logits.vocab, (r + 1) * logits.vocab));
        const row: Record<number, number> = {};
        for (const id of options.probeFirstStep) row[id] = Math.exp(probs[id] ?? Number.NEGATIVE_INFINITY);
        firstStepProbsPerRow.push(row);
      }
    }
    const firstStepProbs = firstStepProbsPerRow[0] ?? {};

    let states = encoderOutput;
    if (per > 1) {
      const order = Array.from({ length: batch * per }, (_, i) => Math.floor(i / per));
      for (const [name, value] of [...self, ...cross]) {
        const tiled = gatherRows(Tensor, await onHost(value), order);
        release(value);
        (isCross(name) ? cross : self).set(name, tiled);
      }
      const expanded = new Float32Array(batch * per * logits.vocab);
      order.forEach((from, to) => expanded.set(logits.data.subarray(from * logits.vocab, (from + 1) * logits.vocab), to * logits.vocab));
      logits = { data: expanded, rows: batch * per, vocab: logits.vocab };
      if (states) states = gatherRows(Tensor, await onHost(states), order);
    }

    strategy.init(batch);
    let steps = 0;
    let emitted = 0;
    const decodeStarted = performance.now();
    try {
      for (;;) {
        await yieldToEventLoop();
        checkCancelled();
        applyProcessors(processors, logits, strategy.sequences);
        const decision = strategy.step(logits);
        steps += 1;
        emitted += decision.finished.filter((done) => !done).length;
        options.onStep?.(decision.nextTokens, steps);

        const stopRows = new Array<boolean>(strategy.sequences.length).fill(false);
        for (const criterion of stopping) {
          criterion.check(strategy.sequences, steps).forEach((stop, r) => (stopRows[r] ||= stop));
        }
        if (stopRows.some(Boolean)) strategy.markStopped(stopRows, "constraint");
        if (decision.allDone || stopRows.every(Boolean)) break;
        if ((strategy.sequences[0]?.length ?? 0) >= maxLength) break;

        if (decision.beamIndices) {
          for (const [name, value] of self) {
            const reordered = gatherRows(Tensor, await onHost(value), decision.beamIndices);
            release(value);
            self.set(name, reordered);
          }
        }

        const rows = decision.nextTokens.length;
        const next = BigInt64Array.from(decision.nextTokens, (t) => BigInt(t));
        const feeds = feedCommon(decodeNames, next, rows, 1, states);
        if (decodeNames.has("use_cache_branch")) feeds["use_cache_branch"] = new Tensor("bool", new Uint8Array([1]), [1]);
        for (const [name, value] of [...self, ...cross]) if (decodeNames.has(name)) feeds[name] = value;
        produced = await this.decode.run(feeds as Record<string, unknown>);
        logits = await logitsOf(this.decode, produced);
        capture(produced);
        pastLength += 1;
      }
    } finally {
      for (const tensor of owned) tensor.dispose?.();
      owned.clear();
    }

    const output = strategy.finalize();
    output.steps = steps;
    output.generatedTokens = emitted;
    output.prefillMs = prefillMs;
    output.decodeMs = performance.now() - decodeStarted;
    output.temperature = options.temperature ?? 0;
    output.firstStepProbs = firstStepProbs;
    output.firstStepProbsPerRow = firstStepProbsPerRow;
    this.#logger?.debug(`generated ${emitted} tokens in ${steps} steps`, { prefillMs, decodeMs: output.decodeMs });
    return output;
  }
}
