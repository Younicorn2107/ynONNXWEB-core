/** A model as its manifest describes it: one session per graph, addressed by role. */

import { InferenceError, ManifestError, ModelNotFoundError } from "../errors.js";
import type { Logger } from "../logger.js";
import { checkAdapterFit, checkWasmFit } from "../backends/memory.js";
import type { AdapterLimits } from "../backends/capabilities.js";
import { BACKEND_LABELS } from "../backends/chain.js";
import { replaceArgMax, replaceIsNaN } from "../models/argmaxHead.js";
import { externalizeWeights } from "../models/externalize.js";
import { gatherBeforeDequantize } from "../models/gatherDequantize.js";
import { CPU_NBITS_ACCURACY, setNBitsAccuracy } from "../models/nbitsAccuracy.js";
import { splitMergedDecoder } from "../models/splitMerged.js";
import { castGatherIndices } from "../models/gatherIndices.js";
import { exclusive } from "./exclusive.js";
import { padMatMulNBitsRows } from "../models/padNBits.js";
import { WEBGPU_NBITS_BLOCK, repackMatMulNBits } from "../models/repackNBits.js";
import { readGraphInfo } from "../models/onnxProto.js";
import type { BackendPlan } from "../backends/select.js";
import type { ModelKey } from "../models/resolve.js";
import {
  decoderRoles as manifestDecoderRoles,
  totalSizeMb,
  type GraphEntry,
  type GraphManifest,
  type GraphRole,
} from "../types/manifest.js";
import { buildSession, readSpecs, type OrtLike, type SessionConfig, type TensorSpec } from "./create.js";
import type { Backend } from "../backends/chain.js";

export interface ModelSession {
  readonly role: GraphRole;
  readonly filename: string;
  readonly backend: Backend;
  readonly inputs: TensorSpec[];
  readonly outputs: TensorSpec[];
  /** The graph's `metadata_props` (exporter notes: class names, task…); empty when none. */
  readonly metadata: Readonly<Record<string, string>>;
  readonly lastRunMs: number;
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
  close(): Promise<void>;
}

export interface BundleOptions {
  ort: OrtLike;
  manifest: GraphManifest;
  /** The key the files were actually fetched with. */
  key: ModelKey;
  /** Graph bytes by filename — already fetched, already in the store. */
  files: Map<string, ArrayBuffer>;
  plan: BackendPlan;
  /** A plan per role, for a model whose encoder and decoder run on different backends. */
  planFor?: Partial<Record<GraphRole, BackendPlan>>;
  /** The adapter's buffer limits, to refuse a weight WebGPU cannot bind. */
  adapterLimits?: AdapterLimits | null;
  config?: SessionConfig;
  logger?: Logger;
  wasmPaths?: string | Record<string, string>;
  /**
   * A run that has not answered after this long is given up, with an error that says so. A device
   * lost under WebGPU leaves a promise that never settles, and nothing else would ever end the
   * wait. Default 5 minutes.
   */
  runTimeoutMs?: number;
  /**
   * Keep the graphs (rewritten for their backend) so `variant()` can build more sessions over them.
   */
  variants?: boolean;
  /** The caller's own rewrite of each graph, before the backend's (`LoadOptions.rewrite`). */
  rewrite?: (graph: Uint8Array, filename: string) => Uint8Array;
}

/** Quantizations a browser backend refuses, and the backend that refuses them. */
const BLOCKED_ON: readonly { marker: string; backend: Backend; reason: string }[] = [
  {
    marker: "hqq",
    backend: "webgpu",
    reason:
      "an HQQ quantization writes float zero points, which WebGPU's MatMulNBits rejects " +
      "('zeroPoints input size error'; checked again on 2026-10-05 with the blocks repacked to 32: " +
      "WebGPU reads only packed 4-bit zero points, and HQQ's are not integers)",
  },
];

/** Take a backend out of a plan, with the reason; null when nothing would be left. */
function without(plan: BackendPlan, backend: Backend, reason: string): BackendPlan | null {
  if (!plan.order.includes(backend)) return plan;
  const order = plan.order.filter((b) => b !== backend);
  if (order.length === 0) return null;
  return { ...plan, order, dropped: [...plan.dropped, { backend, reason }] };
}

type InnerSession = {
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
  release?(): Promise<void>;
};

class Session implements ModelSession {
  #lastRunMs = 0;
  #timeoutMs = 300_000;
  #answered = false;
  /**
   * The same graph on the backends after this one, built on demand: a WebGPU kernel can accept a
   * graph and then refuse to run it (bge-m3's `Attention` with a mask: "Mask not supported"), which
   * no session creation catches. On such a failure, before any run has answered, the session moves
   * there and the call is made again.
   */
  fallback: (() => Promise<Session>) | null = null;
  logger: Logger | null = null;

  constructor(
    readonly role: GraphRole,
    readonly filename: string,
    public backend: Backend,
    public inputs: TensorSpec[],
    public outputs: TensorSpec[],
    readonly metadata: Readonly<Record<string, string>>,
    private inner: InnerSession,
  ) {}

  get lastRunMs(): number {
    return this.#lastRunMs;
  }

  set timeoutMs(value: number) {
    this.#timeoutMs = value;
  }

  async run(feeds: Record<string, unknown>): Promise<Record<string, unknown>> {
    const started = performance.now();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const expired = new Promise<never>((_, reject) => {
      timer = setTimeout(
        () =>
          reject(
            new InferenceError(
              `${this.filename} did not answer within ${Math.round(this.#timeoutMs / 1000)} s on ` +
                `${BACKEND_LABELS[this.backend]}. A device lost under WebGPU never settles; ` +
                `the session should be rebuilt.`,
            ),
          ),
        this.#timeoutMs,
      );
    });
    try {
      const outputs = await Promise.race([exclusive(() => this.inner.run(feeds)), expired]);
      this.#lastRunMs = performance.now() - started;
      this.#answered = true;
      return outputs;
    } catch (error) {
      if (this.#answered || this.backend !== "webgpu" || !this.fallback) throw error;
      const fallback = this.fallback;
      this.fallback = null;
      const next = await fallback().catch(() => null);
      if (!next) throw error;
      this.logger?.warn(`${this.filename}: WebGPU would not run it (${(error as Error).message}); moved to ${BACKEND_LABELS[next.backend]}`);
      await exclusive(async () => this.inner.release?.()).catch(() => undefined);
      this.inner = next.inner;
      this.backend = next.backend;
      this.inputs = next.inputs;
      this.outputs = next.outputs;
      return this.run(feeds);
    } finally {
      clearTimeout(timer);
    }
  }

  async close(): Promise<void> {
    await exclusive(async () => this.inner.release?.());
  }
}

/** A loaded model: its manifest, and a session per graph. */
/** Which outputs are worth keeping on the GPU. */
function keepOnDevice(outputs: readonly string[]): Record<string, "cpu" | "gpu-buffer"> {
  const location: Record<string, "cpu" | "gpu-buffer"> = {};
  for (const name of outputs) {
    if (name.startsWith("present")) location[name] = "gpu-buffer";
  }
  return location;
}

export class ModelBundle {
  /** Builds another session over a kept graph (`variants`), on the backend its first one took. */
  readonly #rebuild = new Map<GraphRole, (config: SessionConfig) => Promise<ModelSession>>();
  readonly #variants = new Set<ModelSession>();

  private constructor(
    readonly manifest: GraphManifest,
    /** The key its files were fetched with — the one to build any further URL from. */
    readonly key: ModelKey,
    private readonly sessions: Map<GraphRole, ModelSession>,
    private readonly assets: ReadonlyMap<string, ArrayBuffer>,
    /** The WebGPU adapter's buffer limits, for sizing batches; null without WebGPU. */
    readonly adapterLimits: AdapterLimits | null = null,
  ) {}

  static async create(options: BundleOptions): Promise<ModelBundle> {
    const { manifest, logger } = options;
    if (manifest.graphs.length === 0) {
      throw new ManifestError(`manifest '${manifest.modelKey}' declares no graphs`);
    }

    let plan = options.plan;
    if (plan.order.includes("wasm")) {
      const verdict = checkWasmFit(totalSizeMb(manifest));
      if (!verdict.ok) {
        const order = plan.order.filter((backend) => backend !== "wasm");
        if (order.length === 0) throw new ManifestError(verdict.reason ?? "model too large for WebAssembly");
        logger?.warn(`WebAssembly is left out of the fallbacks: ${verdict.reason ?? "model too large"}`);
        plan = { ...plan, order, dropped: [...plan.dropped, { backend: "wasm", reason: verdict.reason ?? "too large" }] };
      } else if (verdict.level === "warn" && verdict.reason) {
        logger?.warn(verdict.reason);
      }
    }
    options = { ...options, plan };

    const sessions = new Map<GraphRole, ModelSession>();
    const rebuilders = new Map<GraphRole, (config: SessionConfig) => Promise<ModelSession>>();
    try {
      for (const graph of manifest.graphs) {
        let graphPlan: BackendPlan | null = options.planFor?.[graph.role] ?? plan;

        for (const blocked of BLOCKED_ON) {
          if (!manifest.modelKey.toLowerCase().includes(blocked.marker)) continue;
          graphPlan = graphPlan && without(graphPlan, blocked.backend, blocked.reason);
          if (!graphPlan) throw new ManifestError(`${graph.filename} cannot run here: ${blocked.reason}`);
          logger?.warn(`${graph.filename} will not use ${BACKEND_LABELS[blocked.backend]}: ${blocked.reason}`);
        }
        const bytes = options.files.get(graph.filename);
        if (graphPlan && bytes && options.adapterLimits && graphPlan.order.includes("webgpu")) {
          const largest = readGraphInfo(new Uint8Array(bytes)).largestInitializerBytes;
          const verdict = checkAdapterFit(largest, options.adapterLimits);
          if (!verdict.ok) {
            graphPlan = without(graphPlan, "webgpu", verdict.reason ?? "too large for this adapter");
            if (!graphPlan) throw new ManifestError(`${graph.filename}: ${verdict.reason ?? "too large for this adapter"}`);
            logger?.warn(`${graph.filename} will not use WebGPU: ${verdict.reason ?? ""}`);
          }
        }

        const merged = graph.role === "decoder_merged" && graphPlan!.order[0] === "webgpu" && bytes ? splitMergedDecoder(new Uint8Array(bytes)) : null;
        const parts: [GraphEntry, ArrayBuffer | undefined][] = merged
          ? [
              [{ ...graph, filename: `${graph.filename}#first-pass`, role: "decoder" }, merged.firstPass.slice().buffer],
              [{ ...graph, filename: `${graph.filename}#with-past`, role: "decoder_with_past" }, merged.withPast.slice().buffer],
            ]
          : [[graph, undefined]];
        if (merged) logger?.info(`${graph.filename}: merged decoder split into its two branches for WebGPU`);
        const built = new Map<GraphRole, { session: ModelSession; rebuild: (config: SessionConfig) => Promise<ModelSession> }>();
        try {
          for (const [entry, own] of parts) {
            const files = own ? new Map([...options.files, [entry.filename, own]]) : options.files;
            built.set(entry.role, await ModelBundle.#buildOne(entry, { ...options, files, plan: graphPlan! }));
          }
        } catch (error) {
          if (!merged) throw error;

          await Promise.all([...built.values()].map((b) => b.session.close().catch(() => undefined)));
          built.clear();
          logger?.warn(`${graph.filename}: a branch of the split decoder would not build (${(error as Error).message}); the merged decoder is used`);
          built.set(graph.role, await ModelBundle.#buildOne(graph, { ...options, plan: graphPlan! }));
        }
        for (const [role, { session, rebuild }] of built) {
          sessions.set(role, session);
          if (options.variants) rebuilders.set(role, rebuild);
        }
      }
    } catch (error) {
      await Promise.all([...sessions.values()].map((s) => s.close().catch(() => undefined)));
      throw error;
    }

    const graphFiles = new Set(manifest.graphs.flatMap((g) => [g.filename, ...g.externalData]));
    const assets = new Map<string, ArrayBuffer>();
    for (const [name, bytes] of options.files) {
      if (!graphFiles.has(name)) assets.set(name, bytes);
    }
    const bundle = new ModelBundle(manifest, options.key, sessions, assets, options.adapterLimits ?? null);
    for (const [role, rebuild] of rebuilders) bundle.#rebuild.set(role, rebuild);
    return bundle;
  }

  static async #buildOne(
    graph: GraphEntry,
    options: BundleOptions,
  ): Promise<{ session: ModelSession; rebuild: (config: SessionConfig) => Promise<ModelSession> }> {
    const bytes = options.files.get(graph.filename);
    if (!bytes) {
      throw new ModelNotFoundError(
        `manifest '${options.manifest.modelKey}' declares ${graph.filename}, ` +
          `which was not fetched. Files present: ${[...options.files.keys()].join(", ") || "none"}`,
      );
    }

    const externalData: { path: string; data: Uint8Array }[] = [];
    const missing: string[] = [];
    for (const name of graph.externalData) {
      const data = options.files.get(name);
      if (data) externalData.push({ path: name, data: new Uint8Array(data) });
      else missing.push(name);
    }
    if (missing.length > 0) {
      throw new ModelNotFoundError(
        `${graph.filename} needs its external data (${missing.join(", ")}), which was ` +
          `not fetched. Without it the graph loads with no weights at all.`,
      );
    }

    let model: Uint8Array = new Uint8Array(bytes);
    if (options.rewrite) model = options.rewrite(model, graph.filename);

    const rows = gatherBeforeDequantize(model);
    if (rows.rewritten > 0) {
      model = rows.bytes;
      options.logger?.info(`${graph.filename}: ${rows.rewritten} Gather(s) of a quantized table dequantize the rows they read only`);
    }

    if (options.plan.order.includes("wasm")) {
      const level = setNBitsAccuracy(model);
      if (level.rewritten > 0) {
        model = level.bytes;
        options.logger?.info(`${graph.filename}: ${level.rewritten} MatMulNBits node(s) set to accuracy level ${CPU_NBITS_ACCURACY} (int8 compute on the CPU)`);
      }
    }
    if (options.plan.order.includes("webgpu")) {
      const repacked = repackMatMulNBits(model);
      if (repacked.rewritten > 0) {
        model = repacked.bytes;
        options.logger?.info(`${graph.filename}: ${repacked.rewritten} MatMulNBits node(s) rewritten to blocks of ${WEBGPU_NBITS_BLOCK} for WebGPU`);
      }
      const widened = padMatMulNBitsRows(model);
      if (widened.padded > 0) {
        model = widened.bytes;
        options.logger?.info(`${graph.filename}: ${widened.padded} MatMulNBits node(s) padded to whole-word zero points for WebGPU`);
      }
      const argmaxes = replaceArgMax(model);
      if (argmaxes.replaced > 0) {
        model = argmaxes.bytes;
        options.logger?.info(`${graph.filename}: ${argmaxes.replaced} ArgMax node(s) rebuilt from ReduceMax for WebGPU`);
      }
      const nans = replaceIsNaN(model);
      if (nans.replaced > 0) {
        model = nans.bytes;
        options.logger?.info(`${graph.filename}: ${nans.replaced} IsNaN node(s) rebuilt from Equal for WebGPU`);
      }
      const gathered = castGatherIndices(model);
      if (gathered.casts > 0) {
        model = gathered.bytes;
        options.logger?.info(`${graph.filename}: ${gathered.casts} computed Gather index(es) cast to int32 for WebGPU`);
      }
    }

    if (externalData.length === 0) {
      const moved = externalizeWeights(model, `${graph.filename}.weights`);
      if (moved) {
        model = moved.bytes;
        externalData.push(...moved.files);
        options.logger?.info(`${graph.filename}: weights moved out of the graph into ${moved.files.length} buffer(s), loaded as external data`);
      }
    }
    const info = readGraphInfo(model);
    const metadata = info.metadata;
    const build = async (plan: BackendPlan, config: SessionConfig | undefined): Promise<ModelSession> => {
      const built = await buildSession({
        ort: options.ort,
        model,
        plan,
        config: {
          ...config,
          ...(externalData.length > 0 ? { externalData } : {}),
          ...(config?.preferredOutputLocation === undefined
            ? { preferredOutputLocation: keepOnDevice(info.outputs.map((o) => o.name)) }
            : {}),
        },
        ...(options.logger ? { logger: options.logger } : {}),
        label: graph.filename,
        ...(options.wasmPaths !== undefined ? { wasmPaths: options.wasmPaths } : {}),
      });
      const session = new Session(
        graph.role,
        graph.filename,
        built.backend,
        readSpecs(built.session, "input"),
        readSpecs(built.session, "output"),
        metadata,
        built.session as unknown as {
          run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
          release?(): Promise<void>;
        },
      );
      if (options.runTimeoutMs !== undefined) session.timeoutMs = options.runTimeoutMs;
      if (built.backend === "webgpu") {
        const rest = without(plan, "webgpu", "a WebGPU kernel refused to run it");
        if (rest) session.fallback = async () => (await build(rest, config)) as Session;
        session.logger = options.logger ?? null;
      }
      return session;
    };
    const session = await build(options.plan, options.config);

    const rebuild = (config: SessionConfig) => build({ ...options.plan, order: [session.backend] }, { ...options.config, ...config });
    return { session, rebuild };
  }

  /**
   * Another session over the graph of `role`, with its own options — fixed shapes
   * (`freeDimensionOverrides`), outputs left on the GPU, graph capture — on the backend the first
   * one took. Needs `variants` at load (the graph is kept for it). Closed with the bundle, or on
   * its own.
   */
  async variant(role: GraphRole, config: SessionConfig): Promise<ModelSession> {
    const rebuild = this.#rebuild.get(role);
    if (!rebuild) {
      throw new ManifestError(
        this.sessions.has(role)
          ? `'${this.manifest.modelKey}' was loaded without 'variants': its ${role} graph was not kept`
          : `'${this.manifest.modelKey}' has no ${role} graph`,
      );
    }
    const session = await rebuild(config);
    this.#variants.add(session);
    const close = session.close.bind(session);
    (session as { close(): Promise<void> }).close = async () => {
      this.#variants.delete(session);
      await close();
    };
    return session;
  }

  /** A non-graph file fetched with the model (tokenizer, config), or null. */
  asset(filename: string): ArrayBuffer | null {
    return this.assets.get(filename) ?? null;
  }

  /** The same, as UTF-8 text. */
  assetText(filename: string): string | null {
    const bytes = this.assets.get(filename);
    return bytes ? new TextDecoder().decode(bytes) : null;
  }

  /** The session for a role, or null. Mirrors `GraphManifest.by_role`. */
  get(role: GraphRole): ModelSession | null {
    return this.sessions.get(role) ?? null;
  }

  /** The session for a role, or a message naming what is there instead. */
  require(role: GraphRole): ModelSession {
    const session = this.sessions.get(role);
    if (!session) {
      throw new ManifestError(
        `'${this.manifest.modelKey}' has no ${role} graph ` +
          `(roles present: ${[...this.sessions.keys()].join(", ") || "none"})`,
      );
    }
    return session;
  }

  /** Decoder roles present, in the pack's canonical order. */
  get decoderRoles(): GraphRole[] {
    return manifestDecoderRoles(this.manifest);
  }

  get roles(): GraphRole[] {
    return [...this.sessions.keys()];
  }

  /** The single session, for a one-graph model. */
  get only(): ModelSession {
    if (this.sessions.size !== 1) {
      throw new ManifestError(
        `'${this.manifest.modelKey}' has ${this.sessions.size} graphs; ` +
          `use get(role) rather than only`,
      );
    }
    return [...this.sessions.values()][0]!;
  }

  /** What actually ran, which may not be what was asked for. */
  get backend(): Backend {
    return this.only_backend();
  }

  private only_backend(): Backend {
    const first = [...this.sessions.values()][0];
    if (!first) throw new ManifestError("this bundle holds no sessions");
    return first.backend;
  }

  async close(): Promise<void> {
    await Promise.all([...this.#variants, ...this.sessions.values()].map((s) => s.close()));
    this.sessions.clear();
    this.#rebuild.clear();
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.close();
  }
}
