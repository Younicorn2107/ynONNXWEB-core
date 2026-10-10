/** The two calls a user of this library actually makes. */

import { ManifestError, ModelNotFoundError, QuotaError } from "../errors.js";
import { defaultLogger, type Logger } from "../logger.js";
import { detectCapabilities } from "../backends/capabilities.js";
import { probeBackends } from "../backends/probe.js";
import { selectBackends } from "../backends/select.js";
import { requestPersistence } from "../backends/memory.js";
import { checkSupported, resolveComputeType, warnIfSlow } from "../types/compute.js";
import type { BackendSpec } from "../backends/chain.js";
import type { BackendPlan } from "../backends/select.js";
import { entryName, splitEntry } from "../models/entry.js";
import { buildManifest } from "../models/introspect.js";
import { readGraphInfo } from "../models/onnxProto.js";
import { choosePublishedGraphs, neededGraphs, SIDE_FILES } from "../models/published.js";
import { splitExternalData } from "../models/externalize.js";
import {
  MANIFEST_VERSION_WITH_EXTERNAL_DATA,
  parseManifest,
  serializeManifest,
  totalSizeMb,
  type GraphManifest,
  type GraphRole,
} from "../types/manifest.js";
import {
  serverResolver,
  externalDataCandidates,
  filesFromManifest,
  type AssetPlace,
  type ModelKey,
  type ModelResolver,
} from "../models/resolve.js";
import { fetchAsset, type FetchProgress } from "../models/fetch.js";
import { ensureRoom, entryOf, holdEntries, releaseEntries, touchEntries } from "../models/eviction.js";
import { ReadOnlyStore, defaultStore, type AssetStore } from "../models/store.js";
import { ModelBundle, type ModelSession } from "./bundle.js";
import type { OrtLike, SessionConfig } from "./create.js";

const MIB = 1024 * 1024;

/** Where a manifest described on this side is kept between loads. */
const SYNTHESIZED_MANIFEST = "described_manifest.json";

/** The sha256 of each stored graph, as its published manifest gave it. */
const STORED_DIGESTS = "stored_digests.json";

/** One value for the whole model, or one per half. */
export type PerRole<T> = T | { encoder?: T; decoder?: T; default?: T };

export interface LoadOptions {
  /** The onnxruntime-web module. Injected so the page owns exactly one. */
  ort: OrtLike;
  model: ModelKey | string;
  /**
   * Where the model runs, or where each half runs: `{ encoder: "npu", decoder: "gpu" }`. An encoder
   * has fixed shapes and suits an NPU; a decoder steps with a growing cache and suits a GPU.
   */
  backends?: PerRole<BackendSpec>;
  /**
   * One precision, or one per half — `{ encoder: "int8", decoder: "int4" }` — assembled from the
   * entries of each precision, as on the Python side. Refused before the download when the chosen
   * backend cannot run it.
   */
  computeType?: PerRole<string>;
  /** One source, or several tried in order: the first with the model serves every file. */
  resolver?: ModelResolver | readonly ModelResolver[];
  store?: AssetStore;
  session?: SessionConfig;
  logger?: Logger;
  signal?: AbortSignal;
  onProgress?: (progress: FetchProgress) => void;
  /** Where onnxruntime-web's .wasm files are served from. */
  wasmPaths?: string | Record<string, string>;
  /** Run the live backend probe before choosing. On by default. */
  probe?: boolean;
  /**
   * Files the caller needs beside the graphs (`config.json`, `tokenizer_config.json`…). Fetched
   * when present, skipped when absent; read back with `bundle.asset()`.
   */
  extraFiles?: readonly string[];
  /** Load every graph of the manifest, even those a merged decoder makes redundant. */
  allGraphs?: boolean;
  /**
   * Keep the graphs so `bundle.variant()` can build more sessions over them (fixed shapes, outputs
   * on the GPU, graph capture). Off by default: a language model's graphs would be held twice.
   */
  variants?: boolean;
  /**
   * Rewrite each graph before its session is built (the bytes as fetched, the stored copy
   * untouched): an output reduced in the graph (`argmaxHead`), a node replaced. The backend's own
   * rewrites come after.
   */
  rewrite?: (graph: Uint8Array, filename: string) => Uint8Array;
}

type Half = "encoder" | "decoder";

function isRoleMap<T>(value: PerRole<T> | undefined): value is { encoder?: T; decoder?: T; default?: T } {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    ("encoder" in value || "decoder" in value || "default" in value)
  );
}

function forHalf<T>(value: PerRole<T> | undefined, half: Half, fallback: T): T {
  if (!isRoleMap(value)) return (value as T | undefined) ?? fallback;
  return value[half] ?? value.default ?? fallback;
}

function halfOf(role: GraphRole): Half {
  return role === "encoder" ? "encoder" : "decoder";
}

function asKey(model: ModelKey | string): ModelKey {
  if (typeof model !== "string") return model;

  const [name, computeType] = splitEntry(model);
  return { name, computeType };
}

interface Found {
  manifest: GraphManifest;
  resolver: ModelResolver;
  /** Files already fetched while describing the model. */
  files: Map<string, ArrayBuffer>;
  /** Files that live at the model root rather than beside the graphs. */
  rootFiles: ReadonlySet<string>;
}

function rootFilesOf(manifest: GraphManifest): Set<string> {
  const listed = manifest.extra["root_files"];
  return new Set(Array.isArray(listed) ? listed.filter((v): v is string => typeof v === "string") : []);
}

async function fromStore(key: ModelKey, store: AssetStore): Promise<GraphManifest | null> {
  const bytes = await store.read(`${entryName(key.name, key.computeType)}/${SYNTHESIZED_MANIFEST}`);
  if (!bytes) return null;
  try {
    return parseManifest(JSON.parse(new TextDecoder().decode(bytes)));
  } catch {
    return null;
  }
}

async function published(
  key: ModelKey,
  resolver: ModelResolver,
  options: LoadOptions,
  tried: string[],
): Promise<GraphManifest | null> {
  for (const url of resolver.manifestUrls(key)) {
    let response: Response;
    try {
      response = await fetch(url, { ...(options.signal ? { signal: options.signal } : {}) });
    } catch (cause) {
      if (options.signal?.aborted) throw cause;
      tried.push(`${url} (${cause instanceof Error ? cause.message : String(cause)})`);
      continue;
    }
    if (response.ok) return parseManifest(await response.json());
    tried.push(`${url} (HTTP ${response.status})`);
  }
  return null;
}

/**
 * Drop the stored graphs a newer publication replaced. A stored file is otherwise served forever: a
 * server that re-exports a model under the same name (a dynamic-size YOLO in place of a fixed one)
 * would never be seen. The digests the manifest declares are recorded with the files; a graph
 * stored before any record is hashed once.
 */
async function dropChanged(key: ModelKey, manifest: GraphManifest, store: AssetStore, logger: Logger): Promise<void> {
  const declared = manifest.graphs.filter((g) => g.sha256);
  if (declared.length === 0) return;
  const entry = entryName(key.name, key.computeType);
  let recorded: Record<string, string> = {};
  try {
    const bytes = await store.read(`${entry}/${STORED_DIGESTS}`);
    if (bytes) recorded = JSON.parse(new TextDecoder().decode(bytes)) as Record<string, string>;
  } catch {
    recorded = {};
  }
  let changed = false;
  for (const graph of declared) {
    const at = `${entry}/${graph.filename}`;
    let digest: string | undefined = recorded[graph.filename];
    if (digest === undefined) {
      const bytes = await store.read(at);
      if (!bytes) continue;
      digest = await sha256Hex(bytes);
    }
    if (digest !== graph.sha256) {
      logger.info(`${at} changed upstream: fetching it again`);
      await store.delete(at);
      for (const sidecar of graph.externalData) await store.delete(`${entry}/${sidecar}`);
    }
    if (recorded[graph.filename] !== graph.sha256) changed = true;
    recorded[graph.filename] = graph.sha256!;
  }
  if (changed) await store.write(`${entry}/${STORED_DIGESTS}`, new Response(JSON.stringify(recorded)));
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Describe a model from its own files, for a source that publishes no manifest. */
/**
 * The largest weight file taken in one buffer (Chromium refuses an ArrayBuffer past 2 GB on many
 * machines); a larger one is taken in parts.
 */
const MAX_SINGLE_BUFFER = 2 * 2 ** 30 - 64 * 2 ** 20;

async function describe(
  key: ModelKey,
  resolver: ModelResolver,
  storeIn: AssetStore,
  options: LoadOptions,
  logger: Logger,
): Promise<Found | null> {
  let store = storeIn;
  const listing = await resolver.listFiles?.(key, options.signal);
  if (!listing) return null;

  const entry = entryName(key.name, key.computeType);
  const sizes = new Map(listing.graphs.map((f) => [f.name, f.size ?? 0]));
  const choice = choosePublishedGraphs(listing.graphs.map((f) => f.name), key.computeType);

  const announced = listing.graphs
    .filter((f) => choice.graphs.some((graph) => f.name === graph || f.name.startsWith(`${graph}_data`) || f.name.startsWith(`${graph}.data`)))
    .reduce((sum, f) => sum + (f.size ?? 0), 0);
  const stored = await Promise.all(choice.graphs.map(async (g) => ((await options.store?.has(`${entry}/${g}`)) ? (sizes.get(g) ?? 0) : 0)));
  try {
    await ensureRoom(options.store ?? defaultStore(), Math.max(0, announced - stored.reduce((a, b) => a + b, 0)), new Set([entry]), logger);
  } catch (error) {
    if (!(error instanceof QuotaError)) throw error;
    store = new ReadOnlyStore(store);
    logger.warn(`${error.message} Loaded without being kept: it will be downloaded again next time.`);
  }
  logger.info(
    `${resolver.id} publishes no manifest for '${key.name}': describing ` +
      `${choice.graphs.join(", ")} from the graphs themselves`,
  );

  const files = new Map<string, ArrayBuffer>();
  const fetchOne = async (name: string, place: AssetPlace): Promise<ArrayBuffer> => {
    const asset = resolver.assetUrl(key, name, place);
    const bytes = await fetchAsset(asset.url, `${entry}/${name}`, {
      store,
      logger,
      ...(options.signal ? { signal: options.signal } : {}),
      ...(options.onProgress ? { onProgress: options.onProgress } : {}),
      ...(asset.headers ? { headers: asset.headers } : {}),
    });
    files.set(name, bytes);
    return bytes;
  };

  const graphs = [];
  for (const name of choice.graphs) {
    let bytes: Uint8Array = new Uint8Array(await fetchOne(name, "graphs"));
    const sidecarBytes: Record<string, number> = {};
    for (const sidecar of readGraphInfo(bytes).externalData) {
      if (!sizes.has(sidecar)) {
        throw new ModelNotFoundError(
          `${name} keeps its weights in ${sidecar}, which ${resolver.id} does not publish`,
        );
      }
      if ((sizes.get(sidecar) ?? 0) <= MAX_SINGLE_BUFFER) {
        sidecarBytes[sidecar] = (await fetchOne(sidecar, "graphs")).byteLength;
        continue;
      }

      const split = splitExternalData(bytes, sidecar);
      logger.info(`${sidecar} (${Math.round((sizes.get(sidecar) ?? 0) / 2 ** 20)} MB) is taken in ${split.parts.length} parts`);
      const asset = resolver.assetUrl(key, sidecar, "graphs");
      for (const part of split.parts) {
        const data = await fetchAsset(asset.url, `${entry}/${part.name}`, {
          store,
          logger,
          range: [part.start, part.end],
          ...(options.signal ? { signal: options.signal } : {}),
          ...(options.onProgress ? { onProgress: options.onProgress } : {}),
          ...(asset.headers ? { headers: asset.headers } : {}),
        });
        files.set(part.name, data);
        sidecarBytes[part.name] = data.byteLength;
      }
      bytes = split.bytes;
    }
    if (bytes.buffer !== files.get(name)) {
      files.set(name, bytes.slice().buffer);
      await store.write(`${entry}/${name}`, new Response(bytes.slice(), { headers: { "content-length": String(bytes.length) } }));
    }
    graphs.push({ filename: name, bytes, sidecarBytes });
  }

  const rootNames = new Set(listing.root.map((f) => f.name));
  const graphNames = new Set(listing.graphs.map((f) => f.name));
  const rootFiles = new Set<string>();
  for (const name of new Set([...SIDE_FILES, ...(options.extraFiles ?? [])])) {
    const place: AssetPlace | null = graphNames.has(name) ? "graphs" : rootNames.has(name) ? "root" : null;
    if (place === null) continue;
    await fetchOne(name, place);
    if (place === "root") rootFiles.add(name);
  }

  let architecture: string | null = null;
  const config = files.get("config.json");
  if (config) {
    try {
      const parsed = JSON.parse(new TextDecoder().decode(config)) as { model_type?: unknown };
      if (typeof parsed.model_type === "string") architecture = parsed.model_type;
    } catch {
      // Not fatal: carry on.
    }
  }

  const manifest = buildManifest(graphs, {
    modelKey: entry,
    computeType: key.computeType,
    architecture,
    tokenizerFile: files.has("tokenizer.json") ? "tokenizer.json" : null,
  });
  manifest.extra = { described_from: resolver.id, root_files: [...rootFiles] };
  await store.write(
    `${entry}/${SYNTHESIZED_MANIFEST}`,
    new Response(JSON.stringify(serializeManifest(manifest)), {
      headers: { "content-type": "application/json" },
    }),
  );
  return { manifest, resolver, files, rootFiles };
}

async function findManifest(
  key: ModelKey,
  resolvers: readonly ModelResolver[],
  store: AssetStore,
  options: LoadOptions,
  logger: Logger,
): Promise<Found> {
  const stored = await fromStore(key, store);
  if (stored) {
    const id = stored.extra["described_from"];
    const resolver = resolvers.find((r) => r.id === id) ?? resolvers[0]!;
    return { manifest: stored, resolver, files: new Map(), rootFiles: rootFilesOf(stored) };
  }

  const tried: string[] = [];
  for (const resolver of resolvers) {
    let manifest = await published(key, resolver, options, tried);

    if (!manifest && resolver.prepare) {
      logger.info(`${key.name} (${key.computeType}) is not at ${resolver.id}: asking it to prepare it`);
      const made = await resolver.prepare(key, options.signal);
      if (made.ready) manifest = await published(key, resolver, options, tried);
      else tried.push(`${resolver.id} could not prepare it: ${made.error ?? "no reason given"}`);
    }
    if (manifest) {
      await dropChanged(key, manifest, store, logger);
      return { manifest, resolver, files: new Map(), rootFiles: rootFilesOf(manifest) };
    }
  }
  for (const resolver of resolvers) {
    const found = await describe(key, resolver, store, options, logger);
    if (found) return found;
  }

  if (tried.length === 0) {
    throw new ManifestError(
      `no source can describe '${key.name}' (${resolvers.map((r) => r.id).join(", ")}): none ` +
        `publishes a manifest and none can list its files.`,
    );
  }
  throw new ModelNotFoundError(
    `'${key.name}' at ${key.computeType} was found nowhere. Tried:\n  ${tried.join("\n  ")}`,
  );
}

/** Fetch a file that may legitimately be absent; null when it is. */
async function fetchOptional(
  url: string,
  storeKey: string,
  store: AssetStore,
  options: LoadOptions,
  headers?: Record<string, string>,
): Promise<ArrayBuffer | null> {
  const stored = await store.read(storeKey);
  if (stored) return stored;
  let response: Response;
  try {
    response = await fetch(url, {
      ...(options.signal ? { signal: options.signal } : {}),
      ...(headers ? { headers } : {}),
    });
  } catch (cause) {
    if (options.signal?.aborted) throw cause;
    return null;
  }
  if (!response.ok) return null;
  const bytes = await response.arrayBuffer();
  await store.write(storeKey, new Response(bytes));
  return bytes;
}

/** Load the graphs a model runs on, and return them addressed by role. */
export async function loadModel(options: LoadOptions): Promise<ModelBundle> {
  const logger = options.logger ?? defaultLogger;

  let key = asKey(options.model);
  const resolvers: readonly ModelResolver[] =
    options.resolver === undefined
      ? [serverResolver()]
      : Array.isArray(options.resolver)
        ? (options.resolver as readonly ModelResolver[])
        : [options.resolver as ModelResolver];
  if (resolvers.length === 0) throw new ManifestError("loadModel needs at least one source");
  let store: AssetStore = options.store ?? defaultStore();

  const capabilities = await detectCapabilities();

  const wasmEnv = (options.ort as { env?: { wasm?: { wasmPaths?: unknown } } }).env?.wasm;
  if (options.wasmPaths !== undefined && wasmEnv && wasmEnv.wasmPaths === undefined) wasmEnv.wasmPaths = options.wasmPaths;
  if (options.probe !== false) {
    const candidates = capabilities.webnn
      ? (["webnn", "webgpu", "wasm"] as const)
      : (["webgpu", "wasm"] as const);
    const results = await probeBackends(
      options.ort as unknown as Parameters<typeof probeBackends>[0],
      candidates,
    );
    capabilities.usable = results.filter((r) => r.ok).map((r) => r.backend);
    for (const failed of results.filter((r) => !r.ok)) {
      logger.debug(`${failed.backend} failed its probe: ${failed.error ?? "unknown"}`);
    }
  }

  const plans: Record<Half, BackendPlan> = {
    encoder: selectBackends(forHalf(options.backends, "encoder", "auto"), capabilities, logger),
    decoder: selectBackends(forHalf(options.backends, "decoder", "auto"), capabilities, logger),
  };
  const plan = plans.decoder;

  const precisionFor = (half: Half): string => {
    const requested = forHalf(options.computeType, half, key.computeType);
    const winner = plans[half].order[0];
    const wanted = requested === "auto" && winner ? resolveComputeType(requested, winner) : requested;
    if (requested === "auto" && winner) logger.info(`compute type 'auto' resolved to '${wanted}' for the ${half} on ${winner}`);
    if (wanted && wanted !== "auto" && winner) {
      checkSupported(wanted, winner);

      warnIfSlow(wanted, winner, logger);
    }
    return wanted;
  };
  const decoderPrecision = precisionFor("decoder");
  const encoderPrecision = isRoleMap(options.computeType) || isRoleMap(options.backends) ? precisionFor("encoder") : decoderPrecision;
  key = { ...key, computeType: decoderPrecision };

  const found = await findManifest(key, resolvers, store, options, logger);
  let manifest = options.allGraphs ? found.manifest : neededGraphs(found.manifest);
  const files = found.files;
  const origin = new Map<string, { key: ModelKey; found: Found }>();
  for (const name of filesFromManifest(manifest)) origin.set(name, { key, found });
  for (const name of options.extraFiles ?? []) origin.set(name, { key, found });

  if (encoderPrecision !== decoderPrecision && manifest.graphs.some((g) => g.role === "encoder")) {
    const encoderKey = { ...key, computeType: encoderPrecision };
    const encoderFound = await findManifest(encoderKey, resolvers, store, options, logger);
    const encoderGraph = encoderFound.manifest.graphs.find((g) => g.role === "encoder");
    if (!encoderGraph) {
      throw new ManifestError(`'${key.name}' at ${encoderPrecision} has no encoder to take`);
    }

    manifest = {
      ...manifest,
      modelKey: `${entryName(key.name, decoderPrecision)}+encoder_${encoderPrecision}`,
      computeType: `encoder:${encoderPrecision},decoder:${decoderPrecision}`,
      graphs: [encoderGraph, ...manifest.graphs.filter((g) => g.role !== "encoder")],
    };
    for (const name of [encoderGraph.filename, ...encoderGraph.externalData]) {
      origin.set(name, { key: encoderKey, found: encoderFound });
      const early = encoderFound.files.get(name);
      if (early) files.set(name, early);
      else files.delete(name);
    }
  }

  const totalBytes = totalSizeMb(manifest) * MIB;

  const entries = new Set<string>();
  let present = 0;
  for (const filename of filesFromManifest(manifest)) {
    const from = origin.get(filename) ?? { key, found };
    const storeKey = `${entryName(from.key.name, from.key.computeType)}/${filename}`;
    entries.add(entryOf(storeKey));
    present += files.get(filename)?.byteLength ?? (await store.stat(storeKey))?.bytes ?? 0;
  }
  const missing = Math.max(0, totalBytes - present);

  await requestPersistence();
  if (missing > 0) {
    try {
      await ensureRoom(store, missing, entries, logger);
    } catch (error) {
      if (!(error instanceof QuotaError)) throw error;
      store = new ReadOnlyStore(store);
      logger.warn(`${error.message} Loaded without being kept: it will be downloaded again next time.`);
    }
  }

  let loaded = 0;
  for (const filename of filesFromManifest(manifest)) {
    if (files.has(filename)) {
      loaded += files.get(filename)!.byteLength;
      continue;
    }
    const from = origin.get(filename) ?? { key, found };
    const asset = from.found.resolver.assetUrl(from.key, filename, from.found.rootFiles.has(filename) ? "root" : "graphs");
    const bytes = await fetchAsset(asset.url, `${entryName(from.key.name, from.key.computeType)}/${filename}`, {
      store,
      ...(options.signal ? { signal: options.signal } : {}),
      ...(options.onProgress ? { onProgress: options.onProgress } : {}),
      logger,
      ...(asset.headers ? { headers: asset.headers } : {}),
      overall: { loaded, total: totalBytes },
    });
    files.set(filename, bytes);
    loaded += bytes.byteLength;
  }

  const mustProbe = manifest.version < MANIFEST_VERSION_WITH_EXTERNAL_DATA;
  for (const graph of mustProbe ? manifest.graphs : []) {
    if (graph.externalData.length > 0) continue;
    for (const candidate of externalDataCandidates(graph.filename)) {
      const from = origin.get(graph.filename) ?? { key, found };
      const asset = from.found.resolver.assetUrl(from.key, candidate);
      try {
        const probe = await fetch(asset.url, { method: "HEAD" });
        if (!probe.ok) continue;
      } catch {
        continue;
      }
      logger.info(
        `${graph.filename} has an external-data sidecar (${candidate}) that its ` +
          `manifest does not record; fetching it anyway`,
      );
      const bytes = await fetchAsset(asset.url, `${entryName(from.key.name, from.key.computeType)}/${candidate}`, {
        store,
        ...(options.signal ? { signal: options.signal } : {}),
        logger,
      });
      files.set(candidate, bytes);
      graph.externalData.push(candidate);
      break;
    }
  }

  for (const name of options.extraFiles ?? []) {
    if (files.has(name)) continue;
    const place: AssetPlace = found.rootFiles.has(name) ? "root" : "graphs";
    const asset = found.resolver.assetUrl(key, name, place);
    const bytes = await fetchOptional(asset.url, `${entryName(key.name, key.computeType)}/${name}`, store, options, asset.headers);
    if (bytes) files.set(name, bytes);
  }

  await touchEntries(store, entries);
  const bundle = await ModelBundle.create({
    ort: options.ort,
    manifest,

    key,
    files,
    plan,
    planFor: Object.fromEntries(manifest.graphs.map((g) => [g.role, plans[halfOf(g.role)]])),
    adapterLimits: capabilities.webgpuLimits,
    config: {
      ...(capabilities.webnnDeviceTypes[0]
        ? { webnnDeviceType: capabilities.webnnDeviceTypes[0] as "npu" | "gpu" | "cpu" }
        : {}),
      ...options.session,
    },
    logger,
    ...(options.wasmPaths !== undefined ? { wasmPaths: options.wasmPaths } : {}),
    ...(options.variants ? { variants: true } : {}),
    ...(options.rewrite ? { rewrite: options.rewrite } : {}),
  });

  holdEntries(entries);
  const close = bundle.close.bind(bundle);
  let released = false;
  bundle.close = async () => {
    if (!released) {
      released = true;
      releaseEntries(entries);
    }
    await close();
  };
  return bundle;
}

/**
 * The single-graph shortcut. Refuses, rather than picking one, when the model has several —
 * silently choosing would be a coin flip the caller never sees.
 */
export async function createSession(options: LoadOptions): Promise<ModelSession> {
  const bundle = await loadModel(options);
  return bundle.only;
}
