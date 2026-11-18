/** The two calls a user of this library actually makes. */

import { AssetTooLargeError, ManifestError, ModelNotFoundError, QuotaError } from "../errors.js";
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
  isSafeFileName,
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
  sourceOf,
  type AssetPlace,
  type ListedFile,
  type ModelKey,
  type ModelListing,
  type ModelResolver,
} from "../models/resolve.js";
import { fetchAsset, type FetchProgress } from "../models/fetch.js";
import { ensureRoom, entryOf, holdEntries, releaseEntries, touchEntries } from "../models/eviction.js";
import { checkSize, resolveLimits, type Limits } from "../models/limits.js";
import { sha256Hex } from "../models/sha256.js";
import { ReadOnlyStore, defaultStore, type AssetStore } from "../models/store.js";
import { ModelBundle, type ModelSession } from "./bundle.js";
import type { OrtLike, SessionConfig } from "./create.js";

const MIB = 1024 * 1024;

/** Where a manifest described on this side is kept between loads. */
const SYNTHESIZED_MANIFEST = "described_manifest.json";

/**
 * The SHA-256 of each file of an entry, computed from the bytes when they were downloaded (never
 * copied from a manifest): what a newer publication's declared digest is compared with.
 */
const STORED_DIGESTS = "stored_digests.json";

/** Which source an entry's files came from (`sourceOf(resolver)`). */
const SOURCE_RECORD = "source.json";

/** The largest manifest or small side file read whole (JSON, configs). */
const MAX_SIDE_FILE_BYTES = 64 * MIB;

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
  /**
   * The sizes a download, the whole model and one tensor may take (`DEFAULT_LIMITS` otherwise).
   * Every size a source or a graph states is checked against them before it is allocated or
   * fetched.
   */
  limits?: Partial<Limits>;
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

/** What `loadModel` keeps beside an entry's files, and how each load reads it. */
interface Context {
  options: LoadOptions;
  logger: Logger;
  limits: Limits;
  /** Digests computed while downloading, by entry then file, written once the files are in. */
  digests: Map<string, Record<string, string>>;
}

function rootFilesOf(manifest: GraphManifest): Set<string> {
  const listed = manifest.extra["root_files"];
  return new Set(Array.isArray(listed) ? listed.filter((v): v is string => typeof v === "string") : []);
}

/**
 * A response body read whole, refused past `maxBytes`: a manifest or a config is small, and a
 * source that streams gigabytes in its place must not fill the memory first.
 */
async function readBounded(response: Response, maxBytes: number, url: string): Promise<ArrayBuffer> {
  const declared = Number(response.headers.get("content-length") ?? "");
  if (Number.isFinite(declared) && declared > maxBytes) throw new AssetTooLargeError(`${url}: ${declared} bytes is over the limit of ${maxBytes}`);
  if (!response.body) {
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength > maxBytes) throw new AssetTooLargeError(`${url}: ${bytes.byteLength} bytes is over the limit of ${maxBytes}`);
    return bytes;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    length += value.byteLength;
    if (length > maxBytes) {
      await reader.cancel().catch(() => undefined);
      throw new AssetTooLargeError(`${url}: the body passed the limit of ${maxBytes} bytes`);
    }
    chunks.push(value);
  }
  const out = new Uint8Array(length);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.byteLength;
  }
  return out.buffer;
}

async function readJson(store: AssetStore, key: string): Promise<unknown> {
  try {
    const bytes = await store.read(key);
    return bytes ? (JSON.parse(new TextDecoder().decode(bytes)) as unknown) : null;
  } catch {
    return null;
  }
}

async function recordedSource(store: AssetStore, entry: string): Promise<string | null> {
  const record = await readJson(store, `${entry}/${SOURCE_RECORD}`);
  const source = record && typeof record === "object" ? (record as { source?: unknown }).source : undefined;
  return typeof source === "string" ? source : null;
}

/** Make `entry` hold only bytes from `resolver`'s source. */
async function claimEntry(store: AssetStore, entry: string, resolver: ModelResolver, logger: Logger): Promise<void> {
  const source = sourceOf(resolver);
  const recorded = await recordedSource(store, entry);
  if (recorded === source) return;
  const keys = (await store.keys()).filter((k) => entryOf(k) === entry);
  if (keys.length > 0) {
    logger.info(`${entry} was stored from ${recorded ?? "an unrecorded source"}, not ${source}: fetching it again`);
    for (const k of keys) await store.delete(k).catch(() => false);
  }
  await store.write(`${entry}/${SOURCE_RECORD}`, new Response(JSON.stringify({ source }), { headers: { "content-type": "application/json" } }));
}

async function recordedDigests(store: AssetStore, entry: string): Promise<Record<string, string>> {
  const record = await readJson(store, `${entry}/${STORED_DIGESTS}`);
  if (!record || typeof record !== "object" || Array.isArray(record)) return {};
  return Object.fromEntries(Object.entries(record as Record<string, unknown>).filter((e): e is [string, string] => typeof e[1] === "string"));
}

/** Add the digests computed during this load to each entry's record. */
async function writeDigests(store: AssetStore, digests: ReadonlyMap<string, Record<string, string>>): Promise<void> {
  for (const [entry, computed] of digests) {
    if (Object.keys(computed).length === 0) continue;
    const recorded = { ...(await recordedDigests(store, entry)), ...computed };
    await store.write(`${entry}/${STORED_DIGESTS}`, new Response(JSON.stringify(recorded))).catch(() => undefined);
  }
}

/**
 * A stored manifest described on an earlier load, with the source it was described from — null when
 * none is stored, or when it came from a source not among `resolvers`.
 */
async function fromStore(key: ModelKey, store: AssetStore, resolvers: readonly ModelResolver[]): Promise<{ manifest: GraphManifest; resolver: ModelResolver } | null> {
  const entry = entryName(key.name, key.computeType);
  const raw = await readJson(store, `${entry}/${SYNTHESIZED_MANIFEST}`);
  if (raw === null) return null;
  const source = await recordedSource(store, entry);
  const resolver = resolvers.find((r) => sourceOf(r) === source);
  if (!resolver) return null;
  try {
    return { manifest: parseManifest(raw), resolver };
  } catch {
    return null;
  }
}

/**
 * Whether a described manifest still matches its source: every file it was built from is listed at
 * the size it had then. A repository re-exported under the same names — an upstream fix, or a
 * poisoned copy — is otherwise never seen again. Offline, or when the source cannot list, the
 * stored description stands.
 */
async function stillListed(key: ModelKey, manifest: GraphManifest, resolver: ModelResolver, options: LoadOptions, logger: Logger): Promise<{ fresh: boolean; listing: ModelListing | null }> {
  if (!resolver.listFiles || (globalThis.navigator as { onLine?: boolean } | undefined)?.onLine === false) return { fresh: true, listing: null };
  let listing: ModelListing | null;
  try {
    listing = await resolver.listFiles(key, options.signal);
  } catch (cause) {
    if (options.signal?.aborted) throw cause;
    logger.debug(`${resolver.id} could not be listed (${cause instanceof Error ? cause.message : String(cause)}): the stored description of ${key.name} is used`);
    return { fresh: true, listing: null };
  }
  if (!listing) return { fresh: true, listing: null };
  const sizes = manifest.extra["listed_sizes"];
  if (!sizes || typeof sizes !== "object" || Array.isArray(sizes)) return { fresh: false, listing };
  const now = new Map([...listing.graphs, ...listing.root].map((f) => [f.name, f.size]));
  for (const [name, size] of Object.entries(sizes as Record<string, unknown>)) {
    if (!now.has(name) || now.get(name) !== size) {
      logger.info(`${name} of ${key.name} changed at ${resolver.id} (${String(size)} → ${String(now.get(name) ?? "gone")} bytes): describing it again`);
      return { fresh: false, listing };
    }
  }
  return { fresh: true, listing };
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
      response = await fetch(url, {
        ...(options.signal ? { signal: options.signal } : {}),
        ...(resolver.headers ? { headers: resolver.headers } : {}),
      });
    } catch (cause) {
      if (options.signal?.aborted) throw cause;
      tried.push(`${url} (${cause instanceof Error ? cause.message : String(cause)})`);
      continue;
    }
    if (response.ok) {
      const body = new TextDecoder().decode(await readBounded(response, MAX_SIDE_FILE_BYTES, url));
      let payload: unknown;
      try {
        payload = JSON.parse(body);
      } catch (cause) {
        throw new ManifestError(`${url} is not JSON`, { cause });
      }
      return parseManifest(payload);
    }
    tried.push(`${url} (HTTP ${response.status})`);
  }
  return null;
}

/**
 * Drop the stored graphs a newer publication replaced. A stored file is otherwise served forever: a
 * server that re-exports a model under the same name (a dynamic-size YOLO in place of a fixed one)
 * would never be seen.
 */
async function dropChanged(key: ModelKey, manifest: GraphManifest, store: AssetStore, logger: Logger): Promise<void> {
  const declared = manifest.graphs.filter((g) => g.sha256);
  if (declared.length === 0) return;
  const entry = entryName(key.name, key.computeType);
  const recorded = await recordedDigests(store, entry);
  let changed = false;
  for (const graph of declared) {
    const at = `${entry}/${graph.filename}`;
    let digest: string | undefined = recorded[graph.filename];
    if (digest === undefined) {
      const bytes = await store.read(at);
      if (!bytes) continue;
      digest = await sha256Hex(bytes);
      recorded[graph.filename] = digest;
      changed = true;
    }
    if (digest !== graph.sha256) {
      logger.info(`${at} changed upstream: fetching it again`);
      await store.delete(at);
      delete recorded[graph.filename];
      for (const sidecar of graph.externalData) {
        await store.delete(`${entry}/${sidecar}`);
        delete recorded[sidecar];
      }
      changed = true;
    }
  }
  if (changed) await store.write(`${entry}/${STORED_DIGESTS}`, new Response(JSON.stringify(recorded)));
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
  context: Context,
  listed?: ModelListing | null,
): Promise<Found | null> {
  const { options, logger, limits } = context;
  let store = storeIn;
  const raw = listed ?? (await resolver.listFiles?.(key, options.signal));
  if (!raw) return null;

  const plain = (files: readonly ListedFile[]) => files.filter((f) => isSafeFileName(f.name));
  const listing: ModelListing = { graphs: plain(raw.graphs), root: plain(raw.root) };

  const entry = entryName(key.name, key.computeType);
  await claimEntry(storeIn, entry, resolver, logger);
  const sizes = new Map(listing.graphs.map((f) => [f.name, f.size ?? 0]));
  const choice = choosePublishedGraphs(listing.graphs.map((f) => f.name), key.computeType);

  const announced = listing.graphs
    .filter((f) => choice.graphs.some((graph) => f.name === graph || f.name.startsWith(`${graph}_data`) || f.name.startsWith(`${graph}.data`)))
    .reduce((sum, f) => sum + (f.size ?? 0), 0);
  checkSize(announced, limits.maxModelBytes, `${key.name} at ${resolver.id}`, (m) => new AssetTooLargeError(m));
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
  const computed = context.digests.get(entry) ?? {};
  context.digests.set(entry, computed);
  const fetchOne = async (name: string, place: AssetPlace): Promise<ArrayBuffer> => {
    const asset = resolver.assetUrl(key, name, place);
    const bytes = await fetchAsset(asset.url, `${entry}/${name}`, {
      store,
      logger,
      limits,
      onDigest: (digest) => (computed[name] = digest),
      ...(options.signal ? { signal: options.signal } : {}),
      ...(options.onProgress ? { onProgress: options.onProgress } : {}),
      ...(asset.headers ? { headers: asset.headers } : {}),
    });
    files.set(name, bytes);
    return bytes;
  };

  const graphs = [];
  /** Sidecars taken in parts: in the description by their own name, not their parts'. */
  const splitFiles = new Set<string>();
  for (const name of choice.graphs) {
    let bytes: Uint8Array = new Uint8Array(await fetchOne(name, "graphs"));
    const sidecarBytes: Record<string, number> = {};
    for (const sidecar of readGraphInfo(bytes).externalData) {
      if (!sizes.has(sidecar)) {
        throw new ModelNotFoundError(
          `${name} keeps its weights in ${sidecar}, which ${resolver.id} does not publish`,
        );
      }
      const listedSize = sizes.get(sidecar) ?? 0;
      if (listedSize <= MAX_SINGLE_BUFFER) {
        sidecarBytes[sidecar] = (await fetchOne(sidecar, "graphs")).byteLength;
        continue;
      }

      const split = splitExternalData(bytes, sidecar, listedSize);
      logger.info(`${sidecar} (${Math.round(listedSize / 2 ** 20)} MB) is taken in ${split.parts.length} parts`);
      const asset = resolver.assetUrl(key, sidecar, "graphs");
      for (const part of split.parts) {
        const data = await fetchAsset(asset.url, `${entry}/${part.name}`, {
          store,
          logger,
          limits,
          range: [part.start, part.end],
          onDigest: (digest) => (computed[part.name] = digest),
          ...(options.signal ? { signal: options.signal } : {}),
          ...(options.onProgress ? { onProgress: options.onProgress } : {}),
          ...(asset.headers ? { headers: asset.headers } : {}),
        });
        files.set(part.name, data);
        sidecarBytes[part.name] = data.byteLength;
      }
      bytes = split.bytes;
      splitFiles.add(sidecar);
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

  const used = new Set([...files.keys(), ...splitFiles]);
  const listedSizes = Object.fromEntries([...listing.graphs, ...listing.root].filter((f) => used.has(f.name)).map((f) => [f.name, f.size]));
  manifest.extra = { described_from: resolver.id, root_files: [...rootFiles], listed_sizes: listedSizes };
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
  context: Context,
): Promise<Found> {
  const { options, logger } = context;
  const entry = entryName(key.name, key.computeType);
  let listed: { resolver: ModelResolver; listing: ModelListing } | null = null;
  const stored = await fromStore(key, store, resolvers);
  if (stored) {
    const check = await stillListed(key, stored.manifest, stored.resolver, options, logger);
    if (check.fresh) return { manifest: stored.manifest, resolver: stored.resolver, files: new Map(), rootFiles: rootFilesOf(stored.manifest) };

    for (const k of (await store.keys()).filter((k) => entryOf(k) === entry)) await store.delete(k).catch(() => false);
    listed = check.listing ? { resolver: stored.resolver, listing: check.listing } : null;
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
      await claimEntry(store, entry, resolver, logger);
      await dropChanged(key, manifest, store, logger);
      return { manifest, resolver, files: new Map(), rootFiles: rootFilesOf(manifest) };
    }
  }
  for (const resolver of resolvers) {
    const found = await describe(key, resolver, store, context, listed?.resolver === resolver ? listed.listing : undefined);
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
  context: Context,
  headers?: Record<string, string>,
): Promise<ArrayBuffer | null> {
  const { options } = context;
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
  const bytes = await readBounded(response, context.limits.maxFileBytes, url);
  const entry = entryOf(storeKey);
  const computed = context.digests.get(entry) ?? {};
  context.digests.set(entry, computed);
  computed[storeKey.slice(entry.length + 1)] = await sha256Hex(bytes);
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
  const limits = resolveLimits(options.limits);
  const context: Context = { options, logger, limits, digests: new Map() };

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

  const found = await findManifest(key, resolvers, store, context);
  let manifest = options.allGraphs ? found.manifest : neededGraphs(found.manifest);
  const files = found.files;
  const origin = new Map<string, { key: ModelKey; found: Found }>();
  for (const name of filesFromManifest(manifest)) origin.set(name, { key, found });
  for (const name of options.extraFiles ?? []) origin.set(name, { key, found });

  if (encoderPrecision !== decoderPrecision && manifest.graphs.some((g) => g.role === "encoder")) {
    const encoderKey = { ...key, computeType: encoderPrecision };
    const encoderFound = await findManifest(encoderKey, resolvers, store, context);
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
  checkSize(Math.round(totalBytes), limits.maxModelBytes, `${key.name} (${manifest.computeType}), as its manifest sizes it`, (m) => new AssetTooLargeError(m));

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
    const entry = entryName(from.key.name, from.key.computeType);
    const computed = context.digests.get(entry) ?? {};
    context.digests.set(entry, computed);

    const declared = manifest.graphs.find((g) => g.filename === filename)?.sha256 ?? null;
    const bytes = await fetchAsset(asset.url, `${entry}/${filename}`, {
      store,
      ...(options.signal ? { signal: options.signal } : {}),
      ...(options.onProgress ? { onProgress: options.onProgress } : {}),
      logger,
      limits,
      sha256: declared,
      onDigest: (digest) => (computed[filename] = digest),
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
        const probe = await fetch(asset.url, { method: "HEAD", ...(asset.headers ? { headers: asset.headers } : {}) });
        if (!probe.ok) continue;
      } catch {
        continue;
      }
      logger.info(
        `${graph.filename} has an external-data sidecar (${candidate}) that its ` +
          `manifest does not record; fetching it anyway`,
      );
      const entry = entryName(from.key.name, from.key.computeType);
      const computed = context.digests.get(entry) ?? {};
      context.digests.set(entry, computed);
      const bytes = await fetchAsset(asset.url, `${entry}/${candidate}`, {
        store,
        ...(options.signal ? { signal: options.signal } : {}),
        logger,
        limits,
        onDigest: (digest) => (computed[candidate] = digest),
        ...(asset.headers ? { headers: asset.headers } : {}),
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
    const bytes = await fetchOptional(asset.url, `${entryName(key.name, key.computeType)}/${name}`, store, context, asset.headers);
    if (bytes) files.set(name, bytes);
  }

  await writeDigests(store, context.digests);

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
