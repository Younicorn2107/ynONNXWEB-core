/** Where a model's files come from. */

import { ManifestError } from "../errors.js";
import { entryName } from "./entry.js";
import { isSize } from "./limits.js";
import type { GraphManifest } from "../types/manifest.js";

/** The manifest ynONNX writes beside a converted entry. */
export const MANIFEST_FILE = "ynonnx_manifest.json";

/** The manifest name inside a PUBLISHED repository. */
export const PUBLISHED_MANIFEST_FILE = "onnx_manifest.json";

/** The manifest for one precision in a flat repository. */
export function manifestFileFor(computeType: string): string {
  return `ynonnx_manifest-${computeType}.json`;
}

export interface ModelKey {
  /** Alias or `org/repo`, as the Python catalogue spells it. */
  name: string;
  computeType: string;
}

/** One file a source can list, with its size when the source says it. */
export interface ListedFile {
  name: string;
  size: number | null;
}

/** What a source holds for a model, when it can list it. */
export interface ModelListing {
  /** The graph folder: graphs and their sidecars. */
  graphs: ListedFile[];
  /** The model root: tokenizer and configs. Same as `graphs` for a flat source. */
  root: ListedFile[];
}

/** Where a file sits in a published repository: beside the graphs, or at its root. */
export type AssetPlace = "graphs" | "root";

export interface ResolvedAsset {
  url: string;
  filename: string;
  headers?: Record<string, string>;
}

export interface ModelResolver {
  /** A label for logs and errors — which source answered. */
  readonly id: string;
  /**
   * What the bytes come from: two resolvers with the same `source` serve the same file for the same
   * key (host, repository, revision, folder). The store records it beside each entry, and an entry
   * stored from another source is fetched again rather than reused. `id` when absent.
   */
  readonly source?: string;
  /** Sent with every request this source is asked, manifests included (a bearer token). */
  readonly headers?: Readonly<Record<string, string>>;
  /**
   * Where the manifest may live, most likely first; empty when this source publishes none. Several,
   * because a server cache and a published folder name it differently and both are read, as on the
   * Python side.
   */
  manifestUrls(key: ModelKey): string[];
  assetUrl(key: ModelKey, filename: string, place?: AssetPlace): ResolvedAsset;
  /**
   * The files this source holds for a model, for one that publishes no manifest: the loader then
   * picks the graphs by the published naming convention and describes them itself. Absent when the
   * source cannot list.
   */
  listFiles?(key: ModelKey, signal?: AbortSignal): Promise<ModelListing | null>;
  /**
   * Ask the source to make a model it does not have yet (a server exporting it on demand), and
   * wait: whether it has it now, and why not. Called by the loader when no manifest was found;
   * absent when the source cannot.
   */
  prepare?(key: ModelKey, signal?: AbortSignal): Promise<{ ready: boolean; error?: string }>;
}

/** `org/repo` → `org--repo`, matching the Python `safe_name` / `entry_dir`. */
export function entryDirName(key: ModelKey): string {
  return entryName(key.name, key.computeType);
}

/** What `loadModel` records as an entry's source. */
export function sourceOf(resolver: ModelResolver): string {
  return resolver.source ?? resolver.id;
}

function joinUrl(base: string, ...parts: string[]): string {
  const trimmed = base.endsWith("/") ? base.slice(0, -1) : base;
  return [trimmed, ...parts.map((p) => p.replace(/^\//, ""))].join("/");
}

/**
 * A path below a model's folder, made of names a manifest, a graph or a listing chose: each segment
 * percent-encoded, and a dot segment refused — a URL parser resolves `entry/../other/model.onnx` to
 * another model's folder (with `serverResolver`, a same-origin request carrying the page's
 * cookies). A `/` stays a separator: a caller may ask for `tokenizer/tokenizer.json`.
 */
function pathOf(path: string): string {
  const parts = path.replace(/^\//, "").split("/");
  for (const part of parts) {
    if (part === "" || part === "." || part === ".." || part.includes("\\")) {
      throw new ManifestError(`'${path}' is not a usable file path (empty, dot or backslash segment)`);
    }
  }
  return parts.map(encodeURIComponent).join("/");
}

export interface ServerResolverOptions {
  /** Mount point of the server's model views. */
  base?: string;
  headers?: Record<string, string>;
  /**
   * The server makes the models it lacks (`POST {base}prepare` with `{"name", "computeType"}` as
   * JSON, answered when done): a model not in its cache is exported there, then loaded. Off by
   * default — a plain cache answers 404.
   */
  prepare?: boolean;
}

/** Files served by a server from the Python cache layout. */
export function serverResolver(options: ServerResolverOptions = {}): ModelResolver {
  const base = options.base ?? "/ynonnx/models/";
  return {
    id: `server:${base}`,
    ...(options.headers ? { headers: options.headers } : {}),
    manifestUrls(key) {
      return [MANIFEST_FILE, PUBLISHED_MANIFEST_FILE].map((name) =>
        joinUrl(base, pathOf(entryDirName(key)), name),
      );
    },
    assetUrl(key, filename) {
      return {
        url: joinUrl(base, pathOf(entryDirName(key)), pathOf(filename)),
        filename,
        ...(options.headers ? { headers: options.headers } : {}),
      };
    },
    ...(options.prepare
      ? {
          async prepare(key: ModelKey, signal?: AbortSignal) {
            try {
              const response = await fetch(joinUrl(base, "prepare"), {
                method: "POST",
                cache: "no-store",
                body: JSON.stringify({ name: key.name, computeType: key.computeType }),
                ...(signal ? { signal } : {}),
                headers: { ...(options.headers ?? {}), "Content-Type": "application/json" },
              });
              const answer = (await response.json().catch(() => ({}))) as { ready?: boolean; error?: string };
              return answer.ready ? { ready: true } : { ready: false, error: answer.error ?? `HTTP ${response.status}` };
            } catch (cause) {
              if (signal?.aborted) throw cause;
              return { ready: false, error: cause instanceof Error ? cause.message : String(cause) };
            }
          },
        }
      : {}),
  };
}

export interface HuggingFaceResolverOptions {
  /** `org/repo` on the Hub. */
  repo: string;
  /** Subfolder holding the graphs, usually `onnx`. */
  subfolder?: string;
  revision?: string;
  endpoint?: string;
  headers?: Record<string, string>;
  /**
   * Whether the repo publishes a manifest. True for repositories ynONNX produced, false for
   * everyone else's.
   */
  manifest?: boolean;
  /**
   * Flat layout: every precision in one folder, told apart by a suffix (`decoder_merged-int4.onnx`,
   * `ynonnx_manifest-int4.json`).
   */
  flat?: boolean;
  /** One self-contained folder per variant: `onnx/grain-int4-b64/`. */
  variants?: Readonly<Record<string, string>>;
}

/** Straight from the Hub. */
/** The folder for one precision, or `null` when this repo is not laid out that way. */
function variantFolder(
  options: HuggingFaceResolverOptions,
  computeType: string,
): string | null {
  const named = options.variants?.[computeType];
  return named ?? null;
}

export function huggingFaceResolver(options: HuggingFaceResolverOptions): ModelResolver {
  const endpoint = options.endpoint ?? "https://huggingface.co";
  const revision = options.revision ?? "main";

  async function tree(path: string, signal?: AbortSignal): Promise<ListedFile[] | null> {
    const parts = ["api", "models", options.repo, "tree", revision, ...(path ? [path] : [])];
    const url = joinUrl(endpoint, ...parts);
    const response = await fetch(url, {
      ...(signal ? { signal } : {}),
      ...(options.headers ? { headers: options.headers } : {}),
    });
    if (!response.ok) return null;
    const items = (await response.json()) as unknown;
    if (!Array.isArray(items)) return null;
    return (items as { type?: unknown; path?: unknown; size?: unknown; lfs?: { size?: unknown } | null }[])
      .filter((item) => item !== null && typeof item === "object" && item.type === "file" && typeof item.path === "string")
      .map((item) => {
        const size = item.lfs?.size ?? item.size;
        return { name: (item.path as string).split("/").pop()!, size: isSize(size) ? size : null };
      });
  }

  return {
    id: `huggingface:${options.repo}`,

    source:
      `huggingface:${endpoint}/${options.repo}@${revision}` +
      (options.subfolder ? `/${options.subfolder}` : "") +
      (options.variants ? ` ${JSON.stringify(options.variants)}` : ""),
    ...(options.headers ? { headers: options.headers } : {}),
    manifestUrls(key) {
      if (options.manifest === false) return [];
      const folder = variantFolder(options, key.computeType);
      if (folder !== null) {
        return [
          joinUrl(endpoint, options.repo, "resolve", revision, `${folder}/${PUBLISHED_MANIFEST_FILE}`),
        ];
      }
      const file =
        options.flat === false ? PUBLISHED_MANIFEST_FILE : manifestFileFor(key.computeType);
      const path = options.subfolder ? `${options.subfolder}/${file}` : file;
      return [joinUrl(endpoint, options.repo, "resolve", revision, path)];
    },
    assetUrl(key, filename, place = "graphs") {
      const folder = variantFolder(options, key.computeType) ?? options.subfolder ?? null;
      const path = folder !== null && place === "graphs" ? `${folder}/${pathOf(filename)}` : pathOf(filename);
      return {
        url: joinUrl(endpoint, options.repo, "resolve", revision, path),
        filename,
        ...(options.headers ? { headers: options.headers } : {}),
      };
    },
    async listFiles(key, signal) {
      const folder = variantFolder(options, key.computeType) ?? options.subfolder ?? "";
      const graphs = await tree(folder, signal);
      if (graphs === null) return null;
      const root = folder === "" ? graphs : ((await tree("", signal)) ?? []);
      return { graphs, root };
    },
  };
}

/** Every file a manifest says must be fetched, sidecars included. */
export function filesFromManifest(manifest: GraphManifest): string[] {
  const files: string[] = [];
  for (const graph of manifest.graphs) {
    files.push(graph.filename);
    files.push(...graph.externalData);
  }
  if (manifest.tokenizerFile) files.push(manifest.tokenizerFile);
  return [...new Set(files)];
}

/** The suffixes an external-data sidecar conventionally takes. */
export function externalDataCandidates(graphFilename: string): string[] {
  const stem = graphFilename.replace(/\.onnx$/i, "");
  return [`${graphFilename}_data`, `${graphFilename}.data`, `${stem}.data`];
}

/**
 * `base` seen with one of its graphs as the model's own: the files named `<stem><suffix>.onnx` (and
 * their weight files) listed and fetched as `model<suffix>.onnx`. For repositories that ship
 * several graphs side by side — CLIP's `text_model` and `vision_model`, a hybrid language model's
 * `embed_tokens` and `decoder_model_merged` — each loaded as a model of its own.
 */
export function graphAlias(base: ModelResolver, stem: string): ModelResolver {
  const toAlias = (name: string) => (name.startsWith(stem) ? `model${name.slice(stem.length)}` : name);
  const toReal = (name: string) => (name.startsWith("model") && !name.startsWith(stem) ? `${stem}${name.slice("model".length)}` : name);
  return {
    id: `${base.id}#${stem}`,
    source: `${sourceOf(base)}#${stem}`,
    ...(base.headers ? { headers: base.headers } : {}),
    manifestUrls: () => [],
    assetUrl: (key, filename, place) => ({ ...base.assetUrl(key, toReal(filename), place), filename }),
    listFiles: async (key, signal) => {
      const listing = await base.listFiles?.(key, signal);
      if (!listing) return null;
      const graphs = listing.graphs.filter((f) => f.name.startsWith(stem)).map((f) => ({ ...f, name: toAlias(f.name) }));
      return { graphs, root: listing.root };
    },
  };
}
