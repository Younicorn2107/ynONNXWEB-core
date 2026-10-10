/** Where a model's files come from. */

import { entryName } from "./entry.js";
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

function joinUrl(base: string, ...parts: string[]): string {
  const trimmed = base.endsWith("/") ? base.slice(0, -1) : base;
  return [trimmed, ...parts.map((p) => p.replace(/^\//, ""))].join("/");
}

export interface ServerResolverOptions {
  /** Mount point of the server's model views. */
  base?: string;
  headers?: Record<string, string>;
  /**
   * The server makes the models it lacks (`{base}prepare?name=&computeType=`, answered when done):
   * a model not in its cache is exported there, then loaded. Off by default — a plain cache answers
   * 404.
   */
  prepare?: boolean;
}

/** Files served by a server from the Python cache layout. */
export function serverResolver(options: ServerResolverOptions = {}): ModelResolver {
  const base = options.base ?? "/ynonnx/models/";
  return {
    id: `server:${base}`,
    manifestUrls(key) {
      return [MANIFEST_FILE, PUBLISHED_MANIFEST_FILE].map((name) =>
        joinUrl(base, entryDirName(key), name),
      );
    },
    assetUrl(key, filename) {
      return {
        url: joinUrl(base, entryDirName(key), filename),
        filename,
        ...(options.headers ? { headers: options.headers } : {}),
      };
    },
    ...(options.prepare
      ? {
          async prepare(key: ModelKey, signal?: AbortSignal) {
            const query = new URLSearchParams({ name: key.name, computeType: key.computeType });
            try {
              const response = await fetch(`${joinUrl(base, "prepare")}?${query}`, {
                cache: "no-store",
                ...(signal ? { signal } : {}),
                ...(options.headers ? { headers: options.headers } : {}),
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
    const items = (await response.json()) as { type?: string; path?: string; size?: number; lfs?: { size?: number } }[];
    return items
      .filter((item) => item.type === "file" && typeof item.path === "string")
      .map((item) => ({
        name: item.path!.split("/").pop()!,
        size: item.lfs?.size ?? item.size ?? null,
      }));
  }

  return {
    id: `huggingface:${options.repo}`,
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
      const path = folder !== null && place === "graphs" ? `${folder}/${filename}` : filename;
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
