/** Where fetched model files live between page loads. */

import { StorageError } from "../errors.js";

export interface StoredAsset {
  key: string;
  bytes: number;
  etag: string | null;
}

export interface AssetStore {
  readonly id: string;
  has(key: string): Promise<boolean>;
  /** The stored bytes, or null when absent. */
  read(key: string): Promise<ArrayBuffer | null>;
  /** Metadata without materializing the body — for resume decisions. */
  stat(key: string): Promise<StoredAsset | null>;
  write(key: string, response: Response): Promise<void>;
  delete(key: string): Promise<boolean>;
  /** Every key held, for eviction and for a "clear the cache" button. */
  keys(): Promise<string[]>;
  clear(): Promise<void>;
}

const CACHE_NAME = "ynonnxweb-models-v1";

/**
 * The Cache API needs URL-shaped keys. A synthetic origin keeps our entries from ever colliding
 * with a real request the page made — and makes them obvious in devtools.
 */
function cacheKey(key: string): string {
  return `https://ynonnxweb.invalid/${key.replace(/^\//, "")}`;
}

export class CacheApiStore implements AssetStore {
  readonly id = "cache-api";

  constructor(private readonly cacheName: string = CACHE_NAME) {}

  private async cache(): Promise<Cache> {
    const caches = (globalThis as { caches?: CacheStorage }).caches;
    if (!caches) {
      throw new StorageError(
        "the Cache API is unavailable, so models cannot be stored between page " +
          "loads. It needs a secure context (https, or http://localhost).",
      );
    }
    return caches.open(this.cacheName);
  }

  async has(key: string): Promise<boolean> {
    const cache = await this.cache();
    return (await cache.match(cacheKey(key))) !== undefined;
  }

  async read(key: string): Promise<ArrayBuffer | null> {
    const cache = await this.cache();
    const response = await cache.match(cacheKey(key));
    if (!response) return null;
    return response.arrayBuffer();
  }

  async stat(key: string): Promise<StoredAsset | null> {
    const cache = await this.cache();
    const response = await cache.match(cacheKey(key));
    if (!response) return null;
    const length = response.headers.get("content-length");
    return {
      key,
      bytes: length ? Number(length) : 0,
      etag: response.headers.get("etag"),
    };
  }

  async write(key: string, response: Response): Promise<void> {
    const cache = await this.cache();
    try {
      await cache.put(cacheKey(key), response);
    } catch (cause) {
      throw new StorageError(
        `could not store '${key}': ${cause instanceof Error ? cause.message : String(cause)}. ` +
          `This is usually the origin's storage quota; free space or clear cached models.`,
        { cause },
      );
    }
  }

  async delete(key: string): Promise<boolean> {
    const cache = await this.cache();
    return cache.delete(cacheKey(key));
  }

  async keys(): Promise<string[]> {
    const cache = await this.cache();
    const requests = await cache.keys();
    return requests.map((request) => new URL(request.url).pathname.replace(/^\//, ""));
  }

  async clear(): Promise<void> {
    const caches = (globalThis as { caches?: CacheStorage }).caches;
    await caches?.delete(this.cacheName);
  }
}

/** The Origin Private File System: one file per key, written as a stream. */
export class OpfsStore implements AssetStore {
  readonly id = "opfs";

  constructor(private readonly rootName: string = "ynonnxweb-models") {}

  static available(): boolean {
    const storage = (globalThis as { navigator?: { storage?: { getDirectory?: unknown } } }).navigator?.storage;
    return typeof storage?.getDirectory === "function";
  }

  async #root(): Promise<FileSystemDirectoryHandle> {
    const storage = (globalThis.navigator as Navigator | undefined)?.storage;
    if (!storage?.getDirectory) {
      throw new StorageError("the Origin Private File System is unavailable here (it needs a secure context)");
    }
    const top = await storage.getDirectory();
    return top.getDirectoryHandle(this.rootName, { create: true });
  }

  /** The directory holding a key, and the file's own name. */
  async #locate(key: string, create: boolean): Promise<{ dir: FileSystemDirectoryHandle; name: string } | null> {
    const parts = key.split("/").filter((p) => p.length > 0 && p !== "." && p !== "..");
    const name = parts.pop();
    if (!name) throw new StorageError(`'${key}' is not a usable key`);
    let dir = await this.#root();
    for (const part of parts) {
      try {
        dir = await dir.getDirectoryHandle(part, { create });
      } catch {
        return null;
      }
    }
    return { dir, name };
  }

  async #file(key: string): Promise<File | null> {
    const at = await this.#locate(key, false);
    if (!at) return null;
    try {
      return await (await at.dir.getFileHandle(at.name)).getFile();
    } catch {
      return null;
    }
  }

  async has(key: string): Promise<boolean> {
    return (await this.#file(key)) !== null;
  }

  async read(key: string): Promise<ArrayBuffer | null> {
    const file = await this.#file(key);
    return file ? file.arrayBuffer() : null;
  }

  async stat(key: string): Promise<StoredAsset | null> {
    const file = await this.#file(key);
    return file ? { key, bytes: file.size, etag: null } : null;
  }

  async write(key: string, response: Response): Promise<void> {
    const at = await this.#locate(key, true);
    if (!at) throw new StorageError(`could not create the folders of '${key}'`);
    const partial = `${at.name}.partial`;
    try {
      const handle = await at.dir.getFileHandle(partial, { create: true });
      const writable = await (handle as FileSystemFileHandle & {
        createWritable(): Promise<FileSystemWritableFileStream>;
      }).createWritable();
      if (response.body) await response.body.pipeTo(writable);
      else {
        await writable.write(await response.arrayBuffer());
        await writable.close();
      }
      const moved = handle as FileSystemFileHandle & { move?(name: string): Promise<void> };
      if (typeof moved.move === "function") {
        await moved.move(at.name);
      } else {
        const final = await at.dir.getFileHandle(at.name, { create: true });
        const out = await (final as FileSystemFileHandle & { createWritable(): Promise<FileSystemWritableFileStream> }).createWritable();
        await (await handle.getFile()).stream().pipeTo(out);
        await at.dir.removeEntry(partial);
      }
    } catch (cause) {
      await at.dir.removeEntry(partial).catch(() => undefined);
      throw new StorageError(
        `could not store '${key}': ${cause instanceof Error ? cause.message : String(cause)}. ` +
          `This is usually the origin's storage quota; free space or clear cached models.`,
        { cause },
      );
    }
  }

  async delete(key: string): Promise<boolean> {
    const at = await this.#locate(key, false);
    if (!at) return false;
    try {
      await at.dir.removeEntry(at.name);
      return true;
    } catch {
      return false;
    }
  }

  async keys(): Promise<string[]> {
    const out: string[] = [];
    const walk = async (dir: FileSystemDirectoryHandle, prefix: string): Promise<void> => {
      for await (const [name, handle] of (dir as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries()) {
        if (handle.kind === "directory") await walk(handle as FileSystemDirectoryHandle, `${prefix}${name}/`);
        else if (!name.endsWith(".partial")) out.push(`${prefix}${name}`);
      }
    };
    await walk(await this.#root(), "");
    return out;
  }

  async clear(): Promise<void> {
    const storage = (globalThis.navigator as Navigator | undefined)?.storage;
    const top = await storage?.getDirectory?.();
    await top?.removeEntry(this.rootName, { recursive: true }).catch(() => undefined);
  }
}

/**
 * The store a page should use: OPFS where the browser has it (no size limit per file beyond the
 * quota, streamed writes), the Cache API otherwise.
 */
export function defaultStore(): AssetStore {
  return OpfsStore.available() ? new OpfsStore() : new CacheApiStore();
}

/** An in-memory store, for tests and for a page that has no Cache API. */
export class MemoryStore implements AssetStore {
  readonly id = "memory";
  private readonly entries = new Map<string, { buffer: ArrayBuffer; etag: string | null }>();

  async has(key: string): Promise<boolean> {
    return this.entries.has(key);
  }

  async read(key: string): Promise<ArrayBuffer | null> {
    return this.entries.get(key)?.buffer ?? null;
  }

  async stat(key: string): Promise<StoredAsset | null> {
    const entry = this.entries.get(key);
    return entry ? { key, bytes: entry.buffer.byteLength, etag: entry.etag } : null;
  }

  async write(key: string, response: Response): Promise<void> {
    const etag = response.headers.get("etag");
    this.entries.set(key, { buffer: await response.arrayBuffer(), etag });
  }

  async delete(key: string): Promise<boolean> {
    return this.entries.delete(key);
  }

  async keys(): Promise<string[]> {
    return [...this.entries.keys()];
  }

  async clear(): Promise<void> {
    this.entries.clear();
  }
}

/**
 * A store read from but never written to: a model too large for the room the origin has (after
 * every unused model was evicted) still loads, from the network, and is downloaded again next time
 * rather than refused.
 */
export class ReadOnlyStore implements AssetStore {
  readonly id: string;

  constructor(private readonly backing: AssetStore) {
    this.id = `${backing.id}:read-only`;
  }

  has(key: string): Promise<boolean> {
    return this.backing.has(key);
  }

  read(key: string): Promise<ArrayBuffer | null> {
    return this.backing.read(key);
  }

  stat(key: string): Promise<StoredAsset | null> {
    return this.backing.stat(key);
  }

  async write(): Promise<void> {
  }

  async delete(): Promise<boolean> {
    return false;
  }

  keys(): Promise<string[]> {
    return this.backing.keys();
  }

  async clear(): Promise<void> {
  }
}

/** Hold a named lock while doing something exclusive. */
export async function withLock<T>(name: string, fn: () => Promise<T>): Promise<T> {
  const locks = (navigator as { locks?: { request<R>(n: string, f: () => Promise<R>): Promise<R> } })
    .locks;
  if (!locks) return fn();
  return locks.request(`ynonnxweb:${name}`, fn);
}
