/** Runs that survive a reload, and results that are not computed twice. */

import { CheckpointError } from "../errors.js";
import { combineIds, contentId, hashParams } from "./hashing.js";

/** A small asynchronous key-value store. */
export interface KeyValueStore {
  get(key: string): Promise<unknown>;
  set(key: string, value: unknown): Promise<void>;
  delete(key: string): Promise<void>;
  keys(prefix?: string): Promise<string[]>;
}

/** In memory: for tests, and for a page without IndexedDB. */
export class MemoryKeyValueStore implements KeyValueStore {
  readonly #map = new Map<string, unknown>();
  async get(key: string): Promise<unknown> {
    return structuredClone(this.#map.get(key));
  }
  async set(key: string, value: unknown): Promise<void> {
    this.#map.set(key, structuredClone(value));
  }
  async delete(key: string): Promise<void> {
    this.#map.delete(key);
  }
  async keys(prefix = ""): Promise<string[]> {
    return [...this.#map.keys()].filter((k) => k.startsWith(prefix));
  }
}

/** IndexedDB, one object store, structured-clone values (typed arrays included). */
export class IndexedDbStore implements KeyValueStore {
  #db: Promise<IDBDatabase> | null = null;

  constructor(
    readonly database = "ynonnxweb",
    readonly storeName = "state",
  ) {}

  #open(): Promise<IDBDatabase> {
    this.#db ??= new Promise((resolve, reject) => {
      const idb = (globalThis as { indexedDB?: IDBFactory }).indexedDB;
      if (!idb) {
        reject(new CheckpointError("IndexedDB is unavailable here; use a MemoryKeyValueStore"));
        return;
      }
      const request = idb.open(this.database, 1);
      request.onupgradeneeded = () => request.result.createObjectStore(this.storeName);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(new CheckpointError(`IndexedDB would not open: ${String(request.error)}`));
    });
    return this.#db;
  }

  async #request<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const db = await this.#open();
    return new Promise<T>((resolve, reject) => {
      const request = run(db.transaction(this.storeName, mode).objectStore(this.storeName));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(new CheckpointError(`IndexedDB refused: ${String(request.error)}`));
    });
  }

  get(key: string): Promise<unknown> {
    return this.#request("readonly", (store) => store.get(key));
  }
  async set(key: string, value: unknown): Promise<void> {
    await this.#request("readwrite", (store) => store.put(value, key));
  }
  async delete(key: string): Promise<void> {
    await this.#request("readwrite", (store) => store.delete(key));
  }
  async keys(prefix = ""): Promise<string[]> {
    const all = (await this.#request("readonly", (store) => store.getAllKeys())) as IDBValidKey[];
    return all.map(String).filter((k) => k.startsWith(prefix));
  }
}

/** Finished results shared across runs, keyed by item and settings. */
export class ResultCache {
  constructor(
    readonly store: KeyValueStore,
    readonly namespace = "results",
  ) {}
  get(key: string): Promise<unknown> {
    return this.store.get(`${this.namespace}/${key}`);
  }
  set(key: string, value: unknown): Promise<void> {
    return this.store.set(`${this.namespace}/${key}`, value);
  }
  /** Forget every cached result of this namespace. */
  async clear(): Promise<void> {
    for (const key of await this.store.keys(`${this.namespace}/`)) await this.store.delete(key);
  }
}

/**
 * A journal is a header (the fingerprint) and one entry per finished item, so recording item 10 000
 * writes one entry, not the 9 999 before it.
 */
interface JournalHeader {
  fingerprint: string;
}

export interface OpenRunOptions {
  /** null/false: nothing kept; true: named after the inputs; a string: that name. */
  checkpoint?: boolean | string | null;
  inputs: unknown;
  /** What changes the output: model, precision, options. */
  settings?: Record<string, unknown>;
  /** Where the journal lives. IndexedDB by default. */
  store?: KeyValueStore;
  /** Answer from, and store into, this cache; `inputs` must then be a list. */
  cache?: ResultCache | null;
  prefix?: string;
}

export class ResumableRun {
  readonly #done = new Map<number, unknown>();
  resumed = 0;
  cached = 0;

  private constructor(
    readonly store: KeyValueStore | null,
    readonly journalKey: string | null,
    readonly fingerprint: string,
    readonly cache: ResultCache | null,
    readonly cacheKeys: readonly string[],
    entries: ReadonlyMap<number, unknown>,
  ) {
    for (const [index, result] of entries) this.#done.set(index, result);
    this.resumed = this.#done.size;
  }

  static async open(options: OpenRunOptions): Promise<ResumableRun> {
    const settingsId = await hashParams(options.settings ?? {});
    const inputId = await contentId(options.inputs);
    const fingerprint = await combineIds([inputId, settingsId]);
    const cache = options.cache ?? null;
    const cacheKeys =
      cache && Array.isArray(options.inputs)
        ? await Promise.all(options.inputs.map(async (item) => combineIds([await contentId(item), settingsId])))
        : [];
    const wanted = options.checkpoint;
    if (wanted === null || wanted === undefined || wanted === false) {
      return new ResumableRun(null, null, fingerprint, cache, cacheKeys, new Map());
    }
    const store = options.store ?? new IndexedDbStore();
    const journalKey = `checkpoints/${typeof wanted === "string" ? wanted : `${options.prefix ?? "checkpoint"}-${inputId}`}`;
    const header = (await store.get(journalKey)) as JournalHeader | undefined;
    const entries = new Map<number, unknown>();
    if (header && header.fingerprint === fingerprint) {
      for (const key of await store.keys(`${journalKey}/`)) {
        entries.set(Number(key.slice(journalKey.length + 1)), await store.get(key));
      }
    } else {
      for (const key of await store.keys(`${journalKey}/`)) await store.delete(key);
      await store.set(journalKey, { fingerprint } satisfies JournalHeader);
    }
    return new ResumableRun(store, journalKey, fingerprint, cache, cacheKeys, entries);
  }

  get enabled(): boolean {
    return this.journalKey !== null;
  }

  get size(): number {
    return this.#done.size;
  }

  /** Whether the item is done, from the journal or the cache. */
  async has(index: number): Promise<boolean> {
    if (this.#done.has(index)) return true;
    const key = this.cacheKeys[index];
    if (!this.cache || key === undefined) return false;
    const stored = await this.cache.get(key);
    if (stored === undefined) return false;
    this.#done.set(index, stored);
    this.cached += 1;
    return true;
  }

  get(index: number): unknown {
    return this.#done.get(index);
  }

  /** Keep one finished item, in the journal and the cache. */
  async record(index: number, result: unknown): Promise<void> {
    this.#done.set(index, result);
    const key = this.cacheKeys[index];
    if (this.cache && key !== undefined) await this.cache.set(key, result);
    if (this.store && this.journalKey) await this.store.set(`${this.journalKey}/${index}`, result);
  }

  /** The run is over: its journal goes. */
  async complete(): Promise<void> {
    if (!this.store || !this.journalKey) return;
    for (const key of await this.store.keys(`${this.journalKey}/`)) await this.store.delete(key);
    await this.store.delete(this.journalKey);
  }
}
