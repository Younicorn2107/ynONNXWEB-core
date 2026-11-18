/**
 * Room in the store for the next model: the least recently used models that no session holds are
 * evicted first.
 */

import { QuotaError } from "../errors.js";
import type { Logger } from "../logger.js";
import { readStorageBudget } from "../backends/memory.js";
import type { AssetStore } from "./store.js";

const LRU_PREFIX = "__ynonnx_lru/";

/** Entries (model folders in the store) a loaded bundle holds, with how many hold each. */
const held = new Map<string, number>();

export function holdEntries(entries: Iterable<string>): void {
  for (const e of entries) held.set(e, (held.get(e) ?? 0) + 1);
}

export function releaseEntries(entries: Iterable<string>): void {
  for (const e of entries) {
    const n = (held.get(e) ?? 1) - 1;
    if (n <= 0) held.delete(e);
    else held.set(e, n);
  }
}

export function heldEntries(): ReadonlySet<string> {
  return new Set(held.keys());
}

/** The model folder of a store key: what is before its first `/`. */
export function entryOf(key: string): string {
  const at = key.indexOf("/");
  return at < 0 ? key : key.slice(0, at);
}

/** Record that these entries were just used. */
export async function touchEntries(store: AssetStore, entries: Iterable<string>): Promise<void> {
  const now = String(Date.now());
  await Promise.all([...entries].map((e) => store.write(`${LRU_PREFIX}${e}`, new Response(now)).catch(() => undefined)));
}

async function lastUse(store: AssetStore, entry: string): Promise<number> {
  try {
    const bytes = await store.read(`${LRU_PREFIX}${entry}`);
    return bytes ? Number(new TextDecoder().decode(bytes)) || 0 : 0;
  } catch {
    return 0;
  }
}

export interface Eviction {
  entry: string;
  bytes: number;
}

/**
 * Make room for `neededBytes` more (with a 10 % margin), evicting the least recently used models
 * not in `protect` and not held by a live bundle. Throws `QuotaError` when even that is not enough
 * — decided BEFORE anything is deleted: the size comes from a manifest or a listing, and one
 * declaring a terabyte must not cost the user every model they had stored on its way to failing.
 */
export async function ensureRoom(store: AssetStore, neededBytes: number, protect: ReadonlySet<string>, logger?: Logger): Promise<Eviction[]> {
  if (!Number.isFinite(neededBytes) || neededBytes < 0) {
    throw new QuotaError(`${String(neededBytes)} is not a number of bytes to make room for`, { details: { neededBytes } });
  }
  const needed = neededBytes * 1.1;
  let budget = await readStorageBudget();
  if (budget.availableBytes === null || budget.availableBytes >= needed) return [];
  const mb = (n: number) => `${Math.round(n / 1e6)} MB`;
  if (budget.quotaBytes !== null && needed > budget.quotaBytes) {
    throw new QuotaError(
      `not enough storage: this needs about ${mb(needed)}, more than the origin's whole quota of ${mb(budget.quotaBytes)}; nothing was evicted.`,
      { details: { neededBytes: needed, availableBytes: budget.availableBytes, quotaBytes: budget.quotaBytes, evicted: [] } },
    );
  }

  const keys = await store.keys();
  const byEntry = new Map<string, string[]>();
  for (const key of keys) {
    if (key.startsWith(LRU_PREFIX)) continue;
    const entry = entryOf(key);
    if (protect.has(entry) || held.has(entry)) continue;
    byEntry.set(entry, [...(byEntry.get(entry) ?? []), key]);
  }
  const candidates = await Promise.all(
    [...byEntry].map(async ([entry, entryKeys]) => ({
      entry,
      keys: entryKeys,
      used: await lastUse(store, entry),
      bytes: (await Promise.all(entryKeys.map(async (k) => (await store.stat(k))?.bytes ?? 0))).reduce((a, b) => a + b, 0),
    })),
  );
  candidates.sort((a, b) => a.used - b.used);
  const evictable = candidates.reduce((sum, c) => sum + c.bytes, 0);
  if (budget.availableBytes + evictable < needed) {
    throw new QuotaError(
      `not enough storage: this needs about ${mb(needed)} more and the origin has ${mb(budget.availableBytes)} free of ` +
        `${mb(budget.quotaBytes ?? 0)}; evicting every unused model would free only ${mb(evictable)}, so none was. ` +
        `The models kept are in use (${[...held.keys()].join(", ") || "none"}); close one, or let the browser grant ` +
        `more space (persistent storage, free disk).`,
      { details: { neededBytes: needed, availableBytes: budget.availableBytes, quotaBytes: budget.quotaBytes, evicted: [] } },
    );
  }

  const evicted: Eviction[] = [];
  let freed = 0;
  for (const c of candidates) {
    if (budget.availableBytes + freed >= needed) break;
    for (const k of c.keys) await store.delete(k);
    await store.delete(`${LRU_PREFIX}${c.entry}`);
    freed += c.bytes;
    evicted.push({ entry: c.entry, bytes: c.bytes });
    logger?.info(`evicted ${c.entry} (${Math.round(c.bytes / 1e6)} MB, unused since ${c.used ? new Date(c.used).toISOString() : "unknown"}) to make room`);
  }
  budget = await readStorageBudget();
  const available = budget.availableBytes ?? Infinity;
  if (available < needed) {
    throw new QuotaError(
      `not enough storage: this needs about ${mb(needed)} more and the origin has ${mb(available)} free of ` +
        `${mb(budget.quotaBytes ?? 0)}, after evicting ${evicted.length} unused model(s). The models still kept ` +
        `are in use (${[...held.keys()].join(", ") || "none"}); close one, or let the browser grant more space ` +
        `(persistent storage, free disk).`,
      { details: { neededBytes: needed, availableBytes: available, quotaBytes: budget.quotaBytes, evicted: evicted.map((e) => e.entry) } },
    );
  }
  return evicted;
}
