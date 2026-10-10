/** Fetching a model file, in slices when the server allows it. */

import { DownloadError, IntegrityError } from "../errors.js";
import type { Logger } from "../logger.js";
import { withLock, type AssetStore } from "./store.js";

/** Matching the Python downloader's constants rather than inventing new ones. */
export const MAX_RETRIES = 3;
export const BACKOFF_BASE_MS = 1000;
export const SLICE_BYTES = 16 * 1024 * 1024;
export const MAX_IN_FLIGHT = 3;

export interface FetchProgress {
  file: string;
  loaded: number;
  total: number | null;
  overallLoaded: number;
  overallTotal: number | null;
  phase: "probing" | "downloading" | "assembling" | "cached";
}

export interface FetchOptions {
  store: AssetStore;
  signal?: AbortSignal;
  onProgress?: (progress: FetchProgress) => void;
  logger?: Logger;
  headers?: Record<string, string>;
  /** Overall figures, so one file's bar can show the whole job's position. */
  overall?: { loaded: number; total: number | null };
  /** Skip the HEAD and stream in one go. */
  noRange?: boolean;
  /** Bytes per ranged slice. Defaults to {@link SLICE_BYTES}. */
  sliceBytes?: number;
  /**
   * Only these bytes of the file, `[start, end)`: the asset is that part, a buffer of its own. How
   * a weight file larger than one browser buffer is taken in pieces (`splitExternalData`).
   */
  range?: readonly [number, number];
}

interface HeadInfo {
  length: number | null;
  acceptsRanges: boolean;
  etag: string | null;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function throwIfAborted(signal: AbortSignal | undefined, url: string): void {
  if (signal?.aborted) {
    throw new DownloadError(`the download of ${url} was cancelled`);
  }
}

async function head(url: string, options: FetchOptions): Promise<HeadInfo> {
  try {
    const response = await fetch(url, {
      method: "HEAD",

      cache: "no-store",
      ...(options.signal ? { signal: options.signal } : {}),
      ...(options.headers ? { headers: options.headers } : {}),
    });
    if (!response.ok) {
      return { length: null, acceptsRanges: false, etag: null };
    }
    const length = response.headers.get("content-length");
    return {
      length: length ? Number(length) : null,
      acceptsRanges: (response.headers.get("accept-ranges") ?? "").toLowerCase().includes("bytes"),
      etag: response.headers.get("etag"),
    };
  } catch {
    return { length: null, acceptsRanges: false, etag: null };
  }
}

/** One attempt at one range, with retries. */
async function fetchSlice(
  url: string,
  start: number,
  end: number,
  options: FetchOptions,
): Promise<ArrayBuffer> {
  let lastError: unknown;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt += 1) {
    throwIfAborted(options.signal, url);
    try {
      const response = await fetch(url, {
        cache: "no-store",
        ...(options.signal ? { signal: options.signal } : {}),
        headers: { ...(options.headers ?? {}), Range: `bytes=${start}-${end}` },
      });

      if (response.status === 200) {
        throw new DownloadError(
          `${url} ignored a Range request and returned the whole body; ` +
            `ranged downloads are not usable against this server`,
        );
      }
      if (!response.ok) {
        throw new DownloadError(`${url} returned ${response.status} for bytes ${start}-${end}`);
      }
      return await response.arrayBuffer();
    } catch (error) {
      lastError = error;
      if (options.signal?.aborted) throw error;
      if (attempt < MAX_RETRIES - 1) {
        const wait = BACKOFF_BASE_MS * 2 ** attempt;
        options.logger?.debug(`retrying ${url} bytes ${start}-${end} in ${wait} ms`);
        await sleep(wait);
      }
    }
  }
  throw new DownloadError(
    `could not fetch bytes ${start}-${end} of ${url} after ${MAX_RETRIES} attempts: ` +
      `${lastError instanceof Error ? lastError.message : String(lastError)}`,
    { cause: lastError },
  );
}

function partKey(key: string, index: number, etag: string | null): string {
  return `${key}.part${index}${etag ? `.${etag.replace(/["\W]+/g, "")}` : ""}`;
}

/** Download `url` into the store under `key`, resuming where possible. */
export async function fetchAsset(
  url: string,
  key: string,
  options: FetchOptions,
): Promise<ArrayBuffer> {
  const { store, onProgress, logger } = options;
  const overallLoaded = options.overall?.loaded ?? 0;
  const overallTotal = options.overall?.total ?? null;

  const report = (phase: FetchProgress["phase"], loaded: number, total: number | null): void => {
    onProgress?.({
      file: key,
      loaded,
      total,
      overallLoaded: overallLoaded + loaded,
      overallTotal,
      phase,
    });
  };

  return withLock(key, async () => {
    const existing = await store.read(key);
    if (existing) {
      report("cached", existing.byteLength, existing.byteLength);
      logger?.debug(`${key} is already stored (${existing.byteLength} bytes)`);
      return existing;
    }

    report("probing", 0, null);
    const info = options.noRange === true
      ? { length: null, acceptsRanges: false, etag: null }
      : await head(url, options);

    const base = options.range?.[0] ?? 0;
    if (options.range) {
      if (!info.acceptsRanges || info.length === null) throw new DownloadError(`${url} does not accept ranges: it cannot be taken in parts`);
      info.length = options.range[1] - options.range[0];
    }

    if (!info.acceptsRanges || info.length === null) {
      logger?.debug(`${url} does not accept ranges; streaming without resume`);
      const buffer = await streamWhole(url, key, info.length, options, report);
      await cacheOrWarn(
        store,
        key,
        new Response(buffer, { headers: contentHeaders(buffer.byteLength, info.etag) }),
        logger,
      );
      return buffer;
    }

    const total = info.length;
    const sliceSize = options.sliceBytes ?? SLICE_BYTES;
    const sliceCount = Math.max(1, Math.ceil(total / sliceSize));

    const out = new Uint8Array(total);
    const filled: boolean[] = new Array(sliceCount).fill(false);
    const place = (index: number, part: ArrayBuffer): boolean => {
      const start = index * sliceSize;
      const expected = Math.min(start + sliceSize, total) - start;
      if (part.byteLength !== expected) return false;
      out.set(new Uint8Array(part), start);
      filled[index] = true;
      return true;
    };

    let loaded = 0;
    for (let index = 0; index < sliceCount; index += 1) {
      const stored = await store.read(partKey(key, index, info.etag));
      if (stored && place(index, stored)) loaded += stored.byteLength;
    }
    if (loaded > 0) {
      logger?.info(
        `resuming ${key}: ${Math.round((loaded / total) * 100)}% was already downloaded`,
      );
    }
    report("downloading", loaded, total);

    let cursor = 0;
    const worker = async (): Promise<void> => {
      while (cursor < sliceCount) {
        const index = cursor;
        cursor += 1;
        if (filled[index]) continue;
        const start = index * sliceSize;
        const end = Math.min(start + sliceSize, total) - 1;
        const slice = await fetchSlice(url, base + start, base + end, options);
        if (slice.byteLength !== end - start + 1) {
          throw new IntegrityError(`${url} answered ${slice.byteLength} bytes for the range ${start}-${end}`);
        }
        place(index, slice);
        loaded += slice.byteLength;

        await cacheOrWarn(
          store,
          partKey(key, index, info.etag),
          new Response(slice, { headers: contentHeaders(slice.byteLength, info.etag) }),
          logger,
        );
        report("downloading", loaded, total);
      }
    };
    let assembled: ArrayBuffer;
    try {
      await Promise.all(
        Array.from({ length: Math.min(MAX_IN_FLIGHT, sliceCount) }, () => worker()),
      );
      report("assembling", total, total);
      const missing = filled.indexOf(false);
      if (missing >= 0) throw new DownloadError(`slice ${missing} of ${url} is missing after the download completed`);
      assembled = out.buffer;
    } catch (error) {
      if (!(error instanceof IntegrityError)) throw error;

      logger?.warn(`${error.message}; downloading ${key} again in one piece`);
      for (let index = 0; index < sliceCount; index += 1) await store.delete(partKey(key, index, info.etag)).catch(() => false);
      const whole = await streamWhole(url, key, info.length, options, report);
      if (whole.byteLength !== total) throw new IntegrityError(`${url} sent ${whole.byteLength} bytes but declared ${total}`);
      await cacheOrWarn(store, key, new Response(whole, { headers: contentHeaders(whole.byteLength, info.etag) }), logger);
      return whole;
    }

    const cached = await cacheOrWarn(
      store,
      key,
      new Response(assembled, { headers: contentHeaders(assembled.byteLength, info.etag) }),
      logger,
    );

    if (cached) {
      for (let index = 0; index < sliceCount; index += 1) {
        await store.delete(partKey(key, index, info.etag));
      }
    }
    return assembled;
  });
}

/** Store it, or say why it could not be stored, and carry on. */
async function cacheOrWarn(
  store: AssetStore,
  key: string,
  response: Response,
  logger?: Logger,
): Promise<boolean> {
  try {
    await store.write(key, response);
    return true;
  } catch (error) {
    logger?.warn(
      `'${key}' downloaded but could not be cached ` +
        `(${error instanceof Error ? error.message : String(error)}). ` +
        `It will work now and be downloaded again next time. Free storage, ` +
        `or clear cached models, to keep it.`,
    );
    return false;
  }
}

function contentHeaders(length: number, etag: string | null): Record<string, string> {
  return {
    "content-length": String(length),
    "content-type": "application/octet-stream",
    ...(etag ? { etag } : {}),
  };
}

async function streamWhole(
  url: string,
  key: string,
  total: number | null,
  options: FetchOptions,
  report: (phase: FetchProgress["phase"], loaded: number, total: number | null) => void,
): Promise<ArrayBuffer> {
  const response = await fetch(url, {
    cache: "no-store",
    ...(options.signal ? { signal: options.signal } : {}),
    ...(options.headers ? { headers: options.headers } : {}),
  });
  if (!response.ok) {
    throw new DownloadError(`${url} returned ${response.status} for ${key}`);
  }
  const declared = total ?? (Number(response.headers.get("content-length") ?? 0) || null);

  if (!response.body) {
    const buffer = await response.arrayBuffer();
    report("downloading", buffer.byteLength, declared);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    throwIfAborted(options.signal, url);
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      report("downloading", loaded, declared);
    }
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}
