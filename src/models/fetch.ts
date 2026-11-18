/** Fetching a model file, in slices when the server allows it. */

import { AssetTooLargeError, DownloadError, IntegrityError } from "../errors.js";
import type { Logger } from "../logger.js";
import { DEFAULT_LIMITS, checkSize, isSize, type Limits } from "./limits.js";
import { sha256Hex } from "./sha256.js";
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
  /**
   * The file's SHA-256 (hex) as its manifest declares it. A download is hashed when it lands,
   * before it is stored; a mismatch throws `IntegrityError` and nothing is kept. A copy already in
   * the store is returned as it is: it was checked when it was written (see `loadModel`).
   */
  sha256?: string | null;
  /** Told the SHA-256 of a file just downloaded (never of a stored copy), so it can be recorded. */
  onDigest?: (sha256: string) => void;
  /** The sizes a download may take; `DEFAULT_LIMITS` when absent. */
  limits?: Limits;
}

interface HeadInfo {
  length: number | null;
  acceptsRanges: boolean;
  etag: string | null;
  lastModified: string | null;
}

const NO_HEAD: HeadInfo = { length: null, acceptsRanges: false, etag: null, lastModified: null };

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
    if (!response.ok) return { ...NO_HEAD };
    const length = response.headers.get("content-length");

    const declared = length !== null && /^\d{1,16}$/.test(length.trim()) ? Number(length.trim()) : null;
    return {
      length: declared,
      acceptsRanges: (response.headers.get("accept-ranges") ?? "").toLowerCase().includes("bytes"),
      etag: response.headers.get("etag"),
      lastModified: response.headers.get("last-modified"),
    };
  } catch {
    return { ...NO_HEAD };
  }
}

/** One attempt at one range, with retries. */
async function fetchSlice(
  url: string,
  start: number,
  end: number,
  options: FetchOptions,
  file: { total: number; etag: string | null },
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

      const range = /\/(\d+)\s*$/.exec(response.headers.get("content-range") ?? "");
      if (range && Number(range[1]) !== file.total) {
        throw new IntegrityError(`${url} changed during the download: its size is now ${range[1]} bytes, not ${file.total}`);
      }
      const etag = response.headers.get("etag");
      if (etag && file.etag && etag !== file.etag) {
        throw new IntegrityError(`${url} changed during the download (ETag ${etag}, was ${file.etag})`);
      }
      return await response.arrayBuffer();
    } catch (error) {
      lastError = error;
      if (options.signal?.aborted || error instanceof IntegrityError) throw error;
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

/**
 * Where slice `index` of a download is kept for a resume, or null when it must not be kept at all.
 */
function partKey(key: string, index: number, total: number, validator: string | null): string | null {
  if (!validator) return null;

  let hash = 0x811c9dc5;
  for (let i = 0; i < validator.length; i += 1) hash = Math.imul(hash ^ validator.charCodeAt(i), 0x01000193) >>> 0;
  const token = `${validator.replace(/[^A-Za-z0-9]+/g, "").slice(0, 40)}${hash.toString(16)}`;
  return `${key}.part${index}.${total}.${token}`;
}

/** Download `url` into the store under `key`, resuming where possible. */
export async function fetchAsset(
  url: string,
  key: string,
  options: FetchOptions,
): Promise<ArrayBuffer> {
  const { store, onProgress, logger } = options;
  const limits = options.limits ?? DEFAULT_LIMITS;
  const overallLoaded = options.overall?.loaded ?? 0;
  const overallTotal = options.overall?.total ?? null;
  const tooLarge = (message: string) => new AssetTooLargeError(`${url}: ${message}`);

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

  if (options.range) {
    const [start, end] = options.range;
    if (!isSize(start) || !isSize(end) || end < start) throw new IntegrityError(`${url}: [${start}, ${end}) is not a byte range`);
  }

  return withLock(key, async () => {
    const existing = await store.read(key);
    if (existing) {
      report("cached", existing.byteLength, existing.byteLength);
      logger?.debug(`${key} is already stored (${existing.byteLength} bytes)`);
      return existing;
    }

    report("probing", 0, null);
    const info = options.noRange === true ? { ...NO_HEAD } : await head(url, options);

    const validator = info.etag ?? info.lastModified;
    const fileLength = info.length;

    const base = options.range?.[0] ?? 0;
    if (options.range) {
      if (!info.acceptsRanges || info.length === null) throw new DownloadError(`${url} does not accept ranges: it cannot be taken in parts`);
      checkSize(info.length, limits.maxModelBytes, "the file's declared length", tooLarge);
      if (options.range[1] > info.length) throw new IntegrityError(`${url} is ${info.length} bytes: the range ending at ${options.range[1]} is not in it`);
      info.length = options.range[1] - options.range[0];
    }

    if (info.length !== null) checkSize(info.length, limits.maxFileBytes, "declared length", tooLarge);

    if (!info.acceptsRanges || info.length === null) {
      logger?.debug(`${url} does not accept ranges; streaming without resume`);
      const buffer = await streamWhole(url, key, info.length, options, report, limits);
      await verify(buffer, options, url);
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
    const part = (index: number) => partKey(key, index, fileLength ?? total, validator);
    if (!validator) logger?.debug(`${url} has neither ETag nor Last-Modified: an interrupted download of it restarts from zero`);

    const out = new Uint8Array(total);
    const filled: boolean[] = new Array(sliceCount).fill(false);
    const place = (index: number, piece: ArrayBuffer): boolean => {
      const start = index * sliceSize;
      const expected = Math.min(start + sliceSize, total) - start;
      if (piece.byteLength !== expected) return false;
      out.set(new Uint8Array(piece), start);
      filled[index] = true;
      return true;
    };
    const dropParts = async () => {
      for (let index = 0; index < sliceCount; index += 1) {
        const at = part(index);
        if (at) await store.delete(at).catch(() => false);
      }
    };

    let loaded = 0;
    let resumed = 0;
    for (let index = 0; index < sliceCount; index += 1) {
      const at = part(index);
      const stored = at ? await store.read(at) : null;
      if (stored && place(index, stored)) {
        loaded += stored.byteLength;
        resumed += 1;
      }
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
        const slice = await fetchSlice(url, base + start, base + end, options, { total: fileLength ?? total, etag: info.etag });
        if (slice.byteLength !== end - start + 1) {
          throw new IntegrityError(`${url} answered ${slice.byteLength} bytes for the range ${start}-${end}`);
        }
        place(index, slice);
        loaded += slice.byteLength;

        const at = part(index);
        if (at) {
          await cacheOrWarn(
            store,
            at,
            new Response(slice, { headers: contentHeaders(slice.byteLength, info.etag) }),
            logger,
          );
        }
        report("downloading", loaded, total);
      }
    };
    /** The file again in one piece, once, when its parts do not add up. */
    const whole = async (reason: string): Promise<ArrayBuffer> => {
      logger?.warn(`${reason}; downloading ${key} again in one piece`);
      await dropParts();
      const again = await streamWhole(url, key, total, options, report, limits);
      if (again.byteLength !== total) throw new IntegrityError(`${url} sent ${again.byteLength} bytes but declared ${total}`);
      await verify(again, options, url);
      await cacheOrWarn(store, key, new Response(again, { headers: contentHeaders(again.byteLength, info.etag) }), logger);
      return again;
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

      return whole(error.message);
    }

    try {
      await verify(assembled, options, url);
    } catch (error) {
      if (!(error instanceof IntegrityError) || resumed === 0) {
        await dropParts();
        throw error;
      }
      return whole(error.message);
    }

    const cached = await cacheOrWarn(
      store,
      key,
      new Response(assembled, { headers: contentHeaders(assembled.byteLength, info.etag) }),
      logger,
    );

    if (cached) await dropParts();
    return assembled;
  });
}

/**
 * Hash a download before it is stored, and compare it with the digest its manifest declares. Once
 * per download, never per load: the store is the origin's own, and a stored copy is compared
 * through the digest recorded when it was written.
 */
async function verify(bytes: ArrayBuffer, options: FetchOptions, url: string): Promise<void> {
  if (!options.sha256 && !options.onDigest) return;
  const digest = await sha256Hex(bytes);
  if (options.sha256 && digest !== options.sha256.toLowerCase()) {
    throw new IntegrityError(`${url} does not match the SHA-256 its manifest declares (got ${digest}, expected ${options.sha256})`, {
      details: { url, expected: options.sha256, actual: digest },
    });
  }
  options.onDigest?.(digest);
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
  limits: Limits,
): Promise<ArrayBuffer> {
  const response = await fetch(url, {
    cache: "no-store",
    ...(options.signal ? { signal: options.signal } : {}),
    ...(options.headers ? { headers: options.headers } : {}),
  });
  if (!response.ok) {
    throw new DownloadError(`${url} returned ${response.status} for ${key}`);
  }
  const header = response.headers.get("content-length")?.trim() ?? "";
  const declared = total ?? (/^d{1,16}$/.test(header) ? Number(header) : null);
  const tooLarge = (message: string) => new AssetTooLargeError(`${url}: ${message}`);
  if (declared !== null) checkSize(declared, limits.maxFileBytes, "declared length", tooLarge);

  if (!response.body) {
    const buffer = await response.arrayBuffer();
    checkSize(buffer.byteLength, limits.maxFileBytes, "body", tooLarge);
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
      if (loaded + value.byteLength > limits.maxFileBytes) {
        await reader.cancel().catch(() => undefined);
        throw tooLarge(`the body passed the limit of ${limits.maxFileBytes} bytes`);
      }
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
