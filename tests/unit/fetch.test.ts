/** Downloading, which is where the bugs live. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createHash } from "node:crypto";

import { AssetTooLargeError, DownloadError, IntegrityError, StorageError } from "../../src/errors.js";
import { fetchAsset } from "../../src/models/fetch.js";
import { Sha256, sha256Hex } from "../../src/models/sha256.js";
import { MemoryStore, type AssetStore } from "../../src/models/store.js";

/**
 * A small slice size, so the multi-slice path is exercised without allocating tens of megabytes per
 * test. The production default is 16 MB; the behaviour under test does not depend on the number.
 */
const SLICE = 4096;

/** Bodies carry position-dependent bytes so a misassembly cannot pass. */
function makeBody(size: number): Uint8Array {
  const body = new Uint8Array(size);
  for (let i = 0; i < size; i += 1) body[i] = i % 251;
  return body;
}

interface ServerOptions {
  acceptsRanges?: boolean;
  etag?: string | null;
  failSlices?: Map<number, number>;
  headFails?: boolean;
  ignoreRange?: boolean;
}

function fakeServer(body: Uint8Array, options: ServerOptions = {}) {
  const acceptsRanges = options.acceptsRanges !== false;
  const etag = options.etag === undefined ? '"v1"' : options.etag;
  const failures = options.failSlices ?? new Map<number, number>();
  const stats = { head: 0, ranges: [] as [number, number][], full: 0, bytesSent: 0 };

  const impl = async (url: string, init?: RequestInit): Promise<Response> => {
    if (init?.method === "HEAD") {
      stats.head += 1;
      if (options.headFails) throw new Error("HEAD refused");
      return new Response(null, {
        status: 200,
        headers: {
          "content-length": String(body.byteLength),
          ...(acceptsRanges ? { "accept-ranges": "bytes" } : {}),
          ...(etag ? { etag } : {}),
        },
      });
    }

    const range = (init?.headers as Record<string, string> | undefined)?.["Range"];
    if (range && !options.ignoreRange) {
      const match = /bytes=(\d+)-(\d+)/.exec(range);
      const start = Number(match?.[1]);
      const end = Number(match?.[2]);
      const index = Math.floor(start / SLICE);
      const remaining = failures.get(index) ?? 0;
      if (remaining > 0) {
        failures.set(index, remaining - 1);
        throw new Error(`slice ${index} is having a bad day`);
      }
      stats.ranges.push([start, end]);
      const slice = body.slice(start, end + 1);
      stats.bytesSent += slice.byteLength;
      return new Response(slice.buffer as ArrayBuffer, { status: 206, headers: { "content-length": String(slice.byteLength) } });
    }

    stats.full += 1;
    stats.bytesSent += body.byteLength;
    return new Response(body.buffer as ArrayBuffer, {
      status: 200,
      headers: { "content-length": String(body.byteLength) },
    });
  };

  return { impl: vi.fn(impl), stats };
}

const SIZE = SLICE * 2 + 1024;

let originalFetch: typeof fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("ranged download", () => {
  it("assembles the file byte for byte", async () => {
    const body = makeBody(SIZE);
    const server = fakeServer(body);
    globalThis.fetch = server.impl as unknown as typeof fetch;

    const store = new MemoryStore();
    const out = await fetchAsset("https://x/model.onnx", "model.onnx", { store, sliceBytes: SLICE });

    expect(new Uint8Array(out)).toEqual(body);
  });

  it("asks for exactly the slices it needs", async () => {
    const server = fakeServer(makeBody(SIZE));
    globalThis.fetch = server.impl as unknown as typeof fetch;

    await fetchAsset("https://x/m.onnx", "m.onnx", { store: new MemoryStore(), sliceBytes: SLICE });

    expect(server.stats.ranges).toHaveLength(3);
    expect(server.stats.bytesSent).toBe(SIZE);
  });

  it("reports progress that ends at the total", async () => {
    const server = fakeServer(makeBody(SIZE));
    globalThis.fetch = server.impl as unknown as typeof fetch;
    const seen: number[] = [];

    await fetchAsset("https://x/m.onnx", "m.onnx", {
      store: new MemoryStore(),
      sliceBytes: SLICE,
      onProgress: (p) => seen.push(p.loaded),
    });

    expect(seen.at(-1)).toBe(SIZE);

    expect([...seen].sort((a, b) => a - b)).toEqual(seen);
  });
});

describe("resume", () => {
  it("fetches only the missing bytes after an interruption", async () => {
    const body = makeBody(SIZE);
    const store = new MemoryStore();

    const failing = fakeServer(body, { failSlices: new Map([[1, 99]]) });
    globalThis.fetch = failing.impl as unknown as typeof fetch;
    await expect(
      fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE }),
    ).rejects.toThrow(DownloadError);
    const firstAttemptBytes = failing.stats.bytesSent;
    expect(firstAttemptBytes).toBeGreaterThan(0);

    const healthy = fakeServer(body);
    globalThis.fetch = healthy.impl as unknown as typeof fetch;
    const out = await fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE });

    expect(new Uint8Array(out)).toEqual(body);
    expect(healthy.stats.bytesSent).toBeLessThan(SIZE);
  }, 20000);

  it("discards stored parts when the ETag changed", async () => {
    const store = new MemoryStore();
    const first = fakeServer(makeBody(SIZE), { failSlices: new Map([[2, 99]]), etag: '"v1"' });
    globalThis.fetch = first.impl as unknown as typeof fetch;
    await expect(fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE })).rejects.toThrow();

    const body2 = makeBody(SIZE).map((b) => (b + 1) % 251);
    const second = fakeServer(body2, { etag: '"v2"' });
    globalThis.fetch = second.impl as unknown as typeof fetch;
    const out = await fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE });

    expect(new Uint8Array(out)).toEqual(body2);
    expect(second.stats.bytesSent).toBe(SIZE);
  }, 20000);

  it("returns a stored file without touching the network", async () => {
    const body = makeBody(1024);
    const store = new MemoryStore();
    await store.write("m.onnx", new Response(body.buffer as ArrayBuffer));

    const server = fakeServer(body);
    globalThis.fetch = server.impl as unknown as typeof fetch;
    const out = await fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE });

    expect(new Uint8Array(out)).toEqual(body);
    expect(server.impl).not.toHaveBeenCalled();
  });

  it("says so through the progress phase when it was cached", async () => {
    const store = new MemoryStore();
    await store.write("m.onnx", new Response(makeBody(64).buffer as ArrayBuffer));
    const phases: string[] = [];
    await fetchAsset("https://x/m.onnx", "m.onnx", {
      store,
      sliceBytes: SLICE,
      onProgress: (p) => phases.push(p.phase),
    });
    expect(phases).toEqual(["cached"]);
  });
});

describe("servers that will not do ranges", () => {
  it("streams the whole body instead", async () => {
    const body = makeBody(SIZE);
    const server = fakeServer(body, { acceptsRanges: false });
    globalThis.fetch = server.impl as unknown as typeof fetch;

    const out = await fetchAsset("https://x/m.onnx", "m.onnx", { store: new MemoryStore(), sliceBytes: SLICE });

    expect(new Uint8Array(out)).toEqual(body);
    expect(server.stats.full).toBe(1);
  });

  it("still works when HEAD itself fails", async () => {
    const body = makeBody(4096);
    const server = fakeServer(body, { headFails: true });
    globalThis.fetch = server.impl as unknown as typeof fetch;

    const out = await fetchAsset("https://x/m.onnx", "m.onnx", { store: new MemoryStore(), sliceBytes: SLICE });
    expect(new Uint8Array(out)).toEqual(body);
  });

  it("refuses a server that advertises ranges then ignores them", async () => {
    const server = fakeServer(makeBody(SIZE), { ignoreRange: true });
    globalThis.fetch = server.impl as unknown as typeof fetch;

    await expect(
      fetchAsset("https://x/m.onnx", "m.onnx", { store: new MemoryStore(), sliceBytes: SLICE }),
    ).rejects.toThrow(/ignored a Range request/);
  }, 20000);
});

describe("retries", () => {
  it("recovers from a transient slice failure", async () => {
    const body = makeBody(SIZE);
    const server = fakeServer(body, { failSlices: new Map([[1, 2]]) });
    globalThis.fetch = server.impl as unknown as typeof fetch;

    const out = await fetchAsset("https://x/m.onnx", "m.onnx", { store: new MemoryStore(), sliceBytes: SLICE });
    expect(new Uint8Array(out)).toEqual(body);
  }, 20000);

  it("gives up with a message naming the range", async () => {
    const server = fakeServer(makeBody(SIZE), { failSlices: new Map([[0, 99]]) });
    globalThis.fetch = server.impl as unknown as typeof fetch;

    await expect(
      fetchAsset("https://x/m.onnx", "m.onnx", { store: new MemoryStore(), sliceBytes: SLICE }),
    ).rejects.toThrow(/could not fetch bytes 0-/);
  }, 20000);
});

describe("cancellation", () => {
  it("stops when the signal aborts", async () => {
    const server = fakeServer(makeBody(SIZE));
    globalThis.fetch = server.impl as unknown as typeof fetch;
    const controller = new AbortController();
    controller.abort();

    await expect(
      fetchAsset("https://x/m.onnx", "m.onnx", {
        store: new MemoryStore(),
        sliceBytes: SLICE,
        signal: controller.signal,
      }),
    ).rejects.toThrow();
  });
});

describe("integrity", () => {
  it("catches a short assembly rather than storing it", async () => {
    const body = makeBody(SLICE + 100);
    const server = fakeServer(body);
    globalThis.fetch = (async (url: string, init?: RequestInit) => {
      const response = await server.impl(url, init);
      if (init?.method === "HEAD") {
        return new Response(null, {
          status: 200,
          headers: {
            "content-length": String(body.byteLength + 500),
            "accept-ranges": "bytes",
          },
        });
      }
      return response;
    }) as unknown as typeof fetch;

    await expect(
      fetchAsset("https://x/m.onnx", "m.onnx", { store: new MemoryStore(), sliceBytes: SLICE }),
    ).rejects.toThrow(IntegrityError);
  }, 20000);
});

describe("a cache that refuses the write", () => {
  function refusingStore(failOn: (key: string) => boolean): AssetStore & { written: string[] } {
    const kept = new Map<string, ArrayBuffer>();
    const written: string[] = [];
    return {
      written,
      async read(key) {
        return kept.get(key) ?? null;
      },
      async write(key, response) {
        if (failOn(key)) {
          throw new StorageError(`could not store '${key}': quota`);
        }
        written.push(key);
        kept.set(key, await response.arrayBuffer());
      },
      async delete(key) {
        return kept.delete(key);
      },
      async keys() {
        return [...kept.keys()];
      },
    } as AssetStore & { written: string[] };
  }

  it("still returns the bytes", async () => {
    const body = new Uint8Array(64).fill(7);
    const fetchMock = vi.fn(async () => new Response(body, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const warn = vi.fn();

    const out = await fetchAsset("https://example.test/model.onnx", "model.onnx", {
      store: refusingStore(() => true),
      noRange: true,
      logger: { warn, info: vi.fn(), debug: vi.fn(), error: vi.fn() } as never,
    });

    expect(new Uint8Array(out)).toEqual(body);
    expect(warn.mock.calls.map(String).join(" ")).toContain("could not be cached");
    vi.unstubAllGlobals();
  });

  it("keeps the sliced parts when the whole entry will not fit", async () => {
    const total = 200;
    const body = new Uint8Array(total).fill(3);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: RequestInit) => {
        if (init?.method === "HEAD") {
          return new Response(null, {
            status: 200,

            headers: { "content-length": String(total), "accept-ranges": "bytes", etag: "\"v1\"" },
          });
        }
        const range = /bytes=(\d+)-(\d+)/.exec(String((init?.headers as Record<string, string>)?.["Range"] ?? ""));
        const start = range ? Number(range[1]) : 0;
        const end = range ? Number(range[2]) : total - 1;
        return new Response(body.slice(start, end + 1), { status: 206 });
      }),
    );

    const store = refusingStore((key) => !key.includes(".part"));
    const out = await fetchAsset("https://example.test/big.onnx", "big.onnx", {
      store,
      sliceBytes: 64,
      logger: { warn: vi.fn(), info: vi.fn(), debug: vi.fn(), error: vi.fn() } as never,
    });

    expect(out.byteLength).toBe(total);
    expect(store.written.length).toBeGreaterThan(0);
    expect(await store.keys()).not.toHaveLength(0);
    vi.unstubAllGlobals();
  });
});

const sha = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

describe("resume needs to know which file the parts belong to", () => {
  it("keeps no part and restarts from zero when the server sends no ETag nor Last-Modified", async () => {
    const body = makeBody(SIZE);
    const store = new MemoryStore();
    const failing = fakeServer(body, { failSlices: new Map([[2, 99]]), etag: null });
    globalThis.fetch = failing.impl as unknown as typeof fetch;
    await expect(fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE })).rejects.toThrow(DownloadError);

    expect(await store.keys()).toEqual([]);

    const healthy = fakeServer(body, { etag: null });
    globalThis.fetch = healthy.impl as unknown as typeof fetch;
    const out = await fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE });
    expect(new Uint8Array(out)).toEqual(body);
    expect(healthy.stats.bytesSent).toBe(SIZE);
  }, 20000);

  it("does not resume from the parts of a file of another size, whatever its ETag", async () => {
    const store = new MemoryStore();
    const first = fakeServer(makeBody(SIZE), { failSlices: new Map([[2, 99]]), etag: '"same"' });
    globalThis.fetch = first.impl as unknown as typeof fetch;
    await expect(fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE })).rejects.toThrow();

    const body2 = makeBody(SIZE + 1).map((b) => (b + 7) % 251);
    const second = fakeServer(body2, { etag: '"same"' });
    globalThis.fetch = second.impl as unknown as typeof fetch;
    const out = await fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE });
    expect(new Uint8Array(out)).toEqual(body2);
    expect(second.stats.bytesSent).toBe(SIZE + 1);
  }, 20000);
});

describe("the digest a manifest declares", () => {
  it("is checked before anything is stored: a mismatch throws IntegrityError and keeps nothing", async () => {
    const body = makeBody(SIZE);
    globalThis.fetch = fakeServer(body).impl as unknown as typeof fetch;
    const store = new MemoryStore();
    await expect(
      fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE, sha256: "0".repeat(64) }),
    ).rejects.toBeInstanceOf(IntegrityError);
    expect(await store.keys()).toEqual([]);
  });

  it("is checked on a streamed download too", async () => {
    globalThis.fetch = fakeServer(makeBody(SIZE), { acceptsRanges: false }).impl as unknown as typeof fetch;
    const store = new MemoryStore();
    await expect(fetchAsset("https://x/m.onnx", "m.onnx", { store, sha256: "f".repeat(64) })).rejects.toBeInstanceOf(IntegrityError);
    expect(await store.has("m.onnx")).toBe(false);
  });

  it("passes when it matches, and the digest computed from the bytes is reported", async () => {
    const body = makeBody(SIZE);
    globalThis.fetch = fakeServer(body).impl as unknown as typeof fetch;
    const seen: string[] = [];
    const out = await fetchAsset("https://x/m.onnx", "m.onnx", {
      store: new MemoryStore(),
      sliceBytes: SLICE,
      sha256: sha(body).toUpperCase(),
      onDigest: (d) => seen.push(d),
    });
    expect(new Uint8Array(out)).toEqual(body);
    expect(seen).toEqual([sha(body)]);
  });

  it("drops parts kept from an older file, and takes the file whole when the assembly fails it", async () => {
    const store = new MemoryStore();
    const first = fakeServer(makeBody(SIZE), { failSlices: new Map([[2, 99]]) });
    globalThis.fetch = first.impl as unknown as typeof fetch;
    await expect(fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE })).rejects.toThrow();
    const body2 = makeBody(SIZE).map((b) => (b + 3) % 251);
    const second = fakeServer(body2);
    globalThis.fetch = second.impl as unknown as typeof fetch;
    const out = await fetchAsset("https://x/m.onnx", "m.onnx", { store, sliceBytes: SLICE, sha256: sha(body2) });
    expect(new Uint8Array(out)).toEqual(body2);
    expect(second.stats.full).toBe(1);
    expect((await store.keys()).filter((k) => k.includes(".part"))).toEqual([]);
  }, 20000);

  it("hashes a large buffer in place, the same as the native digest", async () => {
    for (const size of [0, 1, 55, 56, 63, 64, 65, 1000, 70_001]) {
      const bytes = makeBody(size);
      expect(new Sha256().update(bytes.subarray(0, size >> 1)).update(bytes.subarray(size >> 1)).hex()).toBe(sha(bytes));
      expect(await sha256Hex(bytes)).toBe(sha(bytes));
    }
  });
});

describe("sizes a server states", () => {
  const limits = (maxFileBytes: number) => ({ maxFileBytes, maxModelBytes: 2 ** 40, maxTensorBytes: 2 ** 30 });

  it("refuses a Content-Length over the limit before allocating or fetching anything", async () => {
    const server = fakeServer(makeBody(SIZE));
    globalThis.fetch = server.impl as unknown as typeof fetch;
    await expect(
      fetchAsset("https://x/m.onnx", "m.onnx", { store: new MemoryStore(), sliceBytes: SLICE, limits: limits(SIZE - 1) }),
    ).rejects.toBeInstanceOf(AssetTooLargeError);
    expect(server.stats.ranges).toEqual([]);
    expect(server.stats.full).toBe(0);
  });

  it("stops a body with no usable length at the limit", async () => {
    const body = makeBody(SIZE);
    globalThis.fetch = (async (_url: string, init?: RequestInit) => {
      if (init?.method === "HEAD") return new Response(null, { status: 200, headers: { "content-length": "NaN" } });
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(body);
          controller.enqueue(body);
          controller.close();
        },
      });
      return new Response(stream, { status: 200 });
    }) as unknown as typeof fetch;
    await expect(fetchAsset("https://x/m.onnx", "m.onnx", { store: new MemoryStore(), limits: limits(SIZE + 10) })).rejects.toBeInstanceOf(
      AssetTooLargeError,
    );
  });

  it("takes a slice from a file that changed size meanwhile as an integrity failure, not as a part", async () => {
    const body = makeBody(SIZE);
    const server = fakeServer(body);
    globalThis.fetch = (async (url: string, init?: RequestInit) => {
      const response = await server.impl(url, init);
      if (response.status !== 206) return response;
      return new Response(await response.arrayBuffer(), { status: 206, headers: { "content-range": `bytes 0-1/${SIZE * 2}` } });
    }) as unknown as typeof fetch;

    const out = await fetchAsset("https://x/m.onnx", "m.onnx", { store: new MemoryStore(), sliceBytes: SLICE });
    expect(new Uint8Array(out)).toEqual(body);
    expect(server.stats.full).toBe(1);
  });
});
