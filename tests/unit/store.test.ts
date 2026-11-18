/**
 * Store keys are built from names a manifest or a listing chose: none may reach another entry's
 * files.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { StorageError } from "../../src/errors.js";
import { CacheApiStore, MemoryStore, keySegments } from "../../src/models/store.js";

/** A Cache API that keeps responses by URL, as the browser's does (fragment ignored). */
function fakeCaches() {
  const entries = new Map<string, Response>();
  const strip = (url: string) => url.split("#")[0]!;
  const cache = {
    async match(url: string) {
      return entries.get(strip(new URL(url).href))?.clone();
    },
    async put(url: string, response: Response) {
      entries.set(strip(new URL(url).href), response);
    },
    async delete(url: string) {
      return entries.delete(strip(new URL(url).href));
    },
    async keys() {
      return [...entries.keys()].map((url) => new Request(url));
    },
  };
  vi.stubGlobal("caches", { open: async () => cache, delete: async () => true });
  return entries;
}

afterEach(() => vi.unstubAllGlobals());

const TRAVERSALS = ["a_int8/../b_int8/model.onnx", "a_int8/./model.onnx", "a_int8//model.onnx", "../model.onnx", "a_int8/..", "a_int8\\..\\b"];

describe("store keys", () => {
  it.each(TRAVERSALS)("%j is refused by every store, not normalised into another entry", async (key) => {
    expect(() => keySegments(key)).toThrow(StorageError);
    const memory = new MemoryStore();
    await expect(memory.write(key, new Response("x"))).rejects.toBeInstanceOf(StorageError);
    await expect(memory.read(key)).rejects.toBeInstanceOf(StorageError);
    fakeCaches();
    const cache = new CacheApiStore();
    await expect(cache.write(key, new Response("x"))).rejects.toBeInstanceOf(StorageError);
    await expect(cache.read(key)).rejects.toBeInstanceOf(StorageError);
  });

  it("keeps entries whose names hold URL characters apart in the Cache API, and lists them back as written", async () => {
    const entries = fakeCaches();
    const store = new CacheApiStore();

    await store.write("org--repo#embed_int4/model.onnx", new Response("embed"));
    await store.write("org--repo#decoder_int4/model.onnx", new Response("decoder"));
    await store.write("a_int8/model.onnx?x=1", new Response("query"));
    expect(entries.size).toBe(3);
    expect(new TextDecoder().decode((await store.read("org--repo#embed_int4/model.onnx"))!)).toBe("embed");
    expect(new TextDecoder().decode((await store.read("org--repo#decoder_int4/model.onnx"))!)).toBe("decoder");
    expect((await store.keys()).sort()).toEqual(["a_int8/model.onnx?x=1", "org--repo#decoder_int4/model.onnx", "org--repo#embed_int4/model.onnx"]);
  });
});
