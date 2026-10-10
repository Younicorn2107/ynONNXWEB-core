import { afterEach, describe, expect, it, vi } from "vitest";

import { ensureRoom, holdEntries, releaseEntries, touchEntries } from "../../src/models/eviction.js";
import { MemoryStore } from "../../src/models/store.js";

const QUOTA = 1000;

/** navigator.storage reporting the store's own bytes as the usage. */
function stubStorage(store: MemoryStore): void {
  vi.stubGlobal("navigator", {
    storage: {
      estimate: async () => {
        let usage = 0;
        for (const key of await store.keys()) usage += (await store.stat(key))?.bytes ?? 0;
        return { quota: QUOTA, usage };
      },
      persisted: async () => false,
    },
  });
}

async function put(store: MemoryStore, key: string, bytes: number): Promise<void> {
  await store.write(key, new Response(new Uint8Array(bytes)));
}

describe("room for the next model", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("evicts the least recently used models no bundle holds, and only as many as needed", async () => {
    const store = new MemoryStore();
    stubStorage(store);
    await put(store, "old_int8/model.onnx", 300);
    await put(store, "mid_int8/model.onnx", 300);
    await put(store, "busy_int8/model.onnx", 300);
    await touchEntries(store, ["old_int8"]);
    await new Promise((r) => setTimeout(r, 5));
    await touchEntries(store, ["mid_int8"]);
    holdEntries(["busy_int8"]);
    try {
      const evicted = await ensureRoom(store, 250, new Set(["new_int8"]));
      expect(evicted.map((e) => e.entry)).toEqual(["old_int8"]);
      expect(await store.has("mid_int8/model.onnx")).toBe(true);
      expect(await store.has("busy_int8/model.onnx")).toBe(true);
    } finally {
      releaseEntries(["busy_int8"]);
    }
  });

  it("refuses, naming what is held, when even eviction cannot make room", async () => {
    const store = new MemoryStore();
    stubStorage(store);
    await put(store, "busy_int8/model.onnx", 900);
    holdEntries(["busy_int8"]);
    try {
      await expect(ensureRoom(store, 500, new Set())).rejects.toMatchObject({ code: "QuotaError" });
    } finally {
      releaseEntries(["busy_int8"]);
    }
  });

  it("does nothing when there is room", async () => {
    const store = new MemoryStore();
    stubStorage(store);
    await put(store, "a_int8/model.onnx", 100);
    expect(await ensureRoom(store, 100, new Set())).toEqual([]);
    expect(await store.has("a_int8/model.onnx")).toBe(true);
  });
});
