import { describe, expect, it } from "vitest";

import { combineIds, contentId, hashParams, hashText } from "../../src/processing/hashing.js";
import { MemoryKeyValueStore, ResultCache, ResumableRun } from "../../src/processing/resume.js";

describe("content ids", () => {
  it("are 32 hex characters, stable, and blind to key order", async () => {
    const a = await hashText("bonjour");
    expect(a).toMatch(/^[0-9a-f]{32}$/);
    expect(await hashText("bonjour")).toBe(a);
    expect(await hashParams({ model: "x", beam: 1 })).toBe(await hashParams({ beam: 1, model: "x" }));
  });

  it("count a typed array's type, not only its bytes", async () => {
    const bytes = new Uint8Array(16).fill(7);
    expect(await contentId(new Float32Array(bytes.buffer))).not.toBe(await contentId(new Int16Array(bytes.buffer)));
  });

  it("keep the order of a list", async () => {
    expect(await contentId(["a", "b"])).not.toBe(await contentId(["b", "a"]));
    expect(await combineIds(["x", "y"])).not.toBe(await combineIds(["y", "x"]));
  });
});

describe("ResumableRun", () => {
  const inputs = ["one", "two", "three"];

  it("resumes what a stopped run finished, and only with the same settings", async () => {
    const store = new MemoryKeyValueStore();
    const first = await ResumableRun.open({ checkpoint: true, inputs, settings: { model: "m" }, store });
    await first.record(0, "UN");
    await first.record(2, "TROIS");

    const again = await ResumableRun.open({ checkpoint: true, inputs, settings: { model: "m" }, store });
    expect(again.resumed).toBe(2);
    expect(await again.has(0)).toBe(true);
    expect(await again.has(1)).toBe(false);
    expect(again.get(2)).toBe("TROIS");

    const other = await ResumableRun.open({ checkpoint: true, inputs, settings: { model: "other" }, store });
    expect(other.resumed).toBe(0);

    await again.complete();
    expect(await store.keys("checkpoints/")).toEqual([]);
  });

  it("keeps nothing without a checkpoint, and answers from the cache across runs", async () => {
    const store = new MemoryKeyValueStore();
    const cache = new ResultCache(store);
    const first = await ResumableRun.open({ inputs, settings: { model: "m" }, cache });
    expect(first.enabled).toBe(false);
    await first.record(1, { text: "DEUX" });

    const second = await ResumableRun.open({ inputs: ["two", "four"], settings: { model: "m" }, cache });
    expect(await second.has(0)).toBe(true);
    expect(second.get(0)).toEqual({ text: "DEUX" });
    expect(second.cached).toBe(1);
    expect(await second.has(1)).toBe(false);
  });
});
