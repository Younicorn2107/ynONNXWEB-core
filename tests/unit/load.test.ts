/** `loadModel` end to end, against a fake Hub, on the real onnxruntime-web. */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { IntegrityError } from "../../src/errors.js";
import { encode } from "../../src/models/protobuf.js";
import { huggingFaceResolver, serverResolver } from "../../src/models/resolve.js";
import { MemoryStore } from "../../src/models/store.js";
import { loadModel } from "../../src/session/load.js";
import type { OrtLike } from "../../src/session/create.js";

const GRAPHS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "graphs", "decoder_only");
const GRAPH = readFileSync(join(GRAPHS, "model_q4f16.onnx"));
const SIDECAR = readFileSync(join(GRAPHS, "model_q4f16.onnx_data"));
const TOKENIZER = '{"model":{"type":"BPE","vocab":{}}}';
const CONFIG = '{"model_type":"qwen3","num_hidden_layers":2}';

const HUB = "https://hub.test";
const REPO = "someone/tiny-llm";

function file(path: string, size: number) {
  return { type: "file", path, size };
}

const routes: Record<string, () => Response> = {
  [`${HUB}/api/models/${REPO}/tree/main/onnx`]: () =>
    Response.json([
      file("onnx/model_q4.onnx", GRAPH.byteLength),
      file("onnx/model_q4f16.onnx_data", SIDECAR.byteLength),
      file("onnx/model.onnx", 10),
    ]),
  [`${HUB}/api/models/${REPO}/tree/main`]: () =>
    Response.json([file("tokenizer.json", TOKENIZER.length), file("config.json", CONFIG.length)]),
  [`${HUB}/${REPO}/resolve/main/onnx/model_q4.onnx`]: () => new Response(GRAPH),
  [`${HUB}/${REPO}/resolve/main/onnx/model_q4f16.onnx_data`]: () => new Response(SIDECAR),
  [`${HUB}/${REPO}/resolve/main/tokenizer.json`]: () => new Response(TOKENIZER),
  [`${HUB}/${REPO}/resolve/main/config.json`]: () => new Response(CONFIG),
};

let requested: string[];

beforeEach(() => {
  requested = [];
  vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    requested.push(`${init?.method ?? "GET"} ${url}`);
    const route = routes[url];
    if (!route) return new Response("not found", { status: 404 });
    if (init?.method === "HEAD") return new Response(null, { status: 200 });
    return route();
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const resolver = () => huggingFaceResolver({ repo: REPO, subfolder: "onnx", endpoint: HUB, manifest: false });

describe("a repository without a manifest", () => {
  it("is listed, described and loaded", async () => {
    const store = new MemoryStore();
    const bundle = await loadModel({
      ort: ort as unknown as OrtLike,
      model: { name: REPO, computeType: "int4" },
      resolver: resolver(),
      store,
      backends: "wasm",
      probe: false,
    });
    try {
      expect(bundle.manifest.architecture).toBe("qwen3");
      expect(bundle.manifest.kv).toEqual({
        numLayers: 2, numHeads: 4, headDim: 16, hasCrossAttention: false, dtype: "float16",
      });
      expect(bundle.manifest.logitsVocabSize).toBe(1000);
      expect(bundle.roles).toEqual(["decoder_with_past"]);
      expect(bundle.assetText("tokenizer.json")).toBe(TOKENIZER);

      expect(requested.some((r) => r.endsWith("/onnx/model.onnx"))).toBe(false);
    } finally {
      await bundle.close();
    }

    requested = [];
    const again = await loadModel({
      ort: ort as unknown as OrtLike,
      model: { name: REPO, computeType: "int4" },
      resolver: resolver(),
      store,
      backends: "wasm",
      probe: false,
    });
    try {
      expect(requested.filter((r) => r.includes("/api/models/")).length).toBeGreaterThan(0);
      expect(requested.filter((r) => r.includes("/resolve/"))).toEqual([]);
      expect(again.assetText("tokenizer.json")).toBe(TOKENIZER);
    } finally {
      await again.close();
    }
  }, 60000);

  it("is described by the first source that can, after the ones that publish nothing", async () => {
    const bundle = await loadModel({
      ort: ort as unknown as OrtLike,
      model: { name: REPO, computeType: "int4" },

      resolver: [serverResolver({ base: "https://server.test/models/" }), resolver()],
      store: new MemoryStore(),
      backends: "wasm",
      probe: false,
    });
    try {
      expect(bundle.manifest.extra["described_from"]).toBe(`huggingface:${REPO}`);
      expect(requested.filter((r) => r.includes("server.test"))).toHaveLength(2);
    } finally {
      await bundle.close();
    }
  }, 60000);
});

describe("a model assembled from two precisions", () => {
  const SERVER = "https://server.test/models/";
  const folder = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "graphs", "encoder_decoder");
  const manifestFor = (precision: string) => {
    const manifest = JSON.parse(readFileSync(join(folder, "manifest.json"), "utf-8")) as Record<string, unknown>;
    return JSON.stringify({ ...manifest, model_key: `tiny_${precision}`, compute_type: precision });
  };
  const tokenizer = '{"model":{"type":"BPE","vocab":{}}}';

  it("takes the encoder from one entry and the decoders from the other", async () => {
    const served: Record<string, () => Response> = {};
    for (const precision of ["int8", "int4"]) {
      served[`${SERVER}tiny_${precision}/ynonnx_manifest.json`] = () => new Response(manifestFor(precision));
      for (const file of ["encoder_model_q4.onnx", "decoder_model_merged_q4.onnx"]) {
        served[`${SERVER}tiny_${precision}/${file}`] = () => new Response(readFileSync(join(folder, file)));
      }
      served[`${SERVER}tiny_${precision}/tokenizer.json`] = () => new Response(tokenizer);
    }
    vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requested.push(`${init?.method ?? "GET"} ${url}`);
      const route = served[url];
      if (!route) return new Response("not found", { status: 404 });
      if (init?.method === "HEAD") return new Response(null, { status: 200 });
      return route();
    });

    const bundle = await loadModel({
      ort: ort as unknown as OrtLike,
      model: "tiny",
      computeType: { encoder: "int8", decoder: "int4" },
      resolver: serverResolver({ base: SERVER }),
      store: new MemoryStore(),
      backends: "wasm",
      probe: false,
    });
    try {
      expect(bundle.manifest.computeType).toBe("encoder:int8,decoder:int4");
      expect(bundle.roles.sort()).toEqual(["decoder_merged", "encoder"]);
      const gets = requested.filter((r) => r.startsWith("GET") && r.endsWith(".onnx"));
      expect(gets).toContain(`GET ${SERVER}tiny_int8/encoder_model_q4.onnx`);
      expect(gets).toContain(`GET ${SERVER}tiny_int4/decoder_model_merged_q4.onnx`);
      expect(gets).not.toContain(`GET ${SERVER}tiny_int4/encoder_model_q4.onnx`);
      expect(bundle.assetText("tokenizer.json")).toBe(tokenizer);
    } finally {
      await bundle.close();
    }
  }, 60000);
});

describe("a model republished under the same name", () => {
  const SERVER = "https://server.test/models/";
  const folder = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "graphs", "encoder_decoder");
  const files = ["encoder_model_q4.onnx", "decoder_model_merged_q4.onnx"];
  const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
  const original = (file: string) => new Uint8Array(readFileSync(join(folder, file)));
  /**
   * The same graph re-exported: one more metadata entry, so other bytes and another digest, still a
   * valid model.
   */
  const reexported = (file: string) => {
    const text = (value: string) => new TextEncoder().encode(value);
    const note = encode([{ num: 14, wire: 2, data: encode([{ num: 1, wire: 2, data: text("export") }, { num: 2, wire: 2, data: text("2") }]) }]);
    const bytes = original(file);
    const out = new Uint8Array(bytes.length + note.length);
    out.set(bytes);
    out.set(note, bytes.length);
    return out;
  };
  /** The server: these bytes for each graph, these digests in the manifest. */
  const serve = (bytes: Record<string, Uint8Array>, digests: Record<string, string | null>) => {
    const manifest = JSON.parse(readFileSync(join(folder, "manifest.json"), "utf-8")) as { graphs: { filename: string; sha256: string | null }[] };
    for (const graph of manifest.graphs) graph.sha256 = digests[graph.filename] ?? null;
    const served: Record<string, () => Response> = {
      [`${SERVER}tiny_int4/ynonnx_manifest.json`]: () => Response.json(manifest),
      [`${SERVER}tiny_int4/tokenizer.json`]: () => new Response('{"model":{"type":"BPE","vocab":{}}}'),
    };
    for (const file of files) served[`${SERVER}tiny_int4/${file}`] = () => new Response((bytes[file] ?? original(file)) as Uint8Array<ArrayBuffer>);
    vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requested.push(`${init?.method ?? "GET"} ${url}`);
      const route = served[url];
      if (!route) return new Response("not found", { status: 404 });
      if (init?.method === "HEAD") return new Response(null, { status: 200 });
      return route();
    });
  };
  const load = async (store: MemoryStore) => {
    requested = [];
    const bundle = await loadModel({ ort: ort as unknown as OrtLike, model: "tiny", computeType: "int4", resolver: serverResolver({ base: SERVER }), store, backends: "wasm", probe: false });
    await bundle.close();
    return requested.filter((r) => r.startsWith("GET") && r.endsWith(".onnx")).map((r) => r.split("/").pop());
  };
  const real = () => Object.fromEntries(files.map((f) => [f, digest(original(f))]));

  it("fetches again the graphs whose digest changed, and only those", async () => {
    const store = new MemoryStore();

    await store.write("tiny_int4/encoder_model_q4.onnx", new Response(new Uint8Array([1, 2, 3])));
    serve({}, real());
    expect((await load(store)).sort()).toEqual([...files].sort());
    expect(await load(store)).toEqual([]);

    const decoder = reexported("decoder_model_merged_q4.onnx");
    serve({ "decoder_model_merged_q4.onnx": decoder }, { ...real(), "decoder_model_merged_q4.onnx": digest(decoder) });
    expect(await load(store)).toEqual(["decoder_model_merged_q4.onnx"]);
    expect(await load(store)).toEqual([]);
  }, 60000);

  it("records the digest computed from the bytes, not the one the manifest declares", async () => {
    const store = new MemoryStore();
    serve({}, { "encoder_model_q4.onnx": null, "decoder_model_merged_q4.onnx": null });
    await load(store);
    const recorded = JSON.parse(new TextDecoder().decode((await store.read("tiny_int4/stored_digests.json"))!)) as Record<string, string>;

    expect(recorded["encoder_model_q4.onnx"]).toBe(real()["encoder_model_q4.onnx"]);
    expect(recorded["decoder_model_merged_q4.onnx"]).toBe(real()["decoder_model_merged_q4.onnx"]);
    expect(recorded["tokenizer.json"]).toMatch(/^[0-9a-f]{64}$/);
  }, 60000);

  it("refuses a graph whose bytes do not match the declared digest, and keeps nothing of it", async () => {
    const store = new MemoryStore();

    serve({ "decoder_model_merged_q4.onnx": reexported("decoder_model_merged_q4.onnx") }, real());
    await expect(load(store)).rejects.toBeInstanceOf(IntegrityError);
    expect(await store.has("tiny_int4/decoder_model_merged_q4.onnx")).toBe(false);
  }, 60000);
});

describe("an entry stored from one source", () => {
  it("is not served for another: the second source's files are fetched", async () => {
    const store = new MemoryStore();
    const options = { ort: ort as unknown as OrtLike, model: { name: REPO, computeType: "int4" }, store, backends: "wasm" as const, probe: false };
    await (await loadModel({ ...options, resolver: resolver() })).close();

    const MIRROR = "https://mirror.test";
    for (const [url, route] of Object.entries(routes)) routes[url.replace(HUB, MIRROR)] = route;
    requested = [];
    try {
      const bundle = await loadModel({ ...options, resolver: huggingFaceResolver({ repo: REPO, subfolder: "onnx", endpoint: MIRROR, manifest: false }) });
      await bundle.close();
      expect(requested).toContain(`GET ${MIRROR}/${REPO}/resolve/main/onnx/model_q4.onnx`);
      expect(requested.some((r) => r.includes(HUB))).toBe(false);
    } finally {
      for (const url of Object.keys(routes)) if (url.startsWith(MIRROR)) delete routes[url];
    }
  }, 60000);

  it("is described again when its source lists a file at another size", async () => {
    const store = new MemoryStore();
    const options = { ort: ort as unknown as OrtLike, model: { name: REPO, computeType: "int4" }, resolver: resolver(), store, backends: "wasm" as const, probe: false };
    await (await loadModel(options)).close();
    const root = `${HUB}/api/models/${REPO}/tree/main`;
    const listing = routes[root]!;

    routes[root] = () => Response.json([file("tokenizer.json", TOKENIZER.length + 1), file("config.json", CONFIG.length)]);
    requested = [];
    try {
      await (await loadModel(options)).close();
      expect(requested).toContain(`GET ${HUB}/${REPO}/resolve/main/onnx/model_q4.onnx`);
    } finally {
      routes[root] = listing;
    }
  }, 60000);
});

describe("variants", () => {
  it("builds another session over a kept graph, on the same backend, closed with the bundle", async () => {
    const options = { ort: ort as unknown as OrtLike, model: { name: REPO, computeType: "int4" }, resolver: resolver(), store: new MemoryStore(), backends: "wasm" as const, probe: false };
    const plain = await loadModel(options);
    await expect(plain.variant("decoder_with_past", {})).rejects.toThrow(/without 'variants'/);
    await plain.close();
    const bundle = await loadModel({ ...options, variants: true });
    try {
      const variant = await bundle.variant("decoder_with_past", { graphOptimizationLevel: "basic" });
      expect(variant.backend).toBe(bundle.only.backend);
      expect(variant.inputs.map((i) => i.name)).toEqual(bundle.only.inputs.map((i) => i.name));
      await variant.close();
    } finally {
      await bundle.close();
    }
  }, 60000);
});

describe("a server that makes models on demand", () => {
  it("is asked to prepare a model it lacks, then serves it", async () => {
    const SERVER = "https://server.test/models/";
    const folder = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "graphs", "encoder_decoder");
    let made = false;
    const prepared: unknown[] = [];
    vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requested.push(`${init?.method ?? "GET"} ${url}`);
      if (url === `${SERVER}prepare`) {
        const headers = init?.headers as Record<string, string>;
        prepared.push({ method: init?.method, type: headers["Content-Type"], auth: headers["Authorization"], body: JSON.parse(String(init?.body)) });
        made = true;
        return Response.json({ ready: true, entry: "tiny_int4" });
      }
      if (!made) return new Response("not found", { status: 404 });
      if (url === `${SERVER}tiny_int4/ynonnx_manifest.json`) return new Response(readFileSync(join(folder, "manifest.json")));
      if (url === `${SERVER}tiny_int4/tokenizer.json`) return new Response('{"model":{"type":"BPE","vocab":{}}}');
      for (const file of ["encoder_model_q4.onnx", "decoder_model_merged_q4.onnx"]) {
        if (url === `${SERVER}tiny_int4/${file}`) return init?.method === "HEAD" ? new Response(null, { status: 200 }) : new Response(readFileSync(join(folder, file)));
      }
      return new Response("not found", { status: 404 });
    });
    requested = [];
    const bundle = await loadModel({ ort: ort as unknown as OrtLike, model: "tiny", computeType: "int4", resolver: serverResolver({ base: SERVER, prepare: true, headers: { Authorization: "Bearer t" } }), store: new MemoryStore(), backends: "wasm", probe: false });
    try {
      expect(prepared).toEqual([{ method: "POST", type: "application/json", auth: "Bearer t", body: { name: "tiny", computeType: "int4" } }]);
      expect(bundle.roles.sort()).toEqual(["decoder_merged", "encoder"]);
    } finally {
      await bundle.close();
    }
  }, 60000);
});
