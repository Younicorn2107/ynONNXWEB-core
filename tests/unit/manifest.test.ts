/**
 * `detectRole` is the only piece of genuine logic duplicated verbatim between the Python pack and
 * this one, so it gets the most attention.
 */

import { describe, expect, it } from "vitest";

import { ManifestError } from "../../src/errors.js";
import { entryName, safeName, splitEntry } from "../../src/models/entry.js";
import { huggingFaceResolver, manifestFileFor, serverResolver } from "../../src/models/resolve.js";
import {
  MANIFEST_VERSION_WITH_EXTERNAL_DATA,
  decoderRoles,
  detectRole,
  graphByRole,
  hasRole,
  isEncoderDecoder,
  parseManifest,
  serializeManifest,
  totalSizeMb,
  usesMergedDecoder,
} from "../../src/types/manifest.js";

const kv = (i: number, side: "decoder" | "encoder", kind: "key" | "value") =>
  `past_key_values.${i}.${side}.${kind}`;
const present = (i: number, side: "decoder" | "encoder", kind: "key" | "value") =>
  `present.${i}.${side}.${kind}`;

describe("detectRole — IO wins", () => {
  it("use_cache_branch means a merged decoder, whatever the filename says", () => {
    expect(
      detectRole(["input_ids", kv(0, "decoder", "key"), "use_cache_branch"], ["logits"], "encoder_model.onnx"),
    ).toBe("decoder_merged");
  });

  it("past plus ids is a with-past decoder", () => {
    expect(detectRole(["input_ids", kv(0, "decoder", "key")], ["logits"], "model.onnx")).toBe(
      "decoder_with_past",
    );
  });

  it("past plus encoder states, without ids, is still a with-past decoder", () => {
    expect(
      detectRole(["encoder_hidden_states", kv(0, "encoder", "key")], ["logits"], "x.onnx"),
    ).toBe("decoder_with_past");
  });

  it("present plus ids, no past, is the prefill decoder", () => {
    expect(detectRole(["input_ids"], ["logits", present(0, "decoder", "key")], "")).toBe("decoder");
  });

  it("no ids plus present is an encoder", () => {
    expect(detectRole(["input_features"], [present(0, "encoder", "key")], "")).toBe("encoder");
  });

  it("last_hidden_state alone is an encoder", () => {
    expect(detectRole(["input_features"], ["last_hidden_state"], "")).toBe("encoder");
  });

  it("search scalars mean the graph carries its own beam search", () => {
    expect(detectRole(["input_features", "max_length", "num_beams"], ["sequences"], "")).toBe(
      "fused",
    );
  });

  it("sequences alone is enough to call it fused", () => {
    expect(detectRole(["input_features"], ["sequences", "sequences_scores"], "")).toBe("fused");
  });
});

describe("detectRole — filename only as a fallback", () => {
  it.each([
    ["encoder_model.onnx", "encoder"],
    ["decoder_model_merged.onnx", "decoder_merged"],
    ["decoder_with_past_model.onnx", "decoder_with_past"],
    ["decoder_withpast.onnx", "decoder_with_past"],
    ["decoder_model.onnx", "decoder"],
    ["model.onnx", "single"],
    ["", "single"],
  ])("%s → %s when the IO says nothing", (filename, expected) => {
    expect(detectRole(["x"], ["y"], filename)).toBe(expected);
  });

  it("merged beats decoder in the filename, as the order implies", () => {
    expect(detectRole(["x"], ["y"], "decoder_model_merged.onnx")).toBe("decoder_merged");
  });

  it("is case-insensitive", () => {
    expect(detectRole(["x"], ["y"], "ENCODER_MODEL.ONNX")).toBe("encoder");
  });
});

const PUBLISHED = {
  model_key: "large-v3-turbo__int4",
  compute_type: "int4",
  architecture: "whisper",
  graphs: [
    {
      filename: "encoder_model_q4.onnx",
      role: "encoder",
      sha256: null,
      size_mb: 405.2,
      inputs: ["input_features"],
      outputs: ["last_hidden_state"],
    },
    {
      filename: "decoder_model_merged_q4.onnx",
      role: "decoder_merged",
      sha256: null,
      size_mb: 318.7,
      inputs: ["input_ids", "encoder_hidden_states", "use_cache_branch"],
      outputs: ["logits"],
    },
  ],
  tokenizer_file: "tokenizer.json",
  logits_vocab_size: 51866,
  kv: {
    num_layers: 4,
    num_heads: 20,
    head_dim: 64,
    has_cross_attention: true,
    dtype: "float32",
  },
  exposes_cross_qk: false,
  extra: {},
  version: 1,
};

describe("parseManifest", () => {
  it("reads a published entry", () => {
    const m = parseManifest(PUBLISHED);
    expect(m.modelKey).toBe("large-v3-turbo__int4");
    expect(m.graphs).toHaveLength(2);
    expect(m.kv?.numLayers).toBe(4);
    expect(m.tokenizerFile).toBe("tokenizer.json");
  });

  it("round-trips without losing a field", () => {
    const written = structuredClone(PUBLISHED) as Record<string, unknown>;
    for (const graph of written["graphs"] as Record<string, unknown>[]) {
      graph["external_data"] ??= [];
    }
    expect(serializeManifest(parseManifest(written))).toEqual(written);
  });

  it("defaults external_data to empty when the field predates it", () => {
    expect(parseManifest(PUBLISHED).graphs[0]?.externalData).toEqual([]);
  });

  it("keeps external_data when present", () => {
    const withData = structuredClone(PUBLISHED) as Record<string, unknown>;
    (withData["graphs"] as Record<string, unknown>[])[0]!["external_data"] = ["encoder_model.onnx_data"];
    expect(parseManifest(withData).graphs[0]?.externalData).toEqual(["encoder_model.onnx_data"]);
  });

  it("refuses a payload that is not an object", () => {
    expect(() => parseManifest([1, 2])).toThrow(ManifestError);
    expect(() => parseManifest(null)).toThrow(ManifestError);
  });

  it("refuses a manifest with no model_key", () => {
    expect(() => parseManifest({ graphs: [] })).toThrow(ManifestError);
  });

  it("refuses an unknown role rather than defaulting it", () => {
    const bad = structuredClone(PUBLISHED) as Record<string, unknown>;
    (bad["graphs"] as Record<string, unknown>[])[0]!["role"] = "encodr";
    expect(() => parseManifest(bad)).toThrow(/not a graph role/);
  });

  it("names the offending file when a graph has no filename", () => {
    const bad = structuredClone(PUBLISHED) as Record<string, unknown>;
    delete (bad["graphs"] as Record<string, unknown>[])[0]!["filename"];
    expect(() => parseManifest(bad)).toThrow(/graph 0 has no filename/);
  });
});

describe("manifest queries", () => {
  const m = parseManifest(PUBLISHED);

  it("finds a graph by role", () => {
    expect(graphByRole(m, "encoder")?.filename).toBe("encoder_model_q4.onnx");
    expect(graphByRole(m, "fused")).toBeNull();
  });

  it("recognises an encoder-decoder", () => {
    expect(isEncoderDecoder(m)).toBe(true);
    expect(usesMergedDecoder(m)).toBe(true);
    expect(hasRole(m, "decoder_with_past")).toBe(false);
  });

  it("orders decoder roles the way the pack does", () => {
    expect(decoderRoles(m)).toEqual(["decoder_merged"]);
  });

  it("sums the download size before anything is fetched", () => {
    expect(totalSizeMb(m)).toBeCloseTo(723.9, 1);
  });
});

describe("sidecars a manifest records", () => {
  it("treats an empty list as an answer from version 2", () => {
    expect(MANIFEST_VERSION_WITH_EXTERNAL_DATA).toBe(2);

    const parsed = parseManifest({
      model_key: "base__int8",
      version: 2,
      graphs: [{ filename: "encoder_model.onnx", role: "encoder", external_data: [] }],
    });
    expect(parsed.version).toBe(2);
    expect(parsed.graphs[0]?.externalData).toEqual([]);
  });

  it("keeps the names when there are some", () => {
    const parsed = parseManifest({
      model_key: "large-v3__float32",
      version: 2,
      graphs: [
        {
          filename: "encoder_model.onnx",
          role: "encoder",
          external_data: ["encoder_model.onnx_data"],
        },
      ],
    });
    expect(parsed.graphs[0]?.externalData).toEqual(["encoder_model.onnx_data"]);
  });

  it("defaults an unversioned manifest to 1, so it still gets probed", () => {
    const parsed = parseManifest({
      model_key: "old__int8",
      graphs: [{ filename: "encoder_model.onnx", role: "encoder" }],
    });
    expect(parsed.version).toBeLessThan(MANIFEST_VERSION_WITH_EXTERNAL_DATA);
  });
});

describe("entry naming", () => {
  it("splits on the FIRST underscore, not the last", () => {
    expect(splitEntry("large-v3-turbo_int8_float16")).toEqual([
      "large-v3-turbo",
      "int8_float16",
    ]);

    expect(splitEntry("base_float8_e4m3_float16")).toEqual([
      "base",
      "float8_e4m3_float16",
    ]);
  });

  it("reserves the separator by taking it out of the name", () => {
    expect(safeName("some-org/SomeModel2.0_turbo")).toBe(
      "some-org--SomeModel2.0-turbo",
    );
    expect(entryName("some-org/SomeModel2.0_turbo", "int4")).toBe(
      "some-org--SomeModel2.0-turbo_int4",
    );
  });

  it("round-trips every name the catalogue can produce", () => {
    for (const name of ["base", "large-v3-turbo", "distil-large-v3.5", "moonshine-tiny"]) {
      for (const compute of ["int4", "int8_float16", "float8_e4m3_float16", "float32"]) {
        expect(splitEntry(entryName(name, compute))).toEqual([name, compute]);
      }
    }
  });

  it("reads a separator-less directory as float32", () => {
    expect(splitEntry("exported-elsewhere")).toEqual(["exported-elsewhere", "float32"]);
  });
});

describe("the flat repository layout", () => {
  it("names a manifest per precision", () => {
    expect(manifestFileFor("int4")).toBe("ynonnx_manifest-int4.json");
    expect(manifestFileFor("int8_float16")).toBe("ynonnx_manifest-int8_float16.json");
  });

  it("asks the Hub for the right one", () => {
    const resolver = huggingFaceResolver({ repo: "younicorn/whisper-base", subfolder: "onnx" });
    expect(resolver.manifestUrls({ name: "base", computeType: "int4" })).toEqual([
      "https://huggingface.co/younicorn/whisper-base/resolve/main/onnx/ynonnx_manifest-int4.json",
    ]);
  });

  it("takes the variant folder when the repo is laid out that way", () => {
    const resolver = huggingFaceResolver({
      repo: "younicorn/whisper-small",
      variants: { int4: "onnx/grain-int4-b64", int8: "onnx/plain-int8" },
      manifest: true,
    });
    expect(resolver.manifestUrls({ name: "small", computeType: "int4" })).toEqual([
      "https://huggingface.co/younicorn/whisper-small/resolve/main/onnx/grain-int4-b64/onnx_manifest.json",
    ]);
    expect(
      resolver.assetUrl({ name: "small", computeType: "int8" }, "decoder.onnx").url,
    ).toBe(
      "https://huggingface.co/younicorn/whisper-small/resolve/main/onnx/plain-int8/decoder.onnx",
    );
  });

  it("falls back to the flat layout for a precision it has no folder for", () => {
    const resolver = huggingFaceResolver({
      repo: "younicorn/whisper-small",
      subfolder: "onnx",
      variants: { int4: "onnx/grain-int4-b64" },
    });
    expect(resolver.manifestUrls({ name: "small", computeType: "float16" })).toEqual([
      "https://huggingface.co/younicorn/whisper-small/resolve/main/onnx/ynonnx_manifest-float16.json",
    ]);
  });

  it("still supports a third-party repo that publishes no manifest", () => {
    const resolver = huggingFaceResolver({ repo: "someone/else", manifest: false });
    expect(resolver.manifestUrls({ name: "x", computeType: "int4" })).toEqual([]);
  });
});

describe("file names a manifest gives", () => {
  /** A manifest with one graph and these fields changed. */
  const withGraph = (graph: Record<string, unknown>, top: Record<string, unknown> = {}) =>
    parseManifest({ model_key: "a_int8", graphs: [{ filename: "model.onnx", role: "single", ...graph }], ...top });

  const HOSTILE = [
    "../b_int8/model.onnx",
    "..",
    ".",
    "sub/model.onnx",
    "sub\\model.onnx",
    "/etc/passwd",
    ".hidden",
    "model.onnx?x=1",
    "model.onnx#frag",
    "%2e%2e",
    "",
    "stored_digests.json",
    "source.json",
    "described_manifest.json",
  ];

  it.each(HOSTILE)("refuses %j as a graph filename", (name) => {
    expect(() => withGraph({ filename: name })).toThrow(ManifestError);
  });

  it.each(HOSTILE)("refuses %j as external data, tokenizer file and root file", (name) => {
    expect(() => withGraph({ external_data: [name] })).toThrow(ManifestError);
    if (name !== "") expect(() => withGraph({}, { tokenizer_file: name })).toThrow(ManifestError);
    expect(() => withGraph({}, { extra: { root_files: [name] } })).toThrow(ManifestError);
  });

  it("accepts the names exports actually use", () => {
    const m = withGraph(
      { filename: "decoder_model_merged-int4.onnx", external_data: ["decoder_model_merged-int4.onnx_data", "model.onnx.data_1"] },
      { tokenizer_file: "tokenizer.json", extra: { root_files: ["config.json"] } },
    );
    expect(m.graphs[0]!.externalData).toHaveLength(2);
  });

  it("refuses sizes that are not sizes and digests that are not SHA-256", () => {
    for (const size of [Number.NaN, -1, Number.POSITIVE_INFINITY, "12"]) expect(() => withGraph({ size_mb: size })).toThrow(ManifestError);
    expect(() => withGraph({ sha256: "abc" })).toThrow(ManifestError);
    expect(withGraph({ sha256: "A".repeat(64) }).graphs[0]!.sha256).toBe("a".repeat(64));
  });

  it("refuses KV geometry that cannot be a count", () => {
    expect(() => withGraph({}, { kv: { num_layers: 1e12, num_heads: 1, head_dim: 1 } })).toThrow(ManifestError);
    expect(() => withGraph({}, { logits_vocab_size: -5 })).toThrow(ManifestError);
  });
});

describe("URLs built from file names", () => {
  it("encode each segment and refuse dot segments, for the server and the Hub", () => {
    const key = { name: "a", computeType: "int8" };
    const server = serverResolver({ base: "https://s.test/models/" });
    const hub = huggingFaceResolver({ repo: "org/repo", subfolder: "onnx", endpoint: "https://hub.test" });
    for (const resolver of [server, hub]) {
      expect(() => resolver.assetUrl(key, "../b_int8/model.onnx")).toThrow(ManifestError);
      expect(() => resolver.assetUrl(key, "a/./b")).toThrow(ManifestError);
      expect(resolver.assetUrl(key, "model.onnx?x#y").url).toMatch(/model\.onnx%3Fx%23y$/);
    }
    expect(server.assetUrl(key, "model.onnx").url).toBe("https://s.test/models/a_int8/model.onnx");

    expect(hub.assetUrl(key, "text_encoder/tokenizer.json", "root").url).toBe("https://hub.test/org/repo/resolve/main/text_encoder/tokenizer.json");

    expect(() => server.assetUrl({ name: "a", computeType: "x/../../b" }, "model.onnx")).toThrow(ManifestError);
  });

  it("name where the bytes come from: host, repository, revision and folder", () => {
    const a = huggingFaceResolver({ repo: "org/repo", subfolder: "onnx" });
    const b = huggingFaceResolver({ repo: "org/repo", subfolder: "onnx", endpoint: "https://mirror.test" });
    const c = huggingFaceResolver({ repo: "org/repo", subfolder: "onnx", revision: "0123abc" });
    expect(new Set([a.source, b.source, c.source]).size).toBe(3);
    expect(a.id).toBe(b.id);
  });
});
