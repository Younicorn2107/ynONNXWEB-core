import { describe, expect, it } from "vitest";

import { serverResolver } from "../../src/models/resolve.js";

describe("the key a further URL is built from", () => {
  it("is the fetch key, never the manifest's model_key", () => {
    const resolver = serverResolver({ base: "/models/" });
    const key = { name: "base", computeType: "int8" };
    expect(resolver.assetUrl(key, "tokenizer.json").url).toBe(
      "/models/base_int8/tokenizer.json",
    );

    const fromManifest = { name: "whisper-base", computeType: "edge-int8-1layer" };
    expect(resolver.assetUrl(fromManifest, "tokenizer.json").url).toBe(
      "/models/whisper-base_edge-int8-1layer/tokenizer.json",
    );
  });

  it("offers both manifest names, the cache one first", () => {
    const resolver = serverResolver({ base: "/models/" });
    expect(resolver.manifestUrls({ name: "base", computeType: "int8" })).toEqual([
      "/models/base_int8/ynonnx_manifest.json",
      "/models/base_int8/onnx_manifest.json",
    ]);
  });
});
