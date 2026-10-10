/** The shape of the build, asserted rather than assumed. */

import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { toOrtOptions } from "../../src/session/create.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DIST = join(ROOT, "dist");
const built = existsSync(DIST);

const pkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf-8")) as {
  version: string;
  exports: Record<string, { types?: string; import?: string } | string>;
  main: string;
  files: string[];
};

describe("version", () => {
  it("matches the one compiled into the source", async () => {
    const { VERSION } = await import("../../src/version.js");
    expect(VERSION).toBe(pkg.version);
  });
});

describe.runIf(built)("the build output matches what package.json promises", () => {
  const entries = Object.entries(pkg.exports).filter(
    (entry): entry is [string, { types?: string; import?: string }] =>
      typeof entry[1] === "object",
  );

  it.each(entries)("%s resolves to files that exist", (_name, target) => {
    for (const path of [target.import, target.types].filter(Boolean) as string[]) {
      expect(existsSync(join(ROOT, path)), `${path} is declared but missing`).toBe(true);
    }
  });

  it("has a main that exists", () => {
    expect(existsSync(join(ROOT, pkg.main))).toBe(true);
  });
});

describe.runIf(built)("the standalone bundle is genuinely standalone", () => {
  const bundle = () => readFileSync(join(DIST, "ynonnxweb.bundle.js"), "utf-8");

  it("has no bare imports left to resolve", () => {
    const bare = [...bundle().matchAll(/^\s*import\s[^;]*?from\s*["']([^."'][^"']*)["']/gm)].map(
      (match) => match[1],
    );
    expect(bare, `unresolved imports: ${bare.join(", ")}`).toEqual([]);
  });

  it("pulls in no node builtins", () => {
    expect(bundle()).not.toMatch(/from\s*["']node:/);
    expect(bundle()).not.toContain("createRequire");
  });

  it("actually contains the runtime, not a reference to it", () => {
    expect(statSync(join(DIST, "ynonnxweb.bundle.js")).size).toBeGreaterThan(200_000);
  });
});

describe.runIf(built)("the worker bundle can run from a Blob URL", () => {
  it("is one self-contained file with no dynamic import", () => {
    const worker = readFileSync(join(DIST, "ynonnxweb.worker.js"), "utf-8");
    expect(worker).not.toMatch(/\bimport\s*\(/);
    expect(worker).not.toMatch(/^\s*import\s[^;]*?from\s*["'][^."']/m);
  });
});

describe("where outputs land", () => {
  it("keeps the KV cache on the device and brings the logits back", () => {
    const options = toOrtOptions("webgpu", {
      preferredOutputLocation: {
        "present.0.decoder.key": "gpu-buffer",
        logits: "cpu",
      },
    });
    expect(options["preferredOutputLocation"]).toEqual({
      "present.0.decoder.key": "gpu-buffer",
      logits: "cpu",
    });
  });

  it("never asks wasm to keep anything on a device it does not have", () => {
    const options = toOrtOptions("wasm", { preferredOutputLocation: "gpu-buffer" });
    expect(options["preferredOutputLocation"]).toBeUndefined();
  });
})
