/**
 * The end-to-end proof: a real ONNX graph, run through the real session stack, producing the
 * numbers numpy produced.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

import { ModelBundle } from "../../src/session/bundle.js";
import { detectRole, parseManifest } from "../../src/types/manifest.js";
import type { BackendPlan } from "../../src/backends/select.js";
import type { OrtLike } from "../../src/session/create.js";

const MODELS = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "models");

interface Golden {
  model: string;
  input_name: string;
  output_name: string;
  input: number[][];
  input_dims: number[];
  expected: number[][];
  expected_dims: number[];
  tolerance: number;
}

const goldenPath = join(MODELS, "tiny_matmul.golden.json");
const modelPath = join(MODELS, "tiny_matmul.onnx");
const available = existsSync(goldenPath) && existsSync(modelPath);

const plan: BackendPlan = { order: ["wasm"], dropped: [], threads: 1 };

describe.runIf(available)("a real graph produces the numbers numpy produced", () => {
  const golden = available ? (JSON.parse(readFileSync(goldenPath, "utf-8")) as Golden) : null;
  const bytes = available ? readFileSync(modelPath) : null;

  function manifestFor(inputs: string[], outputs: string[]) {
    return parseManifest({
      model_key: "tiny_matmul__float32",
      compute_type: "float32",
      graphs: [
        {
          filename: golden!.model,
          role: detectRole(inputs, outputs, golden!.model),
          size_mb: bytes!.byteLength / 1e6,
          inputs,
          outputs,
        },
      ],
    });
  }

  it("runs through ModelBundle and matches to tolerance", async () => {
    const files = new Map<string, ArrayBuffer>([
      [
        golden!.model,
        bytes!.buffer.slice(bytes!.byteOffset, bytes!.byteOffset + bytes!.byteLength) as ArrayBuffer,
      ],
    ]);

    const bundle = await ModelBundle.create({
      ort: ort as unknown as OrtLike,
      manifest: manifestFor([golden!.input_name], [golden!.output_name]),
      key: { name: "golden", computeType: "int8" },
      files,
      plan,
    });

    try {
      const session = bundle.only;
      expect(session.backend).toBe("wasm");
      expect(session.inputs.map((s) => s.name)).toEqual([golden!.input_name]);

      const flat = Float32Array.from(golden!.input.flat());
      const outputs = await session.run({
        [golden!.input_name]: new ort.Tensor("float32", flat, golden!.input_dims),
      });

      const actual = (outputs[golden!.output_name] as ort.Tensor).data as Float32Array;
      const expected = golden!.expected.flat();

      expect(expected.length).toBeGreaterThan(0);
      expect(actual.length).toBe(expected.length);
      for (const [index, want] of expected.entries()) {
        expect(actual[index]!, `element ${index}`).toBeCloseTo(want, 5);
      }
    } finally {
      await bundle.close();
    }
  }, 60000);

  it("keeps the non-graph files it was given, so nothing fetches them twice", async () => {
    const tokenizer = new TextEncoder().encode('{"model":{"vocab":{}}}');
    const files = new Map<string, ArrayBuffer>([
      [
        golden!.model,
        bytes!.buffer.slice(bytes!.byteOffset, bytes!.byteOffset + bytes!.byteLength) as ArrayBuffer,
      ],
      ["tokenizer.json", tokenizer.buffer as ArrayBuffer],
    ]);
    const bundle = await ModelBundle.create({
      ort: ort as unknown as OrtLike,
      manifest: manifestFor([golden!.input_name], [golden!.output_name]),
      key: { name: "golden", computeType: "float32" },
      files,
      plan,
    });
    try {
      expect(bundle.assetText("tokenizer.json")).toBe('{"model":{"vocab":{}}}');

      expect(bundle.asset(golden!.model)).toBeNull();
      expect(bundle.asset("missing.json")).toBeNull();
    } finally {
      await bundle.close();
    }
  }, 60000);

  it("keeps an HQQ quantization off WebGPU, before building anything", async () => {
    const files = new Map<string, ArrayBuffer>([
      [golden!.model, bytes!.buffer.slice(bytes!.byteOffset, bytes!.byteOffset + bytes!.byteLength) as ArrayBuffer],
    ]);
    const manifest = { ...manifestFor([golden!.input_name], [golden!.output_name]), modelKey: "turbo_algo-int4-hqq" };
    await expect(
      ModelBundle.create({
        ort: ort as unknown as OrtLike,
        manifest,
        key: { name: "golden", computeType: "int4" },
        files,
        plan: { order: ["webgpu"], dropped: [], threads: 1 },
      }),
    ).rejects.toThrow(/HQQ/);

    const bundle = await ModelBundle.create({
      ort: ort as unknown as OrtLike,
      manifest,
      key: { name: "golden", computeType: "int4" },
      files,
      plan: { order: ["webgpu", "wasm"], dropped: [], threads: 1 },
    });
    try {
      expect(bundle.only.backend).toBe("wasm");
    } finally {
      await bundle.close();
    }
  }, 60000);

  it("reports the role the graph's IO implies", async () => {
    const manifest = manifestFor([golden!.input_name], [golden!.output_name]);
    expect(manifest.graphs[0]?.role).toBe("single");
  });

  it("refuses a bundle whose declared file was never fetched", async () => {
    await expect(
      ModelBundle.create({
        ort: ort as unknown as OrtLike,
        manifest: manifestFor([golden!.input_name], [golden!.output_name]),
        key: { name: "golden", computeType: "int8" },
        files: new Map(),
        plan,
      }),
    ).rejects.toThrow(/was not fetched/);
  });
});

describe.runIf(!available)("golden fixtures", () => {
  it("are missing — generate them", () => {
    throw new Error(
      `the golden model fixtures were not found under ${MODELS}. ` +
        `Run: python scripts/gen_fixtures.py`,
    );
  });
});
