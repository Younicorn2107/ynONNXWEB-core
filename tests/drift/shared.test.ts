/** The anti-drift suite: the same fixtures, asserted from both languages. */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import * as errors from "../../src/errors.js";
import { detectRole, parseManifest, serializeManifest } from "../../src/types/manifest.js";
import { parseComputeType } from "../../src/types/compute.js";

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

function fixture<T>(...parts: string[]): T {
  return JSON.parse(readFileSync(join(FIXTURES, ...parts), "utf-8")) as T;
}

interface RoleCase {
  name: string;
  inputs: string[];
  outputs: string[];
  filename: string;
  expect: string;
}

describe("detectRole agrees with ynonnx_core.detect_role", () => {
  const cases = fixture<RoleCase[]>("role_detection.json");

  it("covers a meaningful number of cases", () => {
    expect(cases.length).toBeGreaterThanOrEqual(20);
  });

  it.each(cases.map((c) => [c.name, c] as const))("%s", (_name, testCase) => {
    expect(detectRole(testCase.inputs, testCase.outputs, testCase.filename)).toBe(testCase.expect);
  });
});

interface ComputeFixture {
  accepted: {
    input: string;
    weights: string;
    activations: string;
    canonical: string;
    quantized: boolean;
    bits: number;
  }[];
  rejected: string[];
}

describe("the compute-type grammar agrees with ynonnx_core.types.compute", () => {
  const cases = fixture<ComputeFixture>("compute_types.json");

  it.each(cases.accepted.map((c) => [c.input, c] as const))("parses %j", (_input, expected) => {
    const spec = parseComputeType(expected.input);
    expect(spec.weights).toBe(expected.weights);
    expect(spec.activations).toBe(expected.activations);
    expect(spec.canonical).toBe(expected.canonical);
    expect(spec.isQuantized).toBe(expected.quantized);
    expect(spec.bits).toBe(expected.bits);
  });

  it.each(cases.rejected.map((v) => [v] as const))("refuses %j", (value) => {
    let thrown: unknown;
    try {
      parseComputeType(value);
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBeInstanceOf(errors.ComputeTypeError);
    expect((thrown as errors.ComputeTypeError).code).toBe("ComputeTypeError");
  });
});

describe("manifests round-trip identically on both sides", () => {
  const names = ["published_merged", "unmerged_decoders", "flat_single", "fused"];

  it.each(names)("%s", (name) => {
    const raw = fixture<Record<string, unknown>>("manifests", `${name}.json`);

    expect(serializeManifest(parseManifest(raw))).toEqual(raw);
  });
});

describe("the exception hierarchy matches", () => {
  interface Declared {
    name: string;
    parent: string;
  }
  const declared = fixture<{
    root: string;
    shared: Declared[];
    onnx_root: string;
    onnx: Declared[];
  }>("errors.json");

  /**
   * What the browser deliberately does not have: quantization and its quality gate happen when a
   * model is exported, never in a page. `CalibrationError` exists on both Python layers under one
   * name; the shared one is mirrored.
   */
  const NOT_IN_THE_BROWSER = new Set(["QuantizationError", "QualityError"]);
  const table = errors as unknown as Record<string, typeof errors.YnError>;

  const expected = [
    { name: declared.onnx_root, parent: declared.root },
    ...declared.shared,
    ...declared.onnx.filter((c) => !NOT_IN_THE_BROWSER.has(c.name) && c.name !== "CalibrationError"),
  ];

  it.each(expected.map((c) => [c.name, c] as const))("%s", (name, { parent }) => {
    const Cls = table[name];
    expect(Cls, `${name} is missing from errors.ts`).toBeDefined();
    const Parent = table[parent];
    expect(Parent, `${parent} is missing from errors.ts`).toBeDefined();
    const instance = new Cls!("x");
    expect(instance).toBeInstanceOf(Parent!);
    expect(Object.getPrototypeOf(Cls!.prototype)).toBe(Parent!.prototype);

    expect(instance.code).toBe(name);
    expect(errors.deserializeError(instance.toJSON())).toBeInstanceOf(Cls!);
  });
});
