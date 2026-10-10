/** The convenience entry: onnxruntime-web already wired in. */

import * as ort from "onnxruntime-web";

import { assertSingleOrtInstance } from "./backends/capabilities.js";
import { defaultLogger } from "./logger.js";
import { createSession as createSessionWith, loadModel as loadModelWith } from "./session/load.js";
import type { LoadOptions } from "./session/load.js";
import type { OrtLike } from "./session/create.js";
import type { ModelBundle } from "./session/bundle.js";
import type { ModelSession } from "./session/bundle.js";

export * from "./index.js";

/** The runtime this entry bound at import time. */
export const runtime = ort as unknown as OrtLike;

assertSingleOrtInstance(defaultLogger);

export type AutoLoadOptions = Omit<LoadOptions, "ort"> & { ort?: OrtLike };

export function loadModel(options: AutoLoadOptions): Promise<ModelBundle> {
  return loadModelWith({ ...options, ort: options.ort ?? runtime });
}

export function createSession(options: AutoLoadOptions): Promise<ModelSession> {
  return createSessionWith({ ...options, ort: options.ort ?? runtime });
}
