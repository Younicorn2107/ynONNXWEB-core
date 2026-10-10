/** The browser entry point. */

export * from "./core.js";

export {
  detectCapabilities,
  resetCapabilities,
  assertSecureContext,
  assertSingleOrtInstance,
  detectMemory64,
} from "./backends/capabilities.js";
export type { Capabilities, AdapterLimits } from "./backends/capabilities.js";

export { probeBackend, probeBackends } from "./backends/probe.js";
export type { ProbeResult } from "./backends/probe.js";

export { selectBackends, assertKnownBackends } from "./backends/select.js";
export type { BackendPlan } from "./backends/select.js";

export { configureOrtEnv, appliedOrtEnv, resetOrtEnv } from "./backends/ortEnv.js";
export type { OrtEnvSettings } from "./backends/ortEnv.js";

export {
  checkWasmFit,
  checkAdapterFit,
  readStorageBudget,
  assertStorageFor,
  requestPersistence,
  WASM_HEAP_CEILING_MB,
  WASM_WARN_MB,
  WASM_REFUSE_MB,
  wasmRuntime,
} from "./backends/memory.js";
export type { StorageBudget, FitVerdict, WasmRuntime } from "./backends/memory.js";

export {
  MANIFEST_FILE,
  manifestFileFor,
  serverResolver,
  huggingFaceResolver,
  graphAlias,
  entryDirName,
  filesFromManifest,
  externalDataCandidates,
} from "./models/resolve.js";
export type {
  AssetPlace,
  ListedFile,
  ModelListing,
  ModelResolver,
  ModelKey,
  ResolvedAsset,
  ServerResolverOptions,
  HuggingFaceResolverOptions,
} from "./models/resolve.js";

export { ENTRY_SEPARATOR, entryName, safeName, splitEntry } from "./models/entry.js";
export { CacheApiStore, MemoryStore, OpfsStore, ReadOnlyStore, defaultStore, withLock } from "./models/store.js";
export type { AssetStore, StoredAsset } from "./models/store.js";

export {
  fetchAsset,
  MAX_RETRIES,
  BACKOFF_BASE_MS,
  SLICE_BYTES,
  MAX_IN_FLIGHT,
} from "./models/fetch.js";
export type { FetchProgress, FetchOptions } from "./models/fetch.js";

export { buildSession, readSpecs } from "./session/create.js";
export type { SessionConfig, TensorSpec, BuiltSession, OrtLike } from "./session/create.js";

export { ModelBundle } from "./session/bundle.js";
export type { ModelSession, BundleOptions } from "./session/bundle.js";

export { loadModel, createSession } from "./session/load.js";
export type { LoadOptions, PerRole } from "./session/load.js";

export { TaskHost, transferablesOf } from "./tasks/host.js";
export { TaskClient } from "./tasks/client.js";
export type { CallOptions } from "./tasks/client.js";
export {
  decodeWire,
  encodeWire,
  portTransport,
  transportPair,
  webSocketTransport,
} from "./tasks/transports.js";
export type { TaskContext, TaskHandler, TaskMessage, Transport } from "./tasks/protocol.js";
export { camelCase, fromPythonNames, snakeCase, toPythonNames } from "./tasks/wire.js";

export { WorkerClient, createWorker } from "./worker/client.js";
export type { WorkerClientOptions } from "./worker/client.js";
export { registerTask, registeredTasks, serve, workerHost } from "./worker/host.js";
