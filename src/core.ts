/** The isomorphic entry point: everything here is pure logic. */

export { VERSION } from "./version.js";

export {
  YnError,
  ModelNotFoundError,
  DownloadError,
  ConversionError,
  CacheError,
  VRAMError,
  AllocationError,
  InferenceError,
  ConfigurationError,
  MissingDependencyError,
  LanguageMismatchError,
  SourceError,
  UploadTooLargeError,
  AudioLoadError,
  TokenizerError,
  VADError,
  EmptyCandidateSetError,
  DocumentTooLongError,
  CalibrationError,
  MissingHeadError,
  TextTooLongError,
  PoolingError,
  DimensionMismatchError,
  ExportError,
  CheckpointError,
  TaskError,
  YnONNXError,
  ProviderError,
  SessionError,
  ManifestError,
  GenerationError,
  StorageError,
  QuotaError,
  IntegrityError,
  AssetTooLargeError,
  ComputeTypeError,
  UnsupportedEnvironmentError,
  CancelledError,
  registerErrors,
  errorClass,
  serializeError,
  deserializeError,
} from "./errors.js";
export type { SerializedError } from "./errors.js";

export {
  MANIFEST_VERSION_WITH_EXTERNAL_DATA,
  detectRole,
  parseManifest,
  serializeManifest,
  graphByRole,
  hasRole,
  isEncoderDecoder,
  usesMergedDecoder,
  decoderRoles,
  totalSizeMb,
  isSafeFileName,
  SAFE_FILE_NAME,
} from "./types/manifest.js";
export type { GraphManifest, GraphEntry, GraphRole, KVGeometry } from "./types/manifest.js";

export {
  AUTO_BITS_MAX,
  AUTO_BITS_MIN,
  AUTO_COMPUTE_TYPE,
  SLOW_ON,
  resolveComputeType,
  warnIfSlow,
  parseComputeType,
  normalizeComputeType,
  isSpellable,
  checkSupported,
  backendsSupporting,
  WEIGHT_TYPES,
  ACTIVATION_TYPES,
  QUANTIZED_WEIGHTS,
  BITS,
  BACKEND_SUPPORTED,
} from "./types/compute.js";
export type { ComputeSpec } from "./types/compute.js";

export {
  BACKEND_CHAIN,
  KNOWN_BACKENDS,
  AUTO_EXCLUDED,
  BACKEND_ALIASES,
  BACKEND_LABELS,
  resolveAliases,
  isKnownBackend,
} from "./backends/chain.js";
export type { Backend, BackendSpec } from "./backends/chain.js";

export { DEFAULT_LIMITS, resolveLimits } from "./models/limits.js";
export type { Limits } from "./models/limits.js";
export { readGraphInfo, ONNX_DTYPES } from "./models/onnxProto.js";
export type { GraphDim, GraphInfo, GraphValue } from "./models/onnxProto.js";
export { buildManifest, kvGeometry, logitsVocabSize } from "./models/introspect.js";
export type { BuildManifestOptions, IntrospectedGraph } from "./models/introspect.js";
export {
  PRECISION_SUFFIXES,
  SIDE_FILES,
  choosePublishedGraphs,
  neededGraphs,
} from "./models/published.js";
export type { PublishedChoice } from "./models/published.js";

export * from "./generation/processors.js";
export {
  BeamSearch,
  GreedySearch,
  SamplingSearch,
  logSoftmax,
  seededRandom,
  trimSequence,
  withTemperatureLadder,
} from "./generation/search.js";
export type {
  BeamOptions,
  FinishReason,
  GenerationOutput,
  SamplingOptions,
  SearchStrategy,
  StepDecision,
} from "./generation/search.js";

export { GenerationEngine, yieldToEventLoop } from "./generation/engine.js";
export type { GenerateOptions, TensorFactory, TensorLike } from "./generation/engine.js";

export { Tokenizer, alignOffsets } from "./text/tokenizer.js";
export { castGatherIndices, INT32_SUFFIX } from "./models/gatherIndices.js";
export { ARGMAX_SUFFIX, MAX_SUFFIX, argmaxHead, replaceArgMax, replaceIsNaN } from "./models/argmaxHead.js";
export { splitMergedDecoder } from "./models/splitMerged.js";
export { gatherBeforeDequantize } from "./models/gatherDequantize.js";
export { CPU_NBITS_ACCURACY, setNBitsAccuracy } from "./models/nbitsAccuracy.js";
export { padMatMulNBitsRows, PADDED_SUFFIX } from "./models/padNBits.js";
export { WEBGPU_NBITS_BLOCK, repackMatMulNBits } from "./models/repackNBits.js";
export { CONTEXT_STEPS, assemblePrompt, bucketByLength, quantizeContext } from "./text/prompt.js";
export type { Encoding, PaddedBatch, TokenizerOptions } from "./text/tokenizer.js";

export { combineIds, contentId, hashBytes, hashParams, hashText, stableJson } from "./processing/hashing.js";
export {
  IndexedDbStore,
  MemoryKeyValueStore,
  ResultCache,
  ResumableRun,
} from "./processing/resume.js";
export type { KeyValueStore, OpenRunOptions } from "./processing/resume.js";

export { FastTextModel, fasttextHash } from "./text/fasttext.js";
export { SentencePieceProcessor, readSentencePiece, sentencePieceToTokenizerJson, sentencePieceTokenizer } from "./text/sentencepiece.js";
export { SUBTITLE_EXTENSIONS, mapCues, splitFrontMatter, splitLines, splitLinesKeepEnds } from "./text/documents.js";
export type { SentencePieceModel } from "./text/sentencepiece.js";
export type { FastTextPrediction } from "./text/fasttext.js";
export { LID_MODEL_URL, LanguageDetector, MIN_CONFIDENCE, MIN_SEGMENT_CHARS, PREFIX_CHARS } from "./text/language.js";
export type { LanguageSegment } from "./text/language.js";


export {
  EXPORTERS,
  exportBlob,
  exportString,
  extractRecords,
  extractSegments,
  extractText,
  formatTimestamp,
  roundHalfEven,
  splitLongLines,
  toCsv,
  toJson,
  toMarkdown,
  toSerializable,
  toSrt,
  toTxt,
  toVtt,
  toXml,
} from "./export/index.js";
export type { ArrayPolicy, TimedSegment } from "./export/index.js";

export { StructuredLogger, defaultLogger } from "./logger.js";
export type { Logger, LogLevel } from "./logger.js";
