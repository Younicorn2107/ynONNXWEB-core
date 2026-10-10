/** The exception hierarchy of the pack, mirrored from Python. */

export interface SerializedError {
  /** The class name, identical to the Python class. */
  type: string;
  /** The same name, under the field the audio spec calls contractual (§ 10). */
  code: string;
  message: string;
  /** Structured facts for a program to act on (the missing capability, the offending value…). */
  details?: Record<string, unknown>;
  /** Whether trying again, unchanged, may succeed (a network failure) or cannot (a bad argument). */
  retryable: boolean;
  /** The cause, when it was itself one of ours or a plain Error. */
  cause?: SerializedError;
}

export class YnError extends Error {
  static readonly code: string = "YnError";
  /** Whether the same call, repeated, may succeed. */
  static readonly retryable: boolean = false;
  /** Structured facts, serialized with the error. */
  readonly details: Record<string, unknown> | undefined;

  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(message, options);
    this.details = options?.details;
    this.name = (new.target as typeof YnError).code;

    Object.setPrototypeOf(this, new.target.prototype);
  }

  /** The Python class name: the identity this error keeps across a boundary. */
  get code(): string {
    return (this.constructor as typeof YnError).code;
  }

  toJSON(): SerializedError {
    return serializeError(this);
  }
}

export class ModelNotFoundError extends YnError {
  static override readonly code: string = "ModelNotFoundError";
}
export class DownloadError extends YnError {
  static override readonly code: string = "DownloadError";
  static override readonly retryable: boolean = true;
}
export class ConversionError extends YnError {
  static override readonly code: string = "ConversionError";
}
export class CacheError extends YnError {
  static override readonly code: string = "CacheError";
}
/** The device cannot hold what was asked — VRAM in Python, any memory here. */
export class VRAMError extends YnError {
  static override readonly code: string = "VRAMError";
}
export class AllocationError extends YnError {
  static override readonly code: string = "AllocationError";
}
export class InferenceError extends YnError {
  static override readonly code: string = "InferenceError";
}
export class ConfigurationError extends YnError {
  static override readonly code: string = "ConfigurationError";
}
export class MissingDependencyError extends YnError {
  static override readonly code: string = "MissingDependencyError";
}
export class LanguageMismatchError extends YnError {
  static override readonly code: string = "LanguageMismatchError";
}
export class SourceError extends YnError {
  static override readonly code: string = "SourceError";
}
export class UploadTooLargeError extends SourceError {
  static override readonly code: string = "UploadTooLargeError";
}
export class AudioLoadError extends YnError {
  static override readonly code: string = "AudioLoadError";
}
export class TokenizerError extends YnError {
  static override readonly code: string = "TokenizerError";
}
export class VADError extends YnError {
  static override readonly code: string = "VADError";
}
export class EmptyCandidateSetError extends ConfigurationError {
  static override readonly code: string = "EmptyCandidateSetError";
}
export class DocumentTooLongError extends ConfigurationError {
  static override readonly code: string = "DocumentTooLongError";
}
export class CalibrationError extends YnError {
  static override readonly code: string = "CalibrationError";
}
export class MissingHeadError extends YnError {
  static override readonly code: string = "MissingHeadError";
}
export class TextTooLongError extends ConfigurationError {
  static override readonly code: string = "TextTooLongError";
}
export class PoolingError extends ConfigurationError {
  static override readonly code: string = "PoolingError";
}
export class DimensionMismatchError extends ConfigurationError {
  static override readonly code: string = "DimensionMismatchError";
}
export class ExportError extends YnError {
  static override readonly code: string = "ExportError";
}
export class CheckpointError extends YnError {
  static override readonly code: string = "CheckpointError";
}
export class TaskError extends YnError {
  static override readonly code: string = "TaskError";
}

export class YnONNXError extends YnError {
  static override readonly code: string = "YnONNXError";
}
/** A backend was named that does not exist, or none could be used. */
export class ProviderError extends YnONNXError {
  static override readonly code: string = "ProviderError";
}
/** The ONNX session could not be created on any candidate backend. */
export class SessionError extends YnONNXError {
  static override readonly code: string = "SessionError";
}
/** The manifest is absent, unreadable, or contradicts its files. */
export class ManifestError extends YnONNXError {
  static override readonly code: string = "ManifestError";
}
/** The decode loop cannot run, or ran into an unrecoverable state. */
export class GenerationError extends YnONNXError {
  static override readonly code: string = "GenerationError";
}

/** The browser refused to store, or gave back, what we put in it. */
export class StorageError extends CacheError {
  static override readonly code: string = "StorageError";
}
/** Not enough origin storage for what is about to be fetched. */
export class QuotaError extends StorageError {
  static override readonly code: string = "QuotaError";
}
/** A downloaded asset does not match its declared length or digest. */
export class IntegrityError extends DownloadError {
  static override readonly code: string = "IntegrityError";
}
/** The compute-type string is not spellable, or not runnable here. */
export class ComputeTypeError extends ConfigurationError {
  static override readonly code: string = "ComputeTypeError";
}
/** The environment cannot do this at all: no secure context, no runtime. */
export class UnsupportedEnvironmentError extends YnONNXError {
  static override readonly code: string = "UnsupportedEnvironmentError";
}
/** A task was cancelled by its caller, locally or from a remote peer. */
export class CancelledError extends YnError {
  static override readonly code: string = "CancelledError";
}

type ErrorClass = typeof YnError;

const REGISTRY = new Map<string, ErrorClass>();

/** Make a class known to `deserializeError`. */
export function registerErrors(...classes: ErrorClass[]): void {
  for (const cls of classes) REGISTRY.set(cls.code, cls);
}

registerErrors(
  YnError, ModelNotFoundError, DownloadError, ConversionError, CacheError, VRAMError,
  AllocationError, InferenceError, ConfigurationError, MissingDependencyError,
  LanguageMismatchError, SourceError, UploadTooLargeError, AudioLoadError, TokenizerError,
  VADError, EmptyCandidateSetError, DocumentTooLongError, CalibrationError, MissingHeadError,
  TextTooLongError, PoolingError, DimensionMismatchError, ExportError, CheckpointError,
  TaskError, YnONNXError, ProviderError, SessionError, ManifestError, GenerationError,
  StorageError, QuotaError, IntegrityError, ComputeTypeError, UnsupportedEnvironmentError,
  CancelledError,
);

/** The class registered under a Python name, if any. */
export function errorClass(type: string): ErrorClass | undefined {
  return REGISTRY.get(type);
}

export function serializeError(error: unknown): SerializedError {
  if (error instanceof YnError) {
    const out: SerializedError = { type: error.code, code: error.code, message: error.message, retryable: (error.constructor as typeof YnError).retryable };
    if (error.details !== undefined) out.details = error.details;
    if (error.cause !== undefined) out.cause = serializeError(error.cause);
    return out;
  }
  if (error instanceof Error) {
    if (error.name === "AbortError") return { type: CancelledError.code, code: CancelledError.code, message: error.message, retryable: false };
    const name = error.name || "Error";
    return { type: name, code: name, message: error.message, retryable: false };
  }
  return { type: "Error", code: "Error", message: String(error), retryable: false };
}

/** Rebuild an error that crossed a boundary. */
export function deserializeError(payload: SerializedError | (Omit<SerializedError, "code" | "retryable"> & Partial<SerializedError>)): YnError {
  const cause = payload.cause ? deserializeError(payload.cause) : undefined;
  const options = { ...(cause ? { cause } : {}), ...(payload.details ? { details: payload.details } : {}) };
  const name = payload.code ?? payload.type;
  const Cls = REGISTRY.get(name);
  if (Cls) return new Cls(payload.message, options);
  const error = new YnError(payload.message, options);
  error.name = name;
  return error;
}
