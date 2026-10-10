/**
 * Structured logging, mirroring `ynonnx_core.utils.logger.StructuredLogger` in intent: a message
 * plus a flat context object, never an f-string soup.
 */

export type LogLevel = "debug" | "info" | "warn" | "error" | "silent";

const ORDER: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
  silent: 100,
};

export interface Logger {
  debug(message: string, context?: Record<string, unknown>): void;
  info(message: string, context?: Record<string, unknown>): void;
  warn(message: string, context?: Record<string, unknown>): void;
  error(message: string, context?: Record<string, unknown>): void;
}

export class StructuredLogger implements Logger {
  constructor(
    private readonly prefix = "ynonnxweb",
    private level: LogLevel = "info",
  ) {}

  setLevel(level: LogLevel): void {
    this.level = level;
  }

  private emit(
    level: Exclude<LogLevel, "silent">,
    message: string,
    context?: Record<string, unknown>,
  ): void {
    if (ORDER[level] < ORDER[this.level]) return;
    const line = `[${this.prefix}] ${message}`;
    const method = level === "debug" ? "log" : level;

    if (context && Object.keys(context).length > 0) {
      // eslint-disable-next-line no-console
      console[method](line, context);
    } else {
      // eslint-disable-next-line no-console
      console[method](line);
    }
  }

  debug(message: string, context?: Record<string, unknown>): void {
    this.emit("debug", message, context);
  }
  info(message: string, context?: Record<string, unknown>): void {
    this.emit("info", message, context);
  }
  warn(message: string, context?: Record<string, unknown>): void {
    this.emit("warn", message, context);
  }
  error(message: string, context?: Record<string, unknown>): void {
    this.emit("error", message, context);
  }

  /** The requested backend could not be used, and here is what ran instead. */
  warnBackendFallback(requested: string, actual: string, reason: string): void {
    this.warn(
      `${requested} is unavailable, running on ${actual} instead — ${reason}`,
      { requested, actual },
    );
  }

  /** The single most useful message in this library. */
  warnSingleThreaded(): void {
    this.warn(
      "WebAssembly is running single-threaded: SharedArrayBuffer is unavailable " +
        "because this page is not cross-origin isolated. Serve it with " +
        "Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: " +
        "require-corp to get multi-threading, ideally on the inference page only.",
    );
  }
}

export const defaultLogger = new StructuredLogger();
