/** The channels a task travels on. */

import { ConfigurationError, TaskError } from "../errors.js";
import type { ListenOptions, TaskMessage, Transport } from "./protocol.js";
import { DEFAULT_LIMITS, type WireLimits } from "./trust.js";

interface PortLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: MessageEvent) => void): void;
  removeEventListener(type: "message", listener: (event: MessageEvent) => void): void;
  start?(): void;
  terminate?(): void;
  close?(): void;
}

/** A Worker, a worker's own scope, or a MessagePort. */
export function portTransport(port: PortLike): Transport {
  return {
    peer: "local",
    send(message, transfer) {
      if (transfer && transfer.length > 0) port.postMessage(message, transfer);
      else port.postMessage(message);
    },
    listen(handler) {
      const listener = (event: MessageEvent) => handler(event.data);
      port.addEventListener("message", listener);
      port.start?.();
      return () => port.removeEventListener("message", listener);
    },
    close() {
      port.terminate?.();
      port.close?.();
    },
  };
}

const TYPED = {
  Float32Array,
  Float64Array,
  Int8Array,
  Int16Array,
  Int32Array,
  Uint8Array,
  Uint16Array,
  Uint32Array,
  BigInt64Array,
  BigUint64Array,
} as const;
type TypedName = keyof typeof TYPED;

/**
 * Keys never written into a decoded object: `__proto__` would call the prototype setter (the object
 * then inherits what the peer chose), and the other two shadow what code reaching for a class
 * expects to find.
 */
export const UNSAFE_KEYS: ReadonlySet<string> = new Set(["__proto__", "constructor", "prototype"]);
const BASE64 = /^[A-Za-z0-9+/]*={0,2}$/;
/** A bigint's decimal digits: parsing is quadratic in the length, so it is bounded. */
const BIGINT = /^-?\d{1,1000}$/;

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

function fromBase64(text: string): Uint8Array {
  if (text.length % 4 !== 0 || !BASE64.test(text)) throw new TaskError("invalid task message: a typed array's base64 is malformed");
  const binary = atob(text);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}

/**
 * A value JSON can carry: typed arrays become `{"$typed": name, "base64": …}` (little-endian bytes,
 * as numpy reads them with `frombuffer`), bigints `{"$bigint": "…"}`. Everything else passes
 * through.
 */
export function encodeWire(value: unknown): unknown {
  if (typeof value === "bigint") return { $bigint: value.toString() };
  if (value === null || typeof value !== "object") return value;
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    const name = value.constructor.name;
    if (Object.hasOwn(TYPED, name)) {
      return { $typed: name, base64: toBase64(new Uint8Array(value.buffer, value.byteOffset, value.byteLength)) };
    }
  }
  if (value instanceof ArrayBuffer) return { $typed: "Uint8Array", base64: toBase64(new Uint8Array(value)) };
  if (Array.isArray(value)) return value.map(encodeWire);
  const out: Record<string, unknown> = {};
  const entries = value instanceof Map ? [...value].map(([k, v]) => [String(k), v] as const) : Object.entries(value);
  for (const [key, item] of entries) if (item !== undefined && !UNSAFE_KEYS.has(key)) out[key] = encodeWire(item);
  return out;
}

/**
 * The inverse of `encodeWire`, for text from a peer: anything malformed throws a `TaskError` (never
 * a `RangeError` from an allocation), nesting deeper than `maxDepth` is refused, and the unsafe
 * keys are dropped.
 */
export function decodeWire(value: unknown, maxDepth: number = DEFAULT_LIMITS.maxDepth, depth = 0): unknown {
  if (value === null || typeof value !== "object") return value;
  if (depth >= maxDepth) throw new TaskError(`invalid task message: nested deeper than ${maxDepth}`, { details: { maxDepth } });
  if (Array.isArray(value)) return value.map((item) => decodeWire(item, maxDepth, depth + 1));
  const record = value as Record<string, unknown>;
  if (Object.hasOwn(record, "$bigint") && typeof record["$bigint"] === "string") {
    if (!BIGINT.test(record["$bigint"])) throw new TaskError("invalid task message: a bigint is not a decimal integer of at most 1000 digits");
    return BigInt(record["$bigint"]);
  }
  if (Object.hasOwn(record, "$typed") && typeof record["$typed"] === "string" && typeof record["base64"] === "string") {
    const name = record["$typed"];
    if (!Object.hasOwn(TYPED, name)) throw new TaskError(`invalid task message: no typed array named ${JSON.stringify(name.slice(0, 40))}`);
    const Ctor = TYPED[name as TypedName];
    const bytes = fromBase64(record["base64"]);
    if (bytes.byteLength % Ctor.BYTES_PER_ELEMENT !== 0) {
      throw new TaskError(`invalid task message: ${bytes.byteLength} bytes is not a whole number of ${name} elements`);
    }
    return new Ctor(bytes.buffer as ArrayBuffer, 0, bytes.byteLength / Ctor.BYTES_PER_ELEMENT);
  }
  const out: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(record)) if (!UNSAFE_KEYS.has(key)) out[key] = decodeWire(item, maxDepth, depth + 1);
  return out;
}

/**
 * The id near the start of a JSON message, read without parsing it (`encodeWire` keeps `kind, id`
 * first).
 */
function peekId(text: string): number | null {
  const match = /"id"\s*:\s*(\d{1,15})[,}\s]/.exec(text.slice(0, 256));
  return match ? Number(match[1]) : null;
}

interface SocketLike {
  /** A browser WebSocket's URL; a server-side socket has none. */
  readonly url?: string | undefined;
  send(data: string): void;
  addEventListener(type: "message" | "close" | "error", listener: (event: { data?: unknown }) => void): void;
  removeEventListener(type: "message" | "close" | "error", listener: (event: { data?: unknown }) => void): void;
  close?(): void;
}

export interface WebSocketTransportOptions {
  /**
   * Socket origins (`wss://api.example.com`) the page may talk to besides its own host. A peer on
   * that socket is trusted to ask for any task the page serves on it, with the payloads
   * `TaskHost.serve` lets through.
   */
  allowedOrigins?: readonly string[];
  /**
   * Skip the URL check: a server's own end (it has no URL; the server checks the page's `Origin`
   * header at the upgrade instead) or a test over `ws:`.
   */
  insecure?: boolean;
  /** Decoding limits when the listener names none. */
  limits?: Partial<WireLimits>;
}

/** Refuse a socket that is not `wss:` to the page's host or an allowed origin. */
function checkSocket(socket: SocketLike, options: WebSocketTransportOptions): void {
  if (options.insecure) return;
  if (typeof socket.url !== "string") {
    throw new ConfigurationError("this socket has no URL to check: pass { insecure: true } for a server-side socket");
  }
  let url: URL;
  try {
    url = new URL(socket.url);
  } catch {
    throw new ConfigurationError(`the socket URL ${JSON.stringify(socket.url.slice(0, 200))} cannot be read`);
  }
  if (url.protocol !== "wss:") {
    throw new ConfigurationError(`a task socket must be wss:, not ${url.protocol} (${url.host}): pass { insecure: true } only for local tests`, {
      details: { url: url.origin },
    });
  }
  const pageHost = (globalThis as { location?: { host?: string } }).location?.host;
  const allowed = (options.allowedOrigins ?? []).some((origin) => {
    try {
      return new URL(origin).origin === url.origin;
    } catch {
      return false;
    }
  });
  if (!allowed && url.host !== pageHost) {
    throw new ConfigurationError(`the task socket goes to ${url.origin}, which is neither this page's host nor in allowedOrigins`, {
      details: { url: url.origin },
    });
  }
}

/** A WebSocket (or anything with its shape), carrying JSON text. */
export function webSocketTransport(socket: SocketLike, options: WebSocketTransportOptions = {}): Transport {
  checkSocket(socket, options);
  const defaults = { ...DEFAULT_LIMITS, ...options.limits };
  return {
    peer: "remote",
    send(message) {
      socket.send(JSON.stringify(encodeWire(message)));
    },
    listen(handler, listenOptions: ListenOptions = {}) {
      const limits = { ...defaults, ...listenOptions.limits };
      const listener = (event: { data?: unknown }) => {
        if (typeof event.data !== "string") {
          listenOptions.onInvalid?.(new TaskError("invalid task message: not text"), null);
          return;
        }
        const data = event.data;
        if (data.length > limits.maxMessageBytes) {
          listenOptions.onInvalid?.(
            new TaskError(`invalid task message: ${data.length} characters, more than the ${limits.maxMessageBytes} allowed`, {
              details: { maxMessageBytes: limits.maxMessageBytes },
            }),
            peekId(data),
          );
          return;
        }
        let decoded: unknown;
        try {
          decoded = decodeWire(JSON.parse(data), limits.maxDepth);
        } catch (error) {
          const reason = error instanceof TaskError ? error : new TaskError(`invalid task message: ${error instanceof Error ? error.message : String(error)}`);
          listenOptions.onInvalid?.(reason, peekId(data));
          return;
        }
        handler(decoded);
      };
      socket.addEventListener("message", listener);
      return () => socket.removeEventListener("message", listener);
    },
    onClose(handler) {
      let done = false;
      const ended = (event: unknown) => {
        if (done) return;
        done = true;
        const e = event as { type?: string; code?: number; reason?: string } | null;
        handler(e?.type === "error" ? "the task socket failed" : `the task socket closed${e?.code ? ` (${e.code}${e.reason ? `: ${e.reason}` : ""})` : ""}`);
      };
      socket.addEventListener("close", ended);
      socket.addEventListener("error", ended);
      return () => {
        socket.removeEventListener("close", ended);
        socket.removeEventListener("error", ended);
      };
    },
    close() {
      socket.close?.();
    },
  };
}

/** Two connected ends in one thread. Messages are delivered asynchronously, like a port. */
export function transportPair(): [Transport, Transport] {
  const listeners: [Set<(m: unknown) => void>, Set<(m: unknown) => void>] = [new Set(), new Set()];
  const end = (mine: 0 | 1): Transport => ({
    peer: "local",
    send(message: TaskMessage) {
      const theirs = listeners[mine === 0 ? 1 : 0];
      queueMicrotask(() => {
        for (const handler of theirs) handler(message);
      });
    },
    listen(handler) {
      listeners[mine].add(handler);
      return () => listeners[mine].delete(handler);
    },
  });
  return [end(0), end(1)];
}
