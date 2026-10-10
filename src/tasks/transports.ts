/** The channels a task travels on. */

import type { TaskMessage, Transport } from "./protocol.js";

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
    send(message, transfer) {
      if (transfer && transfer.length > 0) port.postMessage(message, transfer);
      else port.postMessage(message);
    },
    listen(handler) {
      const listener = (event: MessageEvent) => handler(event.data as TaskMessage);
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

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

function fromBase64(text: string): Uint8Array {
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
    const name = value.constructor.name as TypedName;
    if (name in TYPED) {
      return { $typed: name, base64: toBase64(new Uint8Array(value.buffer, value.byteOffset, value.byteLength)) };
    }
  }
  if (value instanceof ArrayBuffer) return { $typed: "Uint8Array", base64: toBase64(new Uint8Array(value)) };
  if (Array.isArray(value)) return value.map(encodeWire);
  if (value instanceof Map) return Object.fromEntries([...value].map(([k, v]) => [String(k), encodeWire(v)]));
  const out: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) if (item !== undefined) out[key] = encodeWire(item);
  return out;
}

/** The inverse of `encodeWire`. */
export function decodeWire(value: unknown): unknown {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(decodeWire);
  const record = value as Record<string, unknown>;
  if (typeof record["$bigint"] === "string") return BigInt(record["$bigint"]);
  if (typeof record["$typed"] === "string" && typeof record["base64"] === "string") {
    const Ctor = TYPED[record["$typed"] as TypedName];
    if (Ctor) {
      const bytes = fromBase64(record["base64"]);
      return new Ctor(bytes.buffer as ArrayBuffer, 0, bytes.byteLength / Ctor.BYTES_PER_ELEMENT);
    }
  }
  const out: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(record)) out[key] = decodeWire(item);
  return out;
}

interface SocketLike {
  send(data: string): void;
  addEventListener(type: "message", listener: (event: { data: unknown }) => void): void;
  removeEventListener(type: "message", listener: (event: { data: unknown }) => void): void;
  close?(): void;
}

/** A WebSocket (or anything with its shape), carrying JSON text. */
export function webSocketTransport(socket: SocketLike): Transport {
  return {
    send(message) {
      socket.send(JSON.stringify(encodeWire(message)));
    },
    listen(handler) {
      const listener = (event: { data: unknown }) => {
        if (typeof event.data !== "string") return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(event.data);
        } catch {
          return;
        }
        handler(decodeWire(parsed) as TaskMessage);
      };
      socket.addEventListener("message", listener);
      return () => socket.removeEventListener("message", listener);
    },
    close() {
      socket.close?.();
    },
  };
}

/** Two connected ends in one thread. Messages are delivered asynchronously, like a port. */
export function transportPair(): [Transport, Transport] {
  const listeners: [Set<(m: TaskMessage) => void>, Set<(m: TaskMessage) => void>] = [new Set(), new Set()];
  const end = (mine: 0 | 1): Transport => ({
    send(message) {
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
