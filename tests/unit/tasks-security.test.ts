/**
 * What a peer can do to a page through the task layer: the flaws of the security review (E.15), each
 * as the input that used to get through.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { ConfigurationError, IntegrityError, TaskError } from "../../src/errors.js";
import { TaskClient } from "../../src/tasks/client.js";
import { TaskHost } from "../../src/tasks/host.js";
import { readMessage, type TaskMessage, type Transport } from "../../src/tasks/protocol.js";
import { decodeWire, encodeWire, transportPair, webSocketTransport } from "../../src/tasks/transports.js";
import { allowedLocations, checkRemoteLocations, pinnedLocations } from "../../src/tasks/trust.js";
import { fromPythonNames } from "../../src/tasks/wire.js";
import { createWorker } from "../../src/worker/client.js";

/** A socket-shaped object: what the page sends is recorded, what the peer sends is `deliver`ed. */
function fakeSocket(url?: string) {
  const listeners = new Map<string, Set<(e: { data?: unknown; type?: string }) => void>>();
  const sent: unknown[] = [];
  return {
    url,
    sent,
    addEventListener(type: string, l: (e: { data?: unknown }) => void) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)!.add(l);
    },
    removeEventListener(type: string, l: (e: { data?: unknown }) => void) {
      listeners.get(type)?.delete(l);
    },
    send(data: string) {
      sent.push(JSON.parse(data));
    },
    deliver(data: unknown) {
      for (const l of listeners.get("message") ?? []) l({ data });
    },
    fire(type: "close" | "error") {
      for (const l of listeners.get(type) ?? []) l({ type });
    },
    count(type: string) {
      return listeners.get(type)?.size ?? 0;
    },
  };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("decodeWire", () => {
  it("never lets a peer set a prototype", () => {
    const decoded = decodeWire(JSON.parse('{"__proto__": {"polluted": true}, "constructor": {"prototype": {"x": 1}}, "a": {"__proto__": {"admin": true}}}')) as Record<string, unknown>;
    expect(Object.getPrototypeOf(decoded)).toBe(Object.prototype);
    expect((decoded as { polluted?: unknown }).polluted).toBeUndefined();
    expect(Object.hasOwn(decoded, "constructor")).toBe(false);
    expect((decoded["a"] as { admin?: unknown }).admin).toBeUndefined();
    expect(({} as { polluted?: unknown }).polluted).toBeUndefined();
  });

  it.each([
    ["a bigint that is not a number", { $bigint: "x" }],
    ["a bigint too long to parse cheaply", { $bigint: "9".repeat(5000) }],
    ["base64 that atob refuses", { $typed: "Uint8Array", base64: "%%%" }],
    ["base64 of the wrong length", { $typed: "Uint8Array", base64: "AAA" }],
    ["bytes that are not whole float32s", { $typed: "Float32Array", base64: "AAAAAAA=" }],
    ["a typed array name that is not one", { $typed: "toString", base64: "" }],
  ])("refuses %s with a TaskError", (_, value) => {
    expect(() => decodeWire(value)).toThrow(TaskError);
  });

  it("refuses nesting deeper than the limit, before the stack does", () => {
    let deep: unknown = 1;
    for (let i = 0; i < 100_000; i += 1) deep = [deep];
    expect(() => decodeWire(deep, 32)).toThrow(TaskError);
  });

  it("still round-trips what encodeWire makes", () => {
    const value = { f: new Float64Array([1.5]), b: -12n };
    expect(decodeWire(JSON.parse(JSON.stringify(encodeWire(value))))).toEqual(value);
  });
});

describe("the envelope", () => {
  it.each([
    [null],
    [[]],
    [{ kind: "eval", id: 1 }],
    [{ kind: "call", id: 1.5, task: "x" }],
    [{ kind: "call", id: -1, task: "x" }],
    [{ kind: "call", id: "1", task: "x" }],
    [{ kind: "call", id: 1, task: 42 }],
    [{ kind: "hello", tasks: "all" }],
  ])("refuses %j", (value) => {
    expect("error" in readMessage(value)).toBe(true);
  });

  it("keeps the id of a call it refuses, so the caller can be answered", () => {
    const read = readMessage({ kind: "call", id: 9, task: 7 });
    expect("error" in read && read.id).toBe(9);
  });

  it("rebuilds a peer's error from known fields only, with a bounded chain of causes", () => {
    let error: Record<string, unknown> = { type: "X", message: "m" };
    for (let i = 0; i < 50; i += 1) error = { type: "X", message: "m", cause: error, extra: "dropped" };
    const read = readMessage({ kind: "error", id: 1, error });
    expect("message" in read).toBe(true);
    let depth = 0;
    let cause = (read as { message: Extract<TaskMessage, { kind: "error" }> }).message.error;
    while (cause.cause) {
      cause = cause.cause;
      depth += 1;
    }
    expect(depth).toBeLessThanOrEqual(8);
    expect(Object.hasOwn(cause, "extra")).toBe(false);
  });
});

describe("a host on a socket", () => {
  const serveOn = (host: TaskHost, options: Parameters<TaskHost["serve"]>[1] = {}) => {
    const socket = fakeSocket();
    host.serve(webSocketTransport(socket, { insecure: true }), options);
    return socket;
  };
  const answer = (socket: ReturnType<typeof fakeSocket>, id: number) =>
    socket.sent.find((m) => (m as { id?: number }).id === id) as { kind: string; error?: { message: string } } | undefined;

  it("answers a call it cannot decode instead of dropping it (the caller used to wait for ever)", () => {
    const socket = serveOn(new TaskHost().register("echo", async (p) => p));
    socket.deliver(JSON.stringify({ kind: "call", id: 7, task: "echo", payload: { $typed: "Float32Array", base64: "AAAAAAA=" } }));
    socket.deliver(JSON.stringify({ kind: "call", id: 8, task: "echo", payload: { $bigint: "x" } }));
    socket.deliver("null");
    socket.deliver('{"kind":"call","id":9,"task":"echo","payload":');
    expect(answer(socket, 7)?.kind).toBe("error");
    expect(answer(socket, 8)?.kind).toBe("error");
    expect(answer(socket, 9)?.kind).toBe("error");
  });

  it("survives random and truncated input within a time budget, never throwing out of the listener", () => {
    const socket = serveOn(new TaskHost().register("echo", async (p) => p));
    const good = JSON.stringify(encodeWire({ kind: "call", id: 3, task: "echo", payload: { a: [1, { b: new Float32Array([1, 2]) }], c: 5n } }));
    let seed = 1;
    const random = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
    const started = performance.now();
    for (let i = 0; i < 3000; i += 1) {
      const cut = good.slice(0, Math.floor(random() * good.length));
      const chars = [...good];
      chars[Math.floor(random() * chars.length)] = String.fromCharCode(32 + Math.floor(random() * 90));
      expect(() => socket.deliver(cut)).not.toThrow();
      expect(() => socket.deliver(chars.join(""))).not.toThrow();
    }
    expect(performance.now() - started).toBeLessThan(5000);
  });

  it("refuses a message over the size limit, answering by the id it can see", () => {
    const socket = serveOn(new TaskHost().register("echo", async (p) => p), { maxMessageBytes: 100 });
    socket.deliver(JSON.stringify({ kind: "call", id: 4, task: "echo", payload: "x".repeat(200) }));
    expect(answer(socket, 4)?.error?.message).toMatch(/more than the 100 allowed/);
  });

  it("bounds the calls one peer runs at once, and refuses a second call under a running id", async () => {
    let release: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => (release = resolve));
    const socket = serveOn(new TaskHost().register("wait", async () => gate), { maxConcurrentCalls: 2 });
    for (const id of [1, 2, 3]) socket.deliver(JSON.stringify({ kind: "call", id, task: "wait", payload: null }));
    socket.deliver(JSON.stringify({ kind: "call", id: 1, task: "wait", payload: null }));
    expect(answer(socket, 3)?.error?.message).toMatch(/too many calls/);
    expect(socket.sent.filter((m) => (m as { id?: number }).id === 1)).toHaveLength(1);
    release();
    await tick();
  });

  it.each([
    ["wasmPaths", { wasmPaths: "https://evil.example/" }],
    ["a per-file wasmPaths", { wasmPaths: { "ort-wasm-simd-threaded.jsep.mjs": "https://evil.example/x.mjs" } }],
    ["endpoint", { endpoint: "https://evil.example" }],
    ["resolverBase", { options: { resolverBase: "//evil.example/models/" } }],
    ["hub.repo", { hub: { repo: "attacker/model" } }],
    ["a model given as a URL", { model: "https://evil.example/model.onnx" }],
  ])("refuses a remote payload naming %s", (_, payload) => {
    const ran = vi.fn(async () => "ran");
    const socket = serveOn(new TaskHost().register("load", ran));
    socket.deliver(JSON.stringify({ kind: "call", id: 5, task: "load", payload }));
    expect(answer(socket, 5)?.error?.message).toMatch(/may not choose where code or models load from/);
    expect(ran).not.toHaveBeenCalled();
  });

  it("lets through the locations the page allowed, and a model by name", async () => {
    const socket = serveOn(new TaskHost().register("load", async () => "ok"), {
      allowedOrigins: ["https://models.example"],
      allowedRepos: ["Younicorn/tiny"],
    });
    socket.deliver(JSON.stringify({ kind: "call", id: 6, task: "load", payload: { model: "whisper-tiny", endpoint: "https://models.example/hf", hub: { repo: "Younicorn/tiny" } } }));
    await tick();
    expect(answer(socket, 6)).toMatchObject({ kind: "result", result: "ok" });
  });

  it("passes a local peer's locations to the handler, and tells it how far the peer is trusted", async () => {
    const [hostEnd, clientEnd] = transportPair();
    const seen: string[] = [];
    new TaskHost().register("load", async (p: { wasmPaths: string }, ctx) => (seen.push(ctx.trust), p.wasmPaths)).serve(hostEnd);
    expect(await new TaskClient(clientEnd).call("load", { wasmPaths: "/ort/" })).toBe("/ort/");
    expect(seen).toEqual(["local"]);
  });

  it("aborts what runs when the socket closes", async () => {
    let signal: AbortSignal | null = null;
    const socket = serveOn(new TaskHost().register("long", async (_: null, ctx) => {
      signal = ctx.signal;
      return new Promise(() => undefined);
    }));
    socket.deliver(JSON.stringify({ kind: "call", id: 1, task: "long", payload: null }));
    await tick();
    socket.fire("close");
    expect(signal!.aborted).toBe(true);
  });
});

describe("webSocketTransport", () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each([
    ["no URL", undefined],
    ["ws:", "ws://example.com/tasks"],
    ["another host", "wss://evil.example/tasks"],
  ])("refuses a socket with %s", (_, url) => {
    vi.stubGlobal("location", { host: "example.com", href: "https://example.com/", origin: "https://example.com" });
    expect(() => webSocketTransport(fakeSocket(url))).toThrow(ConfigurationError);
  });

  it("takes wss: to the page's host or an allowed origin, and anything when told it is a test", () => {
    vi.stubGlobal("location", { host: "example.com", href: "https://example.com/", origin: "https://example.com" });
    expect(() => webSocketTransport(fakeSocket("wss://example.com/tasks"))).not.toThrow();
    expect(() => webSocketTransport(fakeSocket("wss://api.example.org/t"), { allowedOrigins: ["wss://api.example.org"] })).not.toThrow();
    expect(() => webSocketTransport(fakeSocket("ws://localhost:5181"), { insecure: true })).not.toThrow();
  });
});

describe("the client", () => {
  it("rejects what is pending when the socket closes or fails", async () => {
    for (const event of ["close", "error"] as const) {
      const socket = fakeSocket();
      const client = new TaskClient(webSocketTransport(socket, { insecure: true }));
      const call = client.call("anything", null);
      socket.fire(event);
      await expect(call).rejects.toBeInstanceOf(TaskError);
      await expect(client.call("later", null)).rejects.toBeInstanceOf(TaskError);
    }
  });

  it("settles a call whose answer cannot be decoded", async () => {
    const socket = fakeSocket();
    const client = new TaskClient(webSocketTransport(socket, { insecure: true }));
    const call = client.call("x", null);
    socket.deliver(JSON.stringify({ kind: "result", id: 1, result: { $bigint: "nope" } }));
    await expect(call).rejects.toBeInstanceOf(TaskError);
  });

  it("removes its abort listener once the call settles", async () => {
    const [hostEnd, clientEnd] = transportPair();
    new TaskHost().register("one", async () => 1).serve(hostEnd);
    const client = new TaskClient(clientEnd);
    const signal = new AbortController().signal;
    const add = vi.spyOn(signal, "addEventListener");
    const remove = vi.spyOn(signal, "removeEventListener");
    for (let i = 0; i < 5; i += 1) await client.call("one", null, { signal });
    expect(add).toHaveBeenCalledTimes(5);
    expect(remove).toHaveBeenCalledTimes(5);
  });
});

describe("results cross copied unless the handler hands them over", () => {
  const recording = (): [Transport, Transferable[][], (m: unknown) => void] => {
    const transfers: Transferable[][] = [];
    let deliver: (m: unknown) => void = () => undefined;
    return [
      {
        send(_message, transfer) {
          transfers.push(transfer ?? []);
        },
        listen(handler) {
          deliver = handler;
          return () => undefined;
        },
      },
      transfers,
      (m) => deliver(m),
    ];
  };

  it("transfers only the buffers the handler marked", async () => {
    const kept = new Float32Array([1, 2, 3]);
    const given = new Float32Array([4]);
    const [transport, transfers, deliver] = recording();
    new TaskHost()
      .register("cached", async () => kept)
      .register("fresh", async (_, ctx) => (ctx.transfer(given.buffer as ArrayBuffer), given))
      .serve(transport);
    deliver({ kind: "call", id: 1, task: "cached", payload: null });
    deliver({ kind: "call", id: 2, task: "fresh", payload: null });
    await tick();
    expect(transfers.at(-2)).toEqual([]);
    expect(transfers.at(-1)).toEqual([given.buffer]);
  });
});

describe("pinnedLocations", () => {
  it("fixes the first request's locations and refuses others", () => {
    const pin = pinnedLocations();
    pin({ wasmPaths: "https://example.com/ort/", endpoint: undefined });
    pin({ wasmPaths: "https://example.com/ort/", endpoint: undefined });
    expect(() => pin({ wasmPaths: "https://evil.example/" })).toThrow(ConfigurationError);
    expect(() => pin({ endpoint: "https://evil.example" })).toThrow(ConfigurationError);
  });

  it.each(["data:text/javascript,alert(1)", "javascript:alert(1)", "blob:https://example.com/x"])("refuses %s as a location", (url) => {
    expect(() => pinnedLocations()({ wasmPaths: url })).toThrow(ConfigurationError);
  });
});

describe("checkRemoteLocations", () => {
  it("allows a payload that names no location", () => {
    expect(() => checkRemoteLocations({ text: "see https://example.com", model: "bge-small" }, {})).not.toThrow();
  });
});

describe("field names from Python", () => {
  it("never writes __proto__ while renaming", () => {
    const out = fromPythonNames(JSON.parse('{"__proto__": {"admin": true}, "first_step": 1}')) as Record<string, unknown>;
    expect((out as { admin?: unknown }).admin).toBeUndefined();
    expect(out["firstStep"]).toBe(1);
  });

  it("refuses a value nested past its depth", () => {
    let deep: unknown = 1;
    for (let i = 0; i < 1000; i += 1) deep = { a: deep };
    expect(() => fromPythonNames(deep)).toThrow(TaskError);
  });
});

describe("a worker from another origin", () => {
  afterEach(() => vi.unstubAllGlobals());

  const script = new TextEncoder().encode("self.postMessage('hi')");
  const digest = async () => `sha256-${btoa(String.fromCharCode(...new Uint8Array(await crypto.subtle.digest("SHA-256", script))))}`;

  class FakeWorker {
    static made: string[] = [];
    readonly listeners = new Map<string, Set<() => void>>();
    constructor(readonly url: string) {
      FakeWorker.made.push(url);
    }
    addEventListener(type: string, l: () => void) {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type)!.add(l);
    }
    removeEventListener(type: string, l: () => void) {
      this.listeners.get(type)?.delete(l);
    }
    emit(type: string) {
      for (const l of [...(this.listeners.get(type) ?? [])]) l();
    }
  }

  const stub = () => {
    FakeWorker.made = [];
    vi.stubGlobal("location", { origin: "https://app.example", href: "https://app.example/page" });
    vi.stubGlobal("Worker", FakeWorker);
    vi.stubGlobal("fetch", vi.fn(async () => new Response(script)));
  };

  it("is not run without an integrity digest", async () => {
    stub();
    await expect(createWorker("https://cdn.example/worker.js")).rejects.toBeInstanceOf(ConfigurationError);
    expect(FakeWorker.made).toEqual([]);
  });

  it("is not run when its bytes do not match the digest", async () => {
    stub();
    await expect(createWorker("https://cdn.example/worker.js", { integrity: "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=" })).rejects.toBeInstanceOf(IntegrityError);
    expect(FakeWorker.made).toEqual([]);
  });

  it("runs from a Blob when it matches, the URL revoked on its first message only", async () => {
    stub();
    const revoke = vi.spyOn(URL, "revokeObjectURL");
    const worker = (await createWorker("https://cdn.example/worker.js", { integrity: await digest() })) as unknown as FakeWorker;
    expect(worker.url).toMatch(/^blob:/);
    expect(revoke).not.toHaveBeenCalled();
    worker.emit("message");
    expect(revoke).toHaveBeenCalledWith(worker.url);
    revoke.mockRestore();
  });

  it("is loaded directly when same-origin and no digest is given", async () => {
    stub();
    const worker = (await createWorker("/assets/worker.js")) as unknown as FakeWorker;
    expect(worker.url).toBe("https://app.example/assets/worker.js");
  });
});

describe("allowedLocations", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("takes the listed origins, relative URLs on the worker's own, and nothing else", () => {
    vi.stubGlobal("location", { href: "https://app.example/assets/worker.js", origin: "https://app.example" });
    const check = allowedLocations(["https://app.example", "https://huggingface.co"]);
    expect(() => check({ wasmPaths: "/assets/", endpoint: "https://app.example/hf", server: "/models/" })).not.toThrow();
    expect(() => check({ endpoint: "https://huggingface.co" })).not.toThrow();
    expect(() => check({ endpoint: "https://evil.example" })).toThrow(ConfigurationError);
    expect(() => check({ wasmPaths: "data:text/javascript,1" })).toThrow(ConfigurationError);
  });
});
