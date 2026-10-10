import { describe, expect, it } from "vitest";

import { CancelledError, ModelNotFoundError, TaskError } from "../../src/errors.js";
import { TaskClient } from "../../src/tasks/client.js";
import { TaskHost } from "../../src/tasks/host.js";
import { decodeWire, encodeWire, transportPair, webSocketTransport } from "../../src/tasks/transports.js";

function connect(host: TaskHost): TaskClient {
  const [hostEnd, clientEnd] = transportPair();
  const client = new TaskClient(clientEnd);
  host.serve(hostEnd);
  return client;
}

describe("tasks over a transport", () => {
  it("announces its tasks and answers a call", async () => {
    const host = new TaskHost().register("add", async ({ a, b }: { a: number; b: number }) => a + b);
    const client = connect(host);
    expect(await client.ready()).toEqual(["add"]);
    expect(await client.call("add", { a: 2, b: 3 })).toBe(5);
  });

  it("reports progress and streams chunks before the result", async () => {
    const host = new TaskHost().register("count", async (n: number, context) => {
      for (let i = 0; i < n; i += 1) {
        context.progress(i / n);
        context.emit(i);
      }
      return "done";
    });
    const client = connect(host);
    const progress: unknown[] = [];
    const stream = client.stream<number, string>("count", 3, { onProgress: (p) => progress.push(p) });
    const chunks: number[] = [];
    let next = await stream.next();
    while (!next.done) {
      chunks.push(next.value);
      next = await stream.next();
    }
    expect(chunks).toEqual([0, 1, 2]);
    expect(next.value).toBe("done");
    expect(progress).toEqual([0, 1 / 3, 2 / 3]);
  });

  it("brings an error back as the same class", async () => {
    const host = new TaskHost().register("load", async () => {
      throw new ModelNotFoundError("no such model");
    });
    const client = connect(host);
    await expect(client.call("load", null)).rejects.toBeInstanceOf(ModelNotFoundError);
    await expect(client.call("nope", null)).rejects.toBeInstanceOf(TaskError);
  });

  it("cancels a running task from the caller's side", async () => {
    let sawAbort = false;
    const host = new TaskHost().register("long", async (_: null, context) => {
      await new Promise<void>((resolve) => context.signal.addEventListener("abort", () => resolve()));
      sawAbort = true;
      throw context.signal.reason;
    });
    const client = connect(host);
    const controller = new AbortController();
    const call = client.call("long", null, { signal: controller.signal });
    setTimeout(() => controller.abort(), 5);
    await expect(call).rejects.toBeInstanceOf(CancelledError);
    expect(sawAbort).toBe(true);
  });
});

describe("the WebSocket wire", () => {
  it("round-trips typed arrays and bigints through JSON", () => {
    const value = {
      vectors: new Float32Array([1.5, -2, 3.25]),
      ids: new BigInt64Array([1n, 50257n]),
      big: 9007199254740993n,
      nested: [{ mask: new Uint8Array([1, 0, 1]) }],
    };
    const back = decodeWire(JSON.parse(JSON.stringify(encodeWire(value)))) as typeof value;
    expect(back.vectors).toEqual(value.vectors);
    expect(back.ids).toEqual(value.ids);
    expect(back.big).toBe(value.big);
    expect(back.nested[0]!.mask).toEqual(value.nested[0]!.mask);
  });

  it("carries a whole call over a socket-shaped channel", async () => {
    const make = () => {
      const listeners = new Set<(e: { data: unknown }) => void>();
      return {
        peer: null as null | { deliver(data: string): void },
        deliver(data: string) {
          for (const l of listeners) l({ data });
        },
        send(data: string) {
          queueMicrotask(() => this.peer!.deliver(data));
        },
        addEventListener(_: "message", l: (e: { data: unknown }) => void) {
          listeners.add(l);
        },
        removeEventListener(_: "message", l: (e: { data: unknown }) => void) {
          listeners.delete(l);
        },
      };
    };
    const server = make();
    const page = make();
    server.peer = page;
    page.peer = server;

    const host = new TaskHost().register("embed", async (texts: string[]) => new Float32Array(texts.map((t) => t.length)));
    host.serve(webSocketTransport(page));
    const client = new TaskClient(webSocketTransport(server));
    const result = await client.call<Float32Array>("embed", ["ab", "abcd"]);
    expect(result).toEqual(new Float32Array([2, 4]));
  });
});
