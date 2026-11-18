/** The page side: talk to a worker as if it were a promise-returning object. */

import { ConfigurationError, IntegrityError } from "../errors.js";
import type { Logger } from "../logger.js";
import { TaskClient, type CallOptions } from "../tasks/client.js";
import { portTransport } from "../tasks/transports.js";

export interface WorkerClientOptions {
  /** URL of the bundled worker script. */
  url: string;
  logger?: Logger;
  name?: string;
  /**
   * The script's digest, as in Subresource Integrity (`sha256-<base64>`, several separated by
   * spaces). Required for a cross-origin script; given for a same-origin one, the script is fetched
   * and checked too.
   */
  integrity?: string;
}

export interface CreateWorkerOptions {
  name?: string;
  integrity?: string;
}

const ALGORITHMS: Readonly<Record<string, string>> = { sha256: "SHA-256", sha384: "SHA-384", sha512: "SHA-512" };

function base64(bytes: ArrayBuffer): string {
  let binary = "";
  for (const byte of new Uint8Array(bytes)) binary += String.fromCharCode(byte);
  return btoa(binary);
}

/**
 * Whether `bytes` match one of the SRI digests in `integrity`; throws on a list with none it can
 * check.
 */
export async function matchesIntegrity(bytes: ArrayBuffer, integrity: string): Promise<boolean> {
  const wanted = integrity
    .trim()
    .split(/\s+/)
    .map((token) => /^(sha256|sha384|sha512)-([A-Za-z0-9+/]+={0,2})$/.exec(token))
    .filter((match): match is RegExpExecArray => match !== null);
  if (wanted.length === 0) throw new ConfigurationError(`integrity ${JSON.stringify(integrity.slice(0, 120))} names no sha256-, sha384- or sha512- digest`);
  for (const [, algorithm, digest] of wanted) {
    if (base64(await crypto.subtle.digest(ALGORITHMS[algorithm!]!, bytes)) === digest) return true;
  }
  return false;
}

function pageOrigin(): string | undefined {
  return (globalThis as { location?: { origin?: string } }).location?.origin;
}

/** Load a worker: directly when the script is same-origin, from a verified Blob otherwise. */
export async function createWorker(url: string, options: CreateWorkerOptions = {}): Promise<Worker> {
  const workerOptions: WorkerOptions = { type: "module", ...(options.name ? { name: options.name } : {}) };
  const origin = pageOrigin();
  const resolved = new URL(url, (globalThis as { location?: { href?: string } }).location?.href);
  const crossOrigin = origin === undefined || resolved.origin !== origin;
  if (!crossOrigin && !options.integrity) return new Worker(resolved.href, workerOptions);
  if (!options.integrity) {
    throw new ConfigurationError(
      `the worker script at ${resolved.href} is cross-origin: it would run from a Blob URL with this page's origin, ` +
        `so it needs an integrity digest (integrity: "sha256-…") to be run at all.`,
      { details: { url: resolved.href } },
    );
  }
  const response = await fetch(resolved.href, { credentials: "omit" });
  if (!response.ok) {
    throw new ConfigurationError(
      `the worker script at ${resolved.href} could not be loaded (HTTP ${response.status}). ` +
        `When static files are served from another origin, the script is fetched ` +
        `and run from a Blob URL, so it must also be readable by CORS.`,
    );
  }
  const bytes = await response.arrayBuffer();
  if (!(await matchesIntegrity(bytes, options.integrity))) {
    throw new IntegrityError(`the worker script at ${resolved.href} does not match its integrity digest: not run`, { details: { url: resolved.href } });
  }
  const blobUrl = URL.createObjectURL(new Blob([bytes], { type: "text/javascript" }));
  let worker: Worker;
  try {
    worker = new Worker(blobUrl, workerOptions);
  } catch (error) {
    URL.revokeObjectURL(blobUrl);
    throw error;
  }
  
  
  
  const revoke = () => {
    URL.revokeObjectURL(blobUrl);
    worker.removeEventListener("message", revoke);
    worker.removeEventListener("error", revoke);
  };
  worker.addEventListener("message", revoke);
  worker.addEventListener("error", revoke);
  return worker;
}

export class WorkerClient {
  #client: TaskClient | null = null;
  #worker: Worker | null = null;
  /** The worker being made: two calls before it exists share it rather than starting two. */
  #connecting: Promise<TaskClient> | null = null;

  constructor(private readonly options: WorkerClientOptions) {}

  #connect(): Promise<TaskClient> {
    if (this.#client) return Promise.resolve(this.#client);
    this.#connecting ??= this.#open().finally(() => (this.#connecting = null));
    return this.#connecting;
  }

  async #open(): Promise<TaskClient> {
    const worker = await createWorker(this.options.url, {
      ...(this.options.name ? { name: this.options.name } : {}),
      ...(this.options.integrity ? { integrity: this.options.integrity } : {}),
    });
    const client = new TaskClient(portTransport(worker as unknown as Parameters<typeof portTransport>[0]));
    worker.addEventListener("error", (event) => {
      this.options.logger?.error(`the inference worker failed: ${event.message}`);
      client.close(`the inference worker failed: ${event.message}`);
      this.#client = null;
      this.#worker = null;
    });
    this.#worker = worker;
    this.#client = client;
    return client;
  }

  /** Call a task by name and await its result. */
  async call<R = unknown>(task: string, payload: unknown, options: CallOptions = {}): Promise<R> {
    return (await this.#connect()).call<R>(task, payload, options);
  }

  /** Call a task and read what it emits as it comes. */
  async *stream<C = unknown, R = unknown>(task: string, payload: unknown, options: Omit<CallOptions, "onChunk"> = {}): AsyncGenerator<C, R, undefined> {
    return yield* (await this.#connect()).stream<C, R>(task, payload, options);
  }

  /** Stop the worker. Pending calls reject rather than hang for ever. */
  terminate(): void {
    this.#client?.close("the inference worker was terminated");
    this.#client = null;
    this.#worker?.terminate();
    this.#worker = null;
  }
}
