/** What a peer may decide, and the limits a message must stay within. */

import { ConfigurationError, TaskError } from "../errors.js";

/**
 * `local`: the page itself (its Worker, one thread). `remote`: anyone at the other end of a socket.
 */
export type Trust = "local" | "remote";

/** How much one message may hold; checked before anything is built from it. */
export interface WireLimits {
  /** The largest text a socket message may be (UTF-16 code units, as `string.length`). */
  maxMessageBytes: number;
  /** The deepest nesting of arrays and objects. */
  maxDepth: number;
}

/** Room for a few minutes of audio as base64; anything bigger is a mistake or an attack. */
export const DEFAULT_LIMITS: Readonly<WireLimits> = Object.freeze({ maxMessageBytes: 16 * 1024 * 1024, maxDepth: 32 });

/** What a remote payload may name, set by the page when it starts serving. */
export interface LocationPolicy {
  /**
   * Origins (`https://models.example.com`) a URL in a remote payload may point at. None by default.
   */
  allowedOrigins?: readonly string[];
  /** Hub repositories (`org/name`) a remote payload's `hub.repo` may name. None by default. */
  allowedRepos?: readonly string[];
}

/**
 * The payload fields that say where code or models come from, whatever the task: a field of this
 * name anywhere in a remote payload is a location.
 */
export const LOCATION_KEYS: ReadonlySet<string> = new Set([
  "wasmPaths",
  "endpoint",
  "resolverBase",
  "resolver",
  "server",
  "hub",
  "url",
  "baseUrl",
  "base",
  "modelUrl",
  "src",
  "href",
]);

/** A string a loader would take for a URL rather than a model name. */
const LOOKS_LIKE_URL = /^(?:[a-z][a-z0-9+.-]*:|\/|\\|\.\.?(?:\/|\\|$))/i;

function pageBase(): string | undefined {
  return (globalThis as { location?: { href?: string } }).location?.href;
}

function originOf(value: string): string | null {
  try {
    return new URL(value, pageBase()).origin;
  } catch {
    return null;
  }
}

function refuse(path: string, why: string): never {
  throw new TaskError(`a remote peer may not choose where code or models load from: '${path}' ${why}.`, {
    details: { field: path, reason: why },
  });
}

/** Refuse a remote payload that names a location the page did not allow. */
export function checkRemoteLocations(payload: unknown, policy: LocationPolicy, maxDepth = DEFAULT_LIMITS.maxDepth): void {
  const origins = new Set((policy.allowedOrigins ?? []).map((o) => originOf(o)).filter((o): o is string => o !== null && o !== "null"));
  const repos = new Set(policy.allowedRepos ?? []);

  const checkUrl = (value: unknown, path: string, depth: number): void => {
    if (depth > maxDepth) refuse(path, "is nested too deep");
    if (value === null || value === undefined || typeof value === "boolean") return;
    if (typeof value === "string") {
      const origin = originOf(value);
      if (origin === null || origin === "null") refuse(path, "is not a URL this page can resolve");
      if (!origins.has(origin)) refuse(path, `points at ${origin}, which this page did not allow`);
      return;
    }
    if (typeof value === "object") {
      for (const [key, item] of Object.entries(value)) checkUrl(item, `${path}.${key}`, depth + 1);
      return;
    }
    refuse(path, "is not a URL");
  };

  const walk = (value: unknown, path: string, depth: number): void => {
    if (value === null || typeof value !== "object" || ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return;
    if (depth > maxDepth) refuse(path, "is nested too deep");
    if (Array.isArray(value)) {
      value.forEach((item, i) => walk(item, `${path}[${i}]`, depth + 1));
      return;
    }
    for (const [key, item] of Object.entries(value)) {
      const here = path ? `${path}.${key}` : key;
      if (key === "hub" && item !== null && item !== undefined) {
        const repo = typeof item === "string" ? item : (item as { repo?: unknown }).repo;
        if (typeof repo !== "string" || !repos.has(repo)) refuse(here, `names repository ${JSON.stringify(repo)}, which this page did not allow`);
        if (typeof item === "object") {
          const { endpoint } = item as { endpoint?: unknown };
          if (endpoint !== undefined) checkUrl(endpoint, `${here}.endpoint`, depth + 1);
        }
        continue;
      }
      if (LOCATION_KEYS.has(key)) {
        checkUrl(item, here, depth + 1);
        continue;
      }
      if (key === "model" && typeof item === "string" && LOOKS_LIKE_URL.test(item)) refuse(here, "is a URL, not a model name");
      walk(item, here, depth + 1);
    }
  };

  walk(payload, "", 0);
}

/**
 * Every URL among `fields` (a hub's `endpoint`, but not its repo or folders), each with the field
 * it came from: an `http(s)` URL, resolved against the worker's own location — never `data:`,
 * `blob:` or `javascript:`, which would be code handed over inline.
 */
function httpUrls(fields: Record<string, unknown>): [string, URL][] {
  const out: [string, URL][] = [];
  const visit = (name: string, value: unknown, depth: number): void => {
    if (value === undefined || value === null) return;
    if (depth > 4) throw new ConfigurationError(`'${name}' is nested too deep to be a location`);
    if (typeof value === "object") {
      for (const [key, item] of Object.entries(value)) {
        if (name === "hub" && key !== "endpoint") continue;
        visit(`${name}.${key}`, item, depth + 1);
      }
      return;
    }
    if (typeof value !== "string") throw new ConfigurationError(`'${name}' must be a URL, not ${typeof value}`);
    let url: URL;
    try {
      url = new URL(value, pageBase() ?? "http://invalid/");
    } catch {
      throw new ConfigurationError(`'${name}' is not a URL: ${JSON.stringify(value.slice(0, 200))}`);
    }
    if (url.protocol !== "https:" && url.protocol !== "http:") {
      throw new ConfigurationError(`'${name}' must be an http(s) URL, not ${url.protocol}`, { details: { field: name } });
    }
    out.push([name, url]);
  };
  for (const [name, value] of Object.entries(fields)) visit(name, value, 0);
  return out;
}

/** Where a Worker loads from, fixed by the first request that says. */
export function pinnedLocations(): (fields: Record<string, unknown>) => void {
  const pinned = new Map<string, string>();
  return (fields) => {
    httpUrls(fields);
    for (const [name, value] of Object.entries(fields)) {
      const text = JSON.stringify(value ?? null);
      const before = pinned.get(name);
      if (before === undefined) pinned.set(name, text);
      else if (before !== text) {
        throw new ConfigurationError(
          `'${name}' was set by this worker's first request and cannot change (${before} → ${text}): start another worker for another source.`,
          { details: { field: name } },
        );
      }
    }
  };
}

/**
 * Where a Worker may load from, as a list of origins: for a Worker that legitimately switches
 * sources (the Hub, then the page's own model server) and so cannot pin them. Each value must be an
 * `http(s)` URL on one of `origins` (relative ones resolve against the Worker's location).
 */
export function allowedLocations(origins: readonly string[]): (fields: Record<string, unknown>) => void {
  const allowed = new Set(origins.map((o) => originOf(o)).filter((o): o is string => o !== null && o !== "null"));
  return (fields) => {
    for (const [name, url] of httpUrls(fields)) {
      if (!allowed.has(url.origin)) throw new ConfigurationError(`'${name}' points at ${url.origin}, which this worker does not load from`, { details: { field: name } });
    }
  };
}
