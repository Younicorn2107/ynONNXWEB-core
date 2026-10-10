/** Content identifiers: the same input always gets the same id. */

const encoder = new TextEncoder();

function hex(bytes: Uint8Array): string {
  let out = "";
  for (const byte of bytes) out += byte.toString(16).padStart(2, "0");
  return out;
}

/** 32 hexadecimal characters identifying `data`. */
export async function hashBytes(data: ArrayBuffer | ArrayBufferView): Promise<string> {
  const view = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const digest = await crypto.subtle.digest("SHA-256", view as Uint8Array<ArrayBuffer>);
  return hex(new Uint8Array(digest, 0, 16));
}

export function hashText(text: string): Promise<string> {
  return hashBytes(encoder.encode(text));
}

/** A JSON text with sorted keys: the same parameters in any order, one text. */
export function stableJson(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) => {
    if (typeof item === "bigint") return item.toString();
    if (ArrayBuffer.isView(item)) return Array.from(item as unknown as ArrayLike<number>);
    if (item && typeof item === "object" && !Array.isArray(item)) {
      return Object.fromEntries(Object.entries(item as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
    }
    return item;
  });
}

/** The id of a set of parameters, independent of their order. */
export function hashParams(params: Record<string, unknown>): Promise<string> {
  return hashText(stableJson(params));
}

/** One id for several, order included. */
export function combineIds(ids: readonly string[]): Promise<string> {
  return hashText(ids.join("|") + "|");
}

/**
 * The id of any input the libraries take: a text, bytes, a typed array (its values, type and
 * length), a Blob (its bytes), a list (its items' ids, in order) or a record of parameters.
 */
export async function contentId(item: unknown): Promise<string> {
  if (typeof item === "string") return hashText(item);
  if (item instanceof ArrayBuffer) return hashBytes(item);
  if (ArrayBuffer.isView(item)) {
    const header = encoder.encode(`${item.constructor.name}|${item.byteLength}|`);
    const body = new Uint8Array(item.buffer, item.byteOffset, item.byteLength);
    const joined = new Uint8Array(header.length + body.length);
    joined.set(header);
    joined.set(body, header.length);
    return hashBytes(joined);
  }
  if (typeof Blob !== "undefined" && item instanceof Blob) return hashBytes(await item.arrayBuffer());
  if (Array.isArray(item)) return combineIds(await Promise.all(item.map(contentId)));
  if (item && typeof item === "object") return hashParams(item as Record<string, unknown>);
  return hashText(String(item));
}
