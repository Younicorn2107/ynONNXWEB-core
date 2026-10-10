/** How a cache entry is named, in one place. */

/** Between the model name and the precision. One character, reserved. */
export const ENTRY_SEPARATOR = "_";

/** A model name that can sit in a directory name and survive a split. */
export function safeName(name: string): string {
  return name.replace(/\//g, "--").replace(/_/g, "-");
}

/** The directory name for one model at one precision. */
export function entryName(name: string, computeType: string): string {
  return `${safeName(name)}${ENTRY_SEPARATOR}${computeType}`;
}

/** `[name, computeType]` from a directory name. */
export function splitEntry(entry: string): [string, string] {
  const at = entry.indexOf(ENTRY_SEPARATOR);
  if (at < 0) return [entry, "float32"];
  return [entry.slice(0, at), entry.slice(at + ENTRY_SEPARATOR.length)];
}
