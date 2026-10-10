/**
 * Text files with their structure kept: a Markdown front matter, subtitle cues. The port of
 * `ynlibs.text.documents.split_front_matter` and `ynlibs.text.subtitles.map_cues`.
 */

const FRONT_MATTER = /^---\r?\n([\s\S]*?)\r?\n---\r?\n/;
const KEY_VALUE = /^([\p{L}\p{N}_-]+)\s*:\s*(.+)$/u;
/** Python's `str.splitlines` boundaries. */
// eslint-disable-next-line no-control-regex -- the file/group/record separators are line boundaries to Python
const LINE_BREAK = /\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u{2028}\u{2029}]/u;

/** Extensions {@link mapCues} reads. */
export const SUBTITLE_EXTENSIONS: ReadonlySet<string> = new Set([".srt", ".vtt"]);

/** Python's `str.splitlines()`: no trailing empty line, every boundary it knows. */
export function splitLines(text: string): string[] {
  if (!text) return [];
  const lines = text.split(new RegExp(LINE_BREAK.source, "gu"));
  if (lines[lines.length - 1] === "") lines.pop();
  return lines;
}

/** Python's `str.splitlines(keepends=True)`. */
export function splitLinesKeepEnds(text: string): string[] {
  const out: string[] = [];
  const pattern = new RegExp(LINE_BREAK.source, "gu");
  let start = 0;
  for (const match of text.matchAll(pattern)) {
    const end = match.index + match[0].length;
    out.push(text.slice(start, end));
    start = end;
  }
  if (start < text.length) out.push(text.slice(start));
  return out;
}

const stripQuotes = (value: string) => value.replace(/^["']+|["']+$/g, "");

/** `[body, metadata]`: a leading YAML block of `key: value` lines, scalars only. */
export function splitFrontMatter(text: string): [string, Record<string, string>] {
  const match = FRONT_MATTER.exec(text);
  if (!match) return [text, {}];
  const metadata: Record<string, string> = {};
  for (const line of splitLines(match[1]!)) {
    const found = KEY_VALUE.exec(line.trim());
    if (found) metadata[found[1]!] = stripQuotes(found[2]!.trim());
  }
  return [text.slice(match[0].length), metadata];
}

/**
 * `content` with every cue's text replaced by `change` of it, called ONCE with all the cues' texts
 * (so they batch). Numbering, timings, blank lines and line endings are kept. Returns `[content,
 * cues]`.
 */
export async function mapCues(content: string, change: (texts: string[]) => string[] | Promise<string[]>): Promise<[string, number]> {
  const newline = content.includes("\r\n") ? "\r\n" : "\n";
  const blocks = content.split(/(\r?\n\s*\r?\n)/);
  const cues: [number, string[], number][] = [];
  for (let index = 0; index < blocks.length; index += 2) {
    const lines = splitLines(blocks[index]!);
    const timing = lines.findIndex((line) => line.includes("-->"));
    if (timing >= 0 && timing + 1 < lines.length) cues.push([index, lines, timing + 1]);
  }
  if (cues.length === 0) return [content, 0];
  const texts = cues.map(([, lines, first]) => lines.slice(first).join("\n"));
  const changed = await change(texts);
  if (changed.length !== texts.length) throw new Error(`${texts.length} cues but ${changed.length} texts back`);
  cues.forEach(([index, lines, first], i) => {
    const block = blocks[index]!;
    const ending = block.slice(block.replace(/[\r\n]+$/, "").length);
    blocks[index] = [...lines.slice(0, first), ...splitLines(changed[i]!)].join(newline) + ending;
  });
  return [blocks.join(""), cues.length];
}
