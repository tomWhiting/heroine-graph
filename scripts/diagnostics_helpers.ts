/**
 * Parsers for `scripts/diagnostics.sh`.
 *
 * Each subcommand reads a tool's human/JSON output and writes newline-delimited
 * JSON diagnostics to stdout. Nothing else is ever written to stdout.
 *
 *   deno run --allow-read scripts/diagnostics_helpers.ts deno-check   < stderr.txt
 *   deno run --allow-read scripts/diagnostics_helpers.ts deno-lint    < lint.json
 *   deno run --allow-read scripts/diagnostics_helpers.ts deno-fmt     < fmt.txt
 *   deno run --allow-read scripts/diagnostics_helpers.ts deno-test-junit report.xml
 */

// deno-lint-ignore no-control-regex -- matching the ANSI escape byte is the point
const ANSI = /\u001b\[[0-9;]*m/g;

function stripAnsi(s: string): string {
  return s.replace(ANSI, "");
}

function emit(obj: Record<string, unknown>): void {
  console.log(JSON.stringify(obj));
}

async function readStdin(): Promise<string> {
  const chunks: Uint8Array[] = [];
  const buf = new Uint8Array(65536);
  while (true) {
    const n = await Deno.stdin.read(buf);
    if (n === null) break;
    chunks.push(buf.slice(0, n));
  }
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return new TextDecoder().decode(out);
}

/** `file:///abs/path` -> `/abs/path`; anything else is returned unchanged. */
function toPath(loc: string): string {
  if (!loc.startsWith("file://")) return loc;
  try {
    return decodeURIComponent(new URL(loc).pathname);
  } catch {
    return loc;
  }
}

// ---------------------------------------------------------------------------
// deno check
// ---------------------------------------------------------------------------

// `TS2304 [ERROR]: Cannot find name 'x'.`
const TS_HEADER = /^(TS\d+) \[([A-Z]+)\]: (.*)$/;
// `error: Relative import path "x" not prefixed with / or ./ or ../`
const GENERIC_HEADER = /^error: (.*)$/;
// `    at file:///abs/path.ts:12:5`
const AT_LINE = /^\s+at (\S+?):(\d+):(\d+)\s*$/;
// A squiggle/caret underline: the line right above it is the source snippet.
const CARET_LINE = /^\s*[~^]+\s*$/;
// Terminal summary lines that are not diagnostics of their own.
const NOT_A_DIAGNOSTIC = /^(Type checking failed\.|Found \d+ errors?\.)$/;

function parseDenoCheck(input: string): void {
  const lines = stripAnsi(input).split("\n");

  type Header = { index: number; lint: string; level: string; message: string };
  const headers: Header[] = [];

  for (let i = 0; i < lines.length; i++) {
    const ts = TS_HEADER.exec(lines[i]);
    if (ts) {
      headers.push({
        index: i,
        lint: ts[1],
        level: ts[2].toLowerCase().startsWith("warn") ? "warning" : "error",
        message: ts[3],
      });
      continue;
    }
    const generic = GENERIC_HEADER.exec(lines[i]);
    if (generic && !NOT_A_DIAGNOSTIC.test(generic[1].trim())) {
      headers.push({ index: i, lint: "deno-error", level: "error", message: generic[1] });
    }
  }

  for (let h = 0; h < headers.length; h++) {
    const header = headers[h];
    const end = h + 1 < headers.length ? headers[h + 1].index : lines.length;
    const block = lines.slice(header.index, end);

    let file = "<unknown>";
    let line = 0;
    let column = 0;
    let atIdx = -1;
    for (let i = 1; i < block.length; i++) {
      const at = AT_LINE.exec(block[i]);
      if (at) {
        file = toPath(at[1]);
        line = Number(at[2]);
        column = Number(at[3]);
        atIdx = i;
        break;
      }
    }

    // A generic `error:` header with no location is usually a summary line, not
    // a diagnostic; drop it rather than emit a locationless entry.
    if (header.lint === "deno-error" && atIdx === -1) continue;

    // Block layout is: header, [message continuation...], source snippet, caret,
    // `    at ...`. The snippet is indented exactly like continuation lines, so
    // the caret line is the only reliable delimiter.
    const tail = atIdx === -1 ? block.length : atIdx;
    let stop = tail;
    for (let i = tail - 1; i > 0; i--) {
      if (CARET_LINE.test(block[i]) && block[i].trim().length > 0) {
        stop = i - 1; // drop the caret and the source snippet above it
        break;
      }
    }
    const continuation = block.slice(1, Math.max(1, stop))
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    emit({
      type: "deno-check",
      level: header.level,
      file,
      line,
      column,
      lint: header.lint,
      message: [header.message, ...continuation].join(" ").trim(),
    });
  }
}

// ---------------------------------------------------------------------------
// deno lint --json
// ---------------------------------------------------------------------------

interface LintDiagnostic {
  filename?: string;
  range?: { start?: { line?: number; col?: number } };
  code?: string;
  message?: string;
}

interface LintError {
  "file_path"?: string;
  filename?: string;
  file?: string;
  message?: string;
}

interface LintReport {
  diagnostics?: LintDiagnostic[];
  errors?: LintError[];
}

function parseDenoLint(input: string): void {
  const text = input.trim();
  if (text.length === 0) return;
  let root: LintReport;
  try {
    root = JSON.parse(text) as LintReport;
  } catch {
    return;
  }

  for (const d of root?.diagnostics ?? []) {
    const start = d.range?.start;
    emit({
      type: "deno-lint",
      level: "warning",
      file: toPath(d.filename ?? "<unknown>"),
      line: start?.line ?? 0,
      // deno reports a 0-based column; report the 1-based one editors show.
      column: (start?.col ?? 0) + 1,
      lint: d.code ?? "unknown",
      message: d.message ?? "",
    });
  }

  for (const e of root?.errors ?? []) {
    emit({
      type: "deno-lint",
      level: "error",
      file: toPath(e.file_path ?? e.filename ?? e.file ?? "<unknown>"),
      line: 0,
      column: 0,
      lint: "lint-error",
      message: e.message ?? JSON.stringify(e),
    });
  }
}

// ---------------------------------------------------------------------------
// deno fmt --check
// ---------------------------------------------------------------------------

const FMT_FROM = /^from (.+):$/;
const FMT_ERROR = /^Error checking: (.+)$/;

function parseDenoFmt(input: string): void {
  for (const raw of stripAnsi(input).split("\n")) {
    const lineText = raw.trimEnd();
    const from = FMT_FROM.exec(lineText);
    if (from) {
      emit({
        type: "deno-fmt",
        level: "warning",
        file: toPath(from[1]),
        message: "file is not formatted",
      });
      continue;
    }
    const err = FMT_ERROR.exec(lineText);
    if (err) {
      emit({
        type: "deno-fmt",
        level: "error",
        file: toPath(err[1]),
        message: "file could not be parsed for formatting",
      });
    }
  }
}

// ---------------------------------------------------------------------------
// deno test --junit-path
// ---------------------------------------------------------------------------

function unescapeXml(s: string): string {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(Number(d)))
    .replace(/&amp;/g, "&");
}

function attr(tag: string, name: string): string | undefined {
  const m = new RegExp(`\\b${name}="([^"]*)"`).exec(tag);
  return m ? unescapeXml(m[1]) : undefined;
}

const TESTCASE = /<testcase\b([^>]*?)(\/>|>([\s\S]*?)<\/testcase>)/g;
const FAILURE = /<(failure|error)\b([^>]*?)(\/>|>([\s\S]*?)<\/\1>)/;
const SYSTEM_OUT = /<system-out>([\s\S]*?)<\/system-out>/;
const SKIPPED = /<skipped\b/;

function parseJunit(xml: string): void {
  let passed = 0;
  let failed = 0;
  let skipped = 0;

  TESTCASE.lastIndex = 0;
  for (const match of xml.matchAll(TESTCASE)) {
    const attrs = match[1] ?? "";
    const body = match[3] ?? "";
    const name = attr(attrs, "name") ?? "<unnamed>";
    const classname = attr(attrs, "classname");
    const test = classname ? `${classname} > ${name}` : name;

    if (SKIPPED.test(body)) {
      skipped++;
      continue;
    }

    const fail = FAILURE.exec(body);
    if (!fail) {
      passed++;
      continue;
    }
    failed++;

    const message = attr(fail[2] ?? "", "message") ?? "test failed";
    const detail = stripAnsi(unescapeXml(fail[4] ?? "")).trim();
    const sysOut = SYSTEM_OUT.exec(body);
    const captured = sysOut ? stripAnsi(unescapeXml(sysOut[1])).trim() : "";
    const stdout = [detail, captured].filter((s) => s.length > 0).join("\n");

    emit({ type: "deno-test", test, message: stripAnsi(message), stdout });
  }

  emit({ type: "summary", section: "deno-test", passed, failed, skipped });
}

// ---------------------------------------------------------------------------

async function main(): Promise<number> {
  const [subcommand, ...rest] = Deno.args;
  switch (subcommand) {
    case "deno-check":
      parseDenoCheck(await readStdin());
      return 0;
    case "deno-lint":
      parseDenoLint(await readStdin());
      return 0;
    case "deno-fmt":
      parseDenoFmt(await readStdin());
      return 0;
    case "deno-test-junit": {
      const path = rest[0];
      if (!path) {
        console.error("deno-test-junit requires a path to a JUnit XML file");
        return 2;
      }
      let xml: string;
      try {
        xml = await Deno.readTextFile(path);
      } catch {
        return 0; // no report produced; the caller already noted the failure
      }
      if (xml.trim().length === 0) return 0;
      parseJunit(xml);
      return 0;
    }
    default:
      console.error(
        "usage: diagnostics_helpers.ts <deno-check|deno-lint|deno-fmt|deno-test-junit> [args]",
      );
      return 2;
  }
}

if (import.meta.main) {
  Deno.exit(await main());
}
