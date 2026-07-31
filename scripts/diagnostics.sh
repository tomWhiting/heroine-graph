#!/usr/bin/env bash
#
# Repo-wide diagnostics as newline-delimited JSON (one object per line, stdout only).
#
# Sections and their shapes:
#   {"type":"cargo-check","level","file","line","column","lint","message"}
#   {"type":"clippy",     "level","file","line","column","lint","message"}
#   {"type":"nextest",    "test","stdout","message"}          # or "cargo-test" fallback
#   {"type":"deno-check", "level","file","line","column","lint","message"}
#   {"type":"deno-lint",  "level","file","line","column","lint","message"}
#   {"type":"deno-fmt",   "level","file","message"}
#   {"type":"deno-test",  "test","message","stdout"}
#   {"type":"summary","section":"deno-test","passed","failed","skipped"}
#   {"type":"section-skipped","section":"...","reason":"..."}
#
# Usage: scripts/diagnostics.sh [--rust-only|--deno-only] [--no-tests]
# Exit:  0 when there are no error-level diagnostics and no test failures, else 1.
#
# Deliberately NOT `set -e`: one broken tool must not stop the rest of the sweep.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WASM_DIR="$REPO_ROOT/packages/wasm"
HELPER="$REPO_ROOT/scripts/diagnostics_helpers.ts"

export NO_COLOR=1
export CLICOLOR=0
export TERM=dumb

RUN_RUST=1
RUN_DENO=1
RUN_TESTS=1

usage() {
  cat >&2 <<'EOF'
usage: scripts/diagnostics.sh [options]

  --rust-only   only the Cargo sections (packages/wasm)
  --deno-only   only the Deno/TypeScript sections (repo root)
  --no-tests    checks and lints only; skip the test suites
  -h, --help    this message

Emits newline-delimited JSON diagnostics on stdout; all human output is discarded.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --rust-only) RUN_DENO=0 ;;
    --deno-only) RUN_RUST=0 ;;
    --no-tests) RUN_TESTS=0 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "diagnostics.sh: unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/graphmother-diag.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
ALL="$WORK/all.jsonl"
: >"$ALL"

# Print a section's JSONL to stdout and keep a copy for the exit-code tally.
flush() {
  local file="$1"
  [ -s "$file" ] || return 0
  cat "$file"
  cat "$file" >>"$ALL"
}

note_skip() {
  printf '{"type":"section-skipped","section":"%s","reason":"%s"}\n' "$1" "$2" | tee -a "$ALL"
}

have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# Rust (packages/wasm is the only Cargo workspace)
# ---------------------------------------------------------------------------

# Shared jq program for `cargo --message-format=json` compiler messages.
CARGO_MSG_JQ='
  fromjson? // empty
  | select(.reason == "compiler-message")
  | .message as $m
  | ([$m.spans[]? | select(.is_primary)] | first) as $s
  | ($s.file_name // ($m.spans[0]?.file_name) // "<unknown>") as $f
  | {
      type: $TYPE,
      level: $m.level,
      file: (if ($f | startswith("/")) then $f else $ROOT + $f end),
      line: ($s.line_start // ($m.spans[0]?.line_start) // 0),
      column: ($s.column_start // ($m.spans[0]?.column_start) // 0),
      lint: ($m.code.code // "compile-error"),
      message: $m.message
    }
'

run_rust() {
  if ! have cargo; then
    note_skip "rust" "cargo not found on PATH"
    return
  fi
  if ! have jq; then
    note_skip "rust" "jq not found on PATH"
    return
  fi
  if [ ! -f "$WASM_DIR/Cargo.toml" ]; then
    note_skip "rust" "no Cargo workspace at packages/wasm"
    return
  fi

  # --- cargo check ---------------------------------------------------------
  (cd "$WASM_DIR" && cargo check --workspace --all-targets --message-format=json --keep-going 2>/dev/null) |
    jq -cR --arg TYPE "cargo-check" --arg ROOT "$WASM_DIR/" "$CARGO_MSG_JQ" >"$WORK/cargo-check.jsonl" 2>/dev/null
  flush "$WORK/cargo-check.jsonl"

  # --- cargo clippy --------------------------------------------------------
  if (cd "$WASM_DIR" && cargo clippy --version >/dev/null 2>&1); then
    (cd "$WASM_DIR" && cargo clippy --workspace --all-targets --message-format=json --keep-going -- -D warnings 2>/dev/null) |
      jq -cR --arg TYPE "clippy" --arg ROOT "$WASM_DIR/" "$CARGO_MSG_JQ" >"$WORK/clippy.jsonl" 2>/dev/null
    flush "$WORK/clippy.jsonl"
  else
    note_skip "clippy" "cargo clippy not installed"
  fi

  [ "$RUN_TESTS" -eq 1 ] || return

  # --- tests: nextest when available, plain `cargo test` otherwise ---------
  if have cargo-nextest || (cd "$WASM_DIR" && cargo nextest --version >/dev/null 2>&1); then
    (cd "$WASM_DIR" && NEXTEST_EXPERIMENTAL_LIBTEST_JSON=1 cargo nextest run \
      --workspace --all-targets --cargo-quiet \
      --message-format libtest-json-plus --no-fail-fast 2>/dev/null) |
      jq -cR 'fromjson? // empty
              | select(.type == "test" and .event == "failed")
              | {type: "nextest", test: .name, stdout: (.stdout // ""), message: "test failed"}' \
        >"$WORK/nextest.jsonl" 2>/dev/null
    flush "$WORK/nextest.jsonl"
  else
    # Fallback: no stable machine-readable format exists for `cargo test`
    # (libtest JSON is nightly-only), so scrape the one line per failing test.
    (cd "$WASM_DIR" && cargo test --workspace --all-targets --no-fail-fast 2>&1) |
      grep -E '^test .+ \.\.\. FAILED' |
      sed -E 's/^test (.+) \.\.\. FAILED.*$/\1/' |
      jq -cR '{type: "cargo-test", test: ., stdout: "", message: "test failed"}' \
        >"$WORK/cargo-test.jsonl" 2>/dev/null
    flush "$WORK/cargo-test.jsonl"
  fi
}

# ---------------------------------------------------------------------------
# Deno / TypeScript (repo root)
# ---------------------------------------------------------------------------

run_deno() {
  if ! have deno; then
    note_skip "deno" "deno not found on PATH"
    return
  fi
  if [ ! -f "$HELPER" ]; then
    note_skip "deno" "missing scripts/diagnostics_helpers.ts"
    return
  fi

  # --- deno check (mirrors `deno task check`) ------------------------------
  (cd "$REPO_ROOT" && deno check packages/core/mod.ts) >/dev/null 2>"$WORK/deno-check.txt"
  deno run --quiet --allow-read "$HELPER" deno-check <"$WORK/deno-check.txt" \
    >"$WORK/deno-check.jsonl" 2>/dev/null
  flush "$WORK/deno-check.jsonl"

  # --- deno lint (native JSON) ---------------------------------------------
  (cd "$REPO_ROOT" && deno lint --json) >"$WORK/deno-lint.json" 2>/dev/null
  deno run --quiet --allow-read "$HELPER" deno-lint <"$WORK/deno-lint.json" \
    >"$WORK/deno-lint.jsonl" 2>/dev/null
  flush "$WORK/deno-lint.jsonl"

  # --- deno fmt --check ----------------------------------------------------
  (cd "$REPO_ROOT" && deno fmt --check) >"$WORK/deno-fmt.txt" 2>&1
  deno run --quiet --allow-read "$HELPER" deno-fmt <"$WORK/deno-fmt.txt" \
    >"$WORK/deno-fmt.jsonl" 2>/dev/null
  flush "$WORK/deno-fmt.jsonl"

  [ "$RUN_TESTS" -eq 1 ] || return

  # --- deno test (JUnit XML -> JSONL) --------------------------------------
  # JUnit goes to a file, not stdout: tests print to stdout too, which would
  # interleave with the XML. Skipped tests (e.g. no GPU) are never failures.
  local junit="$WORK/deno-test.xml"
  (cd "$REPO_ROOT" && deno test --allow-read --allow-net --allow-env \
    --unstable-webgpu --junit-path="$junit" tests/) >/dev/null 2>"$WORK/deno-test.err"
  if [ -s "$junit" ]; then
    deno run --quiet --allow-read "$HELPER" deno-test-junit "$junit" \
      >"$WORK/deno-test.jsonl" 2>/dev/null
    flush "$WORK/deno-test.jsonl"
    return
  fi

  # No report: the runner never reached the tests, almost always because
  # `deno test` type-checks the suite first. Surface those TS errors (they are
  # in tests/, which the `deno check packages/core/mod.ts` pass never sees)
  # instead of reporting a bare "it did not run".
  deno run --quiet --allow-read "$HELPER" deno-check <"$WORK/deno-test.err" \
    >"$WORK/deno-test-check.jsonl" 2>/dev/null
  if [ -s "$WORK/deno-test-check.jsonl" ]; then
    flush "$WORK/deno-test-check.jsonl"
    note_skip "deno-test" "tests did not run: type-check of tests/ failed"
  else
    note_skip "deno-test" "no JUnit report produced (test runner failed to start)"
  fi
}

[ "$RUN_RUST" -eq 1 ] && run_rust
[ "$RUN_DENO" -eq 1 ] && run_deno

# ---------------------------------------------------------------------------
# Exit code: error-level diagnostics or test failures -> 1
# ---------------------------------------------------------------------------

FAILURES=""
if have jq; then
  FAILURES="$(jq -s '[ .[]
      | select((.level? == "error")
               or (.type? == "nextest")
               or (.type? == "cargo-test")
               or (.type? == "deno-test")) ] | length' "$ALL" 2>/dev/null)"
fi
# No jq, or jq choked on the accumulated file: fall back to a textual count so a
# real failure is never silently downgraded to a clean exit.
case "$FAILURES" in
  '' | *[!0-9]*)
    FAILURES="$(grep -cE '"level":"error"|"type":"(nextest|cargo-test|deno-test)"' "$ALL" 2>/dev/null)"
    ;;
esac
case "$FAILURES" in
  '' | *[!0-9]*) FAILURES=0 ;;
esac

[ "$FAILURES" -eq 0 ] && exit 0
exit 1
