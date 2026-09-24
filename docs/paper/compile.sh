#!/usr/bin/env bash
# Compile the IEEE Typst manuscript.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v typst >/dev/null 2>&1; then
  echo "typst is not on PATH. Install the CLI, e.g.:"
  echo "  curl -fL https://github.com/typst/typst/releases/download/v0.15.1/typst-x86_64-unknown-linux-musl.tar.xz | tar -xJ"
  echo "  install -m 755 typst-x86_64-unknown-linux-musl/typst ~/.local/bin/typst"
  exit 1
fi

cd "$ROOT"
typst compile \
  --font-path "$ROOT/fonts" \
  --root "$ROOT" \
  "$ROOT/main.typ" \
  "$ROOT/geometric-shepherding-rl.pdf"

echo "Wrote $ROOT/geometric-shepherding-rl.pdf"
