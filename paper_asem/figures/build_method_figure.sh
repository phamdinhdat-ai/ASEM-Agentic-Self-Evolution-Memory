#!/usr/bin/env bash
# Build asem_method_figure.pdf with Comic Sans MS.
# Tectonic cannot open absolute Windows font paths, so the .tex and the system
# font files are copied into a temporary folder, compiled there, and only the PDF
# is copied back. The Microsoft font files are never written into the repository.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tectonic="${TECTONIC:-/c/Users/Dat Pham/anaconda3/envs/tectonic/Library/bin/tectonic.exe}"
fonts="${FONT_DIR:-/c/Windows/Fonts}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp "$here/asem_method_figure.tex" "$work/"
for f in comic.ttf comicbd.ttf comici.ttf comicz.ttf; do cp "$fonts/$f" "$work/"; done
(cd "$work" && "$tectonic" asem_method_figure.tex)
cp "$work/asem_method_figure.pdf" "$here/"
echo "wrote $here/asem_method_figure.pdf"
