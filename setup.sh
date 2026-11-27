#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="$ROOT/.coreml.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with: brew install uv" >&2
  exit 1
fi

uv python install "$PYTHON_VERSION"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi
uv pip install --python "$VENV/bin/python" -r "$ROOT/requirements-coreml.txt"

mkdir -p "$ROOT/models"
if ! find "$ROOT/models/hub/models--IDKiro--sdxs-512-0.9/snapshots" -path '*/text_encoder/model.safetensors' -type f -print -quit 2>/dev/null | grep -q .; then
  HF_HOME="$ROOT/models" "$VENV/bin/hf" download IDKiro/sdxs-512-0.9
fi

if [[ ! -f "$ROOT/coreml_models/unet_sdxs_512.mlpackage/Manifest.json" \
   || ! -f "$ROOT/coreml_models/taesd_encoder_512.mlpackage/Manifest.json" \
   || ! -f "$ROOT/coreml_models/taesd_decoder.mlpackage/Manifest.json" ]]; then
  cat >&2 <<'EOF'
Python environment and SDXS text model installed.
The converted CoreML packages are not distributed in this repository.
Copy the three complete .mlpackage directories into coreml_models/ before starting TouchDesigner.
See the README's “Model files and licenses” section.
EOF
  exit 2
fi

printf 'Setup complete. Runtime Python: %s\n' "$VENV/bin/python"
