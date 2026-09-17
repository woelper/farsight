#!/usr/bin/env bash
# Download the slow-tier model and write the daemon config so sentence
# prediction works out of the box.
#
# Usage: ./scripts/setup-model.sh
#
# - Model: Qwen2.5-1.5B base Q4_K_M (~941 MB) into
#   $XDG_DATA_HOME/predict/models (or ~/.local/share/predict/models).
# - Config: ~/.config/predict/predictd.toml (XDG_CONFIG_HOME respected).
#   An existing config is backed up, never overwritten blindly.
set -euo pipefail

MODEL_URL="https://huggingface.co/QuantFactory/Qwen2.5-1.5B-GGUF/resolve/main/Qwen2.5-1.5B.Q4_K_M.gguf"
MODEL_FILE="qwen2.5-1.5b-q4_k_m.gguf"

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/predict/models"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/predict"
CONFIG_FILE="$CONFIG_HOME/predictd.toml"

mkdir -p "$DATA_HOME" "$CONFIG_HOME"

if [[ -f "$DATA_HOME/$MODEL_FILE" ]]; then
    echo "model already present: $DATA_HOME/$MODEL_FILE"
else
    echo "downloading model (~941 MB) to $DATA_HOME/$MODEL_FILE ..."
    curl -sS -L -o "$DATA_HOME/$MODEL_FILE" "$MODEL_URL"
    echo "download complete."
fi

if [[ -f "$CONFIG_FILE" ]]; then
    echo "backing up existing config to $CONFIG_FILE.bak"
    cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
fi

cat > "$CONFIG_FILE" <<EOF
# predictd configuration (written by scripts/setup-model.sh).
[llm]
enabled = true
model_path = "$DATA_HOME/$MODEL_FILE"
max_tokens = 32
confidence_threshold = -1.5
EOF

echo "wrote $CONFIG_FILE"
echo "restart predictd (./scripts/start.sh --stop; ./scripts/start.sh) to pick it up."
