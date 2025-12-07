#!/usr/bin/env bash
set -euo pipefail

# Enable SSH according to runpod expectations
ssh-keygen -A >/dev/null 2>&1 || true
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  install -d -m 700 ~/.ssh
  touch ~/.ssh/authorized_keys
  chmod 600 ~/.ssh/authorized_keys
  grep -qxF "$PUBLIC_KEY" ~/.ssh/authorized_keys || echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
fi

# Start SSH daemon
service ssh start

echo "Starting llmagentsfromscratch/runpod container..."

# Set up persistent ollama storage
echo "[startup] Setting up ollama persistence..."
mkdir -p /workspace/ollama
rm -rf ~/.ollama
ln -s /workspace/ollama ~/.ollama

# Start ollama server
echo "[startup] Starting ollama server..."
ollama serve &
sleep 3

# Pull model if specified
if [[ -n "${OLLAMA_MODEL:-}" ]]; then
  echo "[startup] Pulling ollama model: $OLLAMA_MODEL"
  ollama pull "$OLLAMA_MODEL"
  echo "[startup] Ollama model ready!"
fi

# Clone and install llm-agents-from-scratch
cd /workspace
if [[ -d llm-agents-from-scratch ]]; then
  echo "[startup] Repo already exists, skipping clone."
else
  git clone https://github.com/nerdai/llm-agents-from-scratch.git
fi

cd llm-agents-from-scratch
uv pip install --system -e ".[notebook-utils]"
cd ~

echo "llm-agents-from-scratch installed!"

# Install jupyter lab
pip install ipywidgets jupyterlab

# Keep container running per Runpod docs
if [[ "${DEV:-}" == "1" ]]; then
  echo "[startup] DEV mode: opening shell..."
  cd /workspace
  exec /bin/zsh
else
  echo "[startup] Pod mode: sleeping forever (use SSH to connect)"
  if [[ -n "${JUPYTER_PASSWORD:-}" ]]; then
    cd /workspace
    jupyter lab --allow-root --no-browser --port=8888 --ip=* \
      --ServerApp.terminado_settings='{"shell_command":["/bin/zsh"]}' \
      --ServerApp.token="$JUPYTER_PASSWORD" --ServerApp.allow_origin=* \
      --ServerApp.preferred_dir=/workspace
  else
    sleep infinity
  fi
fi
