#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_ID_DEFAULT="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_ID="${MODEL_ID_DEFAULT}"

FLASH_ATTN="${FLASH_ATTN:-0}" # set to 1 to enable
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}" # set to 1 to auto-install
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

EXTRA_ARGS=()
EXPECT_VALUE="0"
while [[ "${#}" -gt 0 ]]; do
  if [[ "${EXPECT_VALUE}" == "1" ]]; then
    EXTRA_ARGS+=("${1}")
    EXPECT_VALUE="0"
    shift
    continue
  fi

  case "${1}" in
    --install-flash-attn)
      INSTALL_FLASH_ATTN="1"
      shift
      ;;
    --no-install-flash-attn)
      INSTALL_FLASH_ATTN="0"
      shift
      ;;
    --with-base)
      EXTRA_ARGS+=(--base-checkpoint "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  ./run_webui_uv.sh [MODEL_ID] [qwen-tts-demo args...]

Environment variables:
  IP=0.0.0.0 PORT=8000 CONCURRENCY=1
  DEVICE=cuda:0 DTYPE=bfloat16
  FLASH_ATTN=0|1
  INSTALL_FLASH_ATTN=0|1 (or pass --install-flash-attn)
  FLASH_ATTN_VERSION=2.8.3

Examples:
  ./run_webui_uv.sh
  FLASH_ATTN=1 INSTALL_FLASH_ATTN=1 ./run_webui_uv.sh
  ./run_webui_uv.sh --with-base
  ./run_webui_uv.sh Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --max-new-tokens 512
EOF
      exit 0
      ;;
    -c|--checkpoint|--base-checkpoint|--device|--dtype|--ip|--port|--concurrency|--ssl-certfile|--ssl-keyfile|--max-new-tokens|--temperature|--top-k|--top-p|--repetition-penalty|--subtalker-top-k|--subtalker-top-p|--subtalker-temperature)
      EXTRA_ARGS+=("${1}")
      EXPECT_VALUE="1"
      shift
      ;;
    *)
      if [[ "${1}" != -* && "${MODEL_ID}" == "${MODEL_ID_DEFAULT}" ]]; then
        MODEL_ID="${1}"
      else
        EXTRA_ARGS+=("${1}")
      fi
      shift
      ;;
  esac
done

IP="${IP:-0.0.0.0}"
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-1}"

# GPU defaults (override via env if needed)
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  FLASH_ATTN="1"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found in PATH."
  echo "Install uv: https://docs.astral.sh/uv/"
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  uv venv --python 3.12
fi

echo "[1/3] Installing PyTorch (CUDA if available)..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  # Use a torch version with readily available flash-attn wheels (optional) to keep the environment stable.
  uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.*" "torchaudio==2.9.*"
else
  echo "No NVIDIA GPU detected, falling back to CPU mode."
  DEVICE="cpu"
  DTYPE="float32"
  uv pip install torch torchaudio
fi

if [[ "${FLASH_ATTN}" == "1" && "${DEVICE}" == cpu* ]]; then
  echo "Note: FLASH_ATTN=1 requested, but running on CPU. Disabling FlashAttention."
  FLASH_ATTN="0"
  INSTALL_FLASH_ATTN="0"
fi

echo "[2/3] Installing project dependencies..."
uv pip install -e .

if [[ "${FLASH_ATTN}" == "1" ]]; then
  TORCH_MM="$(
    .venv/bin/python - <<'PY'
import torch
torch_version = torch.__version__.split('+', 1)[0]
print('.'.join(torch_version.split('.')[:2]))
PY
  )"
  if [[ "${TORCH_MM}" != "2.9" ]]; then
    echo "Error: FLASH_ATTN=1 expects torch 2.9.* (got torch ${TORCH_MM}.x)."
    echo "Tip: remove .venv and re-run, or disable FLASH_ATTN."
    exit 1
  fi
fi

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "[2.5/3] Installing flash-attn (may take a while the first time)..."
  if .venv/bin/python -c "import flash_attn" >/dev/null 2>&1; then
    echo "flash-attn already installed."
  else
    read -r TORCH_MM PY_TAG CXX11ABI ARCH_TAG <<<"$(
      .venv/bin/python - <<'PY'
import platform
import sys
import torch

torch_version = torch.__version__.split('+', 1)[0]
torch_mm = '.'.join(torch_version.split('.')[:2])
py_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
cxx11 = 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'
arch = platform.machine().lower()
arch_tag = 'linux_x86_64' if arch in ('x86_64', 'amd64') else 'linux_aarch64'
print(torch_mm, py_tag, cxx11, arch_tag)
PY
    )"

    if [[ "${TORCH_MM}" != "2.9" ]]; then
      echo "Error: flash-attn auto-install expects torch 2.9.* (got torch ${TORCH_MM}.x)."
      echo "Tip: remove .venv and re-run, or force install torch 2.9.* first."
      exit 1
    fi

    WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu12torch${TORCH_MM}cxx11abi${CXX11ABI}-${PY_TAG}-${PY_TAG}-${ARCH_TAG}.whl"
    URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${WHEEL}"

    uv pip install --no-deps "${URL}"
    .venv/bin/python -c "import flash_attn" >/dev/null
    echo "flash-attn installed."
  fi
fi

echo "[3/3] Launching Web UI..."
echo "Model: ${MODEL_ID}"
echo "URL:   http://${IP}:${PORT}"

args=(
  "${MODEL_ID}"
  --ip "${IP}"
  --port "${PORT}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --concurrency "${CONCURRENCY}"
)

if [[ "${FLASH_ATTN}" == "1" ]]; then
  args+=(--flash-attn)
else
  args+=(--no-flash-attn)
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  args+=("${EXTRA_ARGS[@]}")
fi

exec uv run --no-sync qwen-tts-demo "${args[@]}"
