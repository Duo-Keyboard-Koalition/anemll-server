# anemll

Run LLMs on Apple Neural Engine with an ollama-like CLI and OpenAI-compatible API.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12 (`coremltools` does not support 3.13+)
- Git LFS (`brew install git-lfs && git lfs install`)

## Install

```bash
# Clone and install
git clone git@github.com:Duo-Keyboard-Koalition/anemll-server.git
cd anemll-server
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

## Quick Start

```bash
# Download a model
anemll pull anemll/anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2

# Interactive chat
anemll run anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2

# Start OpenAI-compatible server
anemll serve anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2
```

## Commands

| Command | Description |
|---------|-------------|
| `anemll pull <model>` | Download a model from HuggingFace |
| `anemll list` | List downloaded models |
| `anemll run <model>` | Interactive chat |
| `anemll run <model> -p "prompt"` | Single prompt |
| `anemll serve <model>` | Start API server (default: port 8000) |
| `anemll rm <model>` | Remove a downloaded model |

## API

The server exposes OpenAI-compatible endpoints:

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anemll-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# List models
curl http://localhost:8000/v1/models
```

Use `http://localhost:8000/v1` as the base URL in Open WebUI or any OpenAI-compatible client.

## Available Models

Browse models at [huggingface.co/anemll](https://huggingface.co/anemll):

```bash
anemll pull anemll/anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2
anemll pull anemll/anemll-google-gemma-3-4b-it-qat-int4-unquantized-ctx4096_0.3.5
```

## Known Issues

- Occasional GIL issue on first inference after startup (the warmup phase mitigates this).
- `coremltools` may warn about untested Torch versions. Safe to ignore.

## License

MIT
