"""Model management: pull, list, remove models from HuggingFace."""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import yaml

MODELS_DIR = Path.home() / ".anemll" / "models"


def get_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def list_models() -> list[dict]:
    """List all downloaded models with their metadata."""
    models_dir = get_models_dir()
    models = []
    for entry in sorted(models_dir.iterdir()):
        meta_path = entry / "meta.yaml"
        if entry.is_dir() and meta_path.exists():
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            params = meta.get("model_info", {}).get("parameters", {})
            models.append({
                "name": entry.name,
                "path": str(entry),
                "context_length": params.get("context_length"),
                "batch_size": params.get("batch_size"),
                "num_chunks": params.get("num_chunks"),
            })
    return models


def resolve_model(name: str) -> Path:
    """Resolve a model name to its directory path.

    Accepts:
      - Full HF repo id: anemll/anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2
      - Just the model name: anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2
      - Absolute path to a directory
    """
    # Absolute path
    p = Path(name)
    if p.is_absolute() and p.exists() and (p / "meta.yaml").exists():
        return p

    # Strip org prefix if present
    model_name = name.split("/")[-1]

    model_path = get_models_dir() / model_name
    if model_path.exists() and (model_path / "meta.yaml").exists():
        return model_path

    raise FileNotFoundError(
        f"Model '{name}' not found. Run: anemll pull {name}"
    )


def pull_model(repo_id: str) -> Path:
    """Pull a model from HuggingFace and prepare it for use.

    Args:
        repo_id: HuggingFace repo id, e.g. 'anemll/anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2'
                 or just the model name (will prepend 'anemll/' if no org given).
    """
    if "/" not in repo_id:
        repo_id = f"anemll/{repo_id}"

    model_name = repo_id.split("/")[-1]
    model_path = get_models_dir() / model_name

    if model_path.exists() and (model_path / "meta.yaml").exists():
        # Check if mlmodelc directories exist (not just zips)
        has_compiled = any(model_path.glob("*.mlmodelc"))
        if has_compiled:
            print(f"Model '{model_name}' already exists at {model_path}")
            return model_path
        print(f"Model '{model_name}' exists but needs unpacking...")
    else:
        hf_url = f"https://huggingface.co/{repo_id}"
        print(f"Pulling {repo_id} from {hf_url}")

        # Check git-lfs is available
        if shutil.which("git-lfs") is None:
            print("Error: git-lfs is required. Install with: brew install git-lfs")
            raise SystemExit(1)

        # Ensure LFS is initialized
        subprocess.run(["git", "lfs", "install"], capture_output=True)

        # Clone the repo
        print(f"Cloning into {model_path}...")
        result = subprocess.run(
            ["git", "clone", hf_url, str(model_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Error cloning: {result.stderr}")
            raise SystemExit(1)

        # Pull LFS files
        print("Downloading model files (this may take a while)...")
        result = subprocess.run(
            ["git", "lfs", "pull"],
            cwd=str(model_path),
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Error pulling LFS files: {result.stderr}")
            raise SystemExit(1)

    # Unzip model files
    zip_files = list(model_path.glob("*.zip"))
    if zip_files:
        print("Unpacking model files...")
        for zf in zip_files:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(model_path)
        print(f"Unpacked {len(zip_files)} model archives")

    # Verify
    if not (model_path / "meta.yaml").exists():
        print(f"Error: meta.yaml not found in {model_path}")
        raise SystemExit(1)

    print(f"Model ready: {model_name}")
    return model_path


def remove_model(name: str) -> None:
    """Remove a downloaded model."""
    model_name = name.split("/")[-1]
    model_path = get_models_dir() / model_name

    if not model_path.exists():
        print(f"Model '{model_name}' not found")
        raise SystemExit(1)

    shutil.rmtree(model_path)
    print(f"Removed model '{model_name}'")
