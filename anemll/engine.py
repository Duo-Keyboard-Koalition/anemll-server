"""CoreML inference engine for ANEMLL models on Apple Neural Engine."""

import re
import glob as glob_module
import warnings
from pathlib import Path

import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Suppress coremltools torch version warning (printed to stderr at import time)
warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")
_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    import coremltools as ct
finally:
    _captured = sys.stderr.getvalue()
    sys.stderr = _stderr
    # Re-print any real errors (not the torch version warning)
    for line in _captured.splitlines():
        if "has not been tested with coremltools" not in line and line.strip():
            print(line, file=sys.stderr)

from transformers import AutoTokenizer


class ANEEngine:
    """Loads and runs ANEMLL models on Apple Neural Engine."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).resolve()
        self._load_meta()
        self._load_models()
        self._load_tokenizer()
        self._init_state()

    def _load_meta(self):
        meta_path = self.model_path / "meta.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.yaml not found in {self.model_path}")

        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        params = meta["model_info"]["parameters"]
        self.context_length = int(params["context_length"])
        self.batch_size = int(params["batch_size"])
        self.num_chunks = int(params["num_chunks"])
        self.prefix = params.get("model_prefix", "llama")
        self.lut_ffn = params["lut_ffn"]
        self.lut_lmhead = params["lut_lmhead"]
        self.model_name = meta["model_info"]["name"]

    def _parse_model_path(self, path: Path) -> str:
        """Find model file with .mlmodelc or .mlpackage extension."""
        for candidate in [path, path.with_suffix(".mlmodelc"), path.with_suffix(".mlpackage")]:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"Model not found: {path}")

    def _load_coreml(self, path: str, function_name: str | None = None):
        """Load a CoreML model."""
        p = Path(path)
        compute_unit = ct.ComputeUnit.CPU_AND_NE
        if p.suffix == ".mlmodelc":
            if function_name:
                return ct.models.CompiledMLModel(path, compute_unit, function_name=function_name)
            return ct.models.CompiledMLModel(path, compute_unit)
        else:
            if function_name:
                return ct.models.MLModel(path, function_name=function_name)
            return ct.models.MLModel(path)

    def _load_models(self):
        d = self.model_path

        # Embeddings
        embed_path = self._parse_model_path(d / f"{self.prefix}_embeddings")
        self.embed_model = self._load_coreml(embed_path)

        # LM head
        lut_suffix = f"_lut{self.lut_lmhead}" if self.lut_lmhead != "none" else ""
        lmhead_path = self._parse_model_path(d / f"{self.prefix}_lm_head{lut_suffix}")
        self.lmhead_model = self._load_coreml(lmhead_path)

        # FFN chunks
        ffn_lut = f"_lut{self.lut_ffn}" if self.lut_ffn != "none" else ""
        ffn_base = d / f"{self.prefix}_FFN_PF{ffn_lut}_chunk_01of{self.num_chunks:02d}"
        ffn_path = self._parse_model_path(ffn_base)

        # Find all chunks
        pattern = re.sub(r"_chunk_\d+of\d+", "_chunk_*", str(ffn_path))
        chunk_paths = sorted(glob_module.glob(pattern))

        self.ffn_models = []
        for cp in chunk_paths:
            self.ffn_models.append({
                "infer": self._load_coreml(cp, function_name="infer"),
                "prefill": self._load_coreml(cp, function_name="prefill"),
            })

        # Read metadata from model
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> dict:
        """Extract metadata from loaded models."""
        meta = {}
        model = self.ffn_models[0] if self.ffn_models else self.embed_model
        if isinstance(model, dict):
            model = model.get("prefill", model.get("infer"))
        if hasattr(model, "user_defined_metadata"):
            m = model.user_defined_metadata
            meta["context_length"] = int(m.get("com.anemll.context_length", self.context_length))
            meta["state_length"] = int(m.get("com.anemll.state_length", meta["context_length"]))
            meta["batch_size"] = int(m.get("com.anemll.batch_size", self.batch_size))
            meta["lut_bits"] = int(m.get("com.anemll.lut_bits", 0))
            meta["num_chunks"] = int(m.get("com.anemll.num_chunks", self.num_chunks))
        else:
            meta["context_length"] = self.context_length
            meta["state_length"] = self.context_length
            meta["batch_size"] = self.batch_size
            meta["lut_bits"] = 4
            meta["num_chunks"] = self.num_chunks

        # Override from meta.yaml values
        meta["batch_size"] = self.batch_size
        meta["context_length"] = self.context_length
        meta["state_length"] = self.context_length
        return meta

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), use_fast=False, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def _init_state(self):
        if isinstance(self.ffn_models[0], dict):
            self.state = self.ffn_models[0]["prefill"].make_state()
        else:
            self.state = self.ffn_models[0].make_state()
        self.causal_mask = self._make_causal_mask(self.context_length, 0)
        self.causal_mask = torch.tensor(self.causal_mask, dtype=torch.float16)

    @staticmethod
    def _make_causal_mask(length: int, start: int):
        mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
        row_indices = np.arange(length).reshape(length, 1)
        col_indices = np.arange(length).reshape(1, length)
        mask[:, :, col_indices <= (row_indices + start)] = 0
        return mask

    def run_prefill(self, input_ids, current_pos: int):
        """Run prefill on the input sequence."""
        batch_pos = 0
        while batch_pos < current_pos:
            batch_end = min(batch_pos + self.batch_size, current_pos)
            current_batch_size = batch_end - batch_pos

            batch_input = input_ids[:, batch_pos:batch_end]
            batch_input = F.pad(batch_input, (0, self.batch_size - current_batch_size), value=0)

            position_ids = torch.arange(batch_pos, batch_pos + self.batch_size, dtype=torch.int32)
            batch_causal_mask = self.causal_mask[:, :, batch_pos:batch_pos + self.batch_size, :]

            hidden_states = torch.from_numpy(
                self.embed_model.predict({"input_ids": batch_input.numpy()})["hidden_states"]
            )

            for ffn_model in self.ffn_models:
                if isinstance(ffn_model, dict):
                    inputs = {
                        "hidden_states": hidden_states.numpy(),
                        "position_ids": position_ids.numpy(),
                        "causal_mask": batch_causal_mask.numpy(),
                        "current_pos": np.array([batch_pos], dtype=np.int32),
                    }
                    output = ffn_model["prefill"].predict(inputs, self.state)
                    hidden_states = torch.from_numpy(output["output_hidden_states"])

            batch_pos = batch_end

    def generate_next_token(self, input_ids, pos: int, temperature: float = 0.0) -> int:
        """Generate the next token."""
        current_token = input_ids[:, pos - 1 : pos]

        hidden_states = torch.from_numpy(
            self.embed_model.predict({"input_ids": current_token.numpy()})["hidden_states"]
        )

        update_mask = torch.zeros((1, 1, self.context_length, 1), dtype=torch.float16)
        update_mask[0, 0, pos - 1, 0] = 1.0
        position_ids = torch.tensor([pos - 1], dtype=torch.int32)
        single_causal_mask = self.causal_mask[:, :, pos - 1 : pos, :]

        for ffn_model in self.ffn_models:
            if isinstance(ffn_model, dict):
                inputs = {
                    "hidden_states": hidden_states.numpy(),
                    "update_mask": update_mask.numpy(),
                    "position_ids": position_ids.numpy(),
                    "causal_mask": single_causal_mask.numpy(),
                    "current_pos": position_ids.numpy(),
                }
                output = ffn_model["infer"].predict(inputs, self.state)
                hidden_states = torch.from_numpy(output["output_hidden_states"])

        lm_output = self.lmhead_model.predict({"hidden_states": hidden_states.numpy()})

        if "logits1" in lm_output:
            logits_parts = []
            for i in range(1, 9):
                key = f"logits{i}"
                if key in lm_output:
                    logits_parts.append(torch.from_numpy(lm_output[key]))
            logits = torch.cat(logits_parts, dim=-1)
        else:
            logits = torch.from_numpy(lm_output["output_logits"])

        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits[0, -1, :], dim=-1)
            return torch.multinomial(probs, num_samples=1).item()
        return torch.argmax(logits[0, -1, :]).item()

    def tokenize_chat(self, messages: list[dict]) -> torch.Tensor:
        """Apply chat template and return input_ids tensor."""
        template_output = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
        if hasattr(template_output, "input_ids"):
            return template_output.input_ids.to(torch.int32)
        return template_output.to(torch.int32)

    def generate(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 1000):
        """Generate tokens from a chat conversation. Yields token strings."""
        input_ids = self.tokenize_chat(messages)

        if input_ids.size(1) > self.context_length:
            raise ValueError(
                f"Input ({input_ids.size(1)} tokens) exceeds context length ({self.context_length})"
            )

        pos = input_ids.size(1)
        input_ids = F.pad(input_ids, (0, self.context_length - pos), value=0)

        self.run_prefill(input_ids, pos)

        tokens_generated = 0
        while tokens_generated < max_tokens:
            if pos >= self.context_length - 2:
                # Window shift
                max_batches = self.context_length // self.batch_size
                desired_batches = max(1, max_batches - 2)
                new_size = min(desired_batches * self.batch_size, self.context_length - self.batch_size)

                tmp = torch.zeros((1, self.context_length), dtype=torch.int32)
                tmp[:, :new_size] = input_ids[:, pos - new_size : pos]
                input_ids = tmp

                self.run_prefill(input_ids, new_size)
                pos = new_size

            next_token_id = self.generate_next_token(input_ids, pos, temperature)

            if next_token_id == self.tokenizer.eos_token_id:
                break

            input_ids[0, pos] = next_token_id
            yield self.tokenizer.decode([next_token_id])

            pos += 1
            tokens_generated += 1

    def warmup(self):
        """Run warmup inference to avoid GIL issues with CoreML."""
        messages = [{"role": "user", "content": "hi"}]
        for i, _ in enumerate(self.generate(messages, temperature=0.0, max_tokens=5)):
            if i >= 4:
                break
