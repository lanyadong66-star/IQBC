"""Run inference by loading Qwen3-VL weights directly with Transformers."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

from .common import IMAGE_TYPES, add_prompt_arguments, build_messages, write_result
from .prompts import DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT


SUPPORTED_DTYPES = {"auto", "bfloat16", "float16", "float32"}


@dataclass(frozen=True)
class LocalInferenceConfig:
    """Validated settings shared by file-based and command-line inference."""

    model: str
    image: Path | None = None
    image_dir: Path | None = None
    output: Path | None = None
    output_dir: Path | None = None
    overwrite: bool = False
    device_map: str = "auto"
    dtype: str = "auto"
    max_tokens: int = 8192
    prompt: str = DEFAULT_PROMPT
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


def _nonempty_string(data, key):
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field '{key}' must be a non-empty string")
    return value.strip()


def _resolve_path(value, base_dir, key, *, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field '{key}' must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_model(value, base_dir):
    """Resolve local model paths while preserving Hugging Face model IDs."""
    value = _nonempty_string({"model": value}, "model")
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else base_dir / path
    looks_local = (
        path.is_absolute()
        or value.startswith((".", "models/", "models\\", "checkpoints/", "checkpoints\\"))
        or candidate.exists()
    )
    return str(candidate.resolve()) if looks_local else value


def load_config(path):
    """Load and validate a JSON inference configuration."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Inference config must contain one JSON object")

    allowed = {
        "model", "image", "image_dir", "output", "output_dir", "overwrite",
        "device_map", "dtype", "max_tokens", "prompt", "system_prompt",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown config field(s): {', '.join(unknown)}")

    base_dir = config_path.parent
    model = _resolve_model(data.get("model"), base_dir)
    image = _resolve_path(data.get("image"), base_dir, "image", optional=True)
    image_dir = _resolve_path(data.get("image_dir"), base_dir, "image_dir", optional=True)
    if (image is None) == (image_dir is None):
        raise ValueError("Set exactly one of 'image' or 'image_dir'")
    output = _resolve_path(data.get("output"), base_dir, "output", optional=True)
    output_dir = _resolve_path(data.get("output_dir"), base_dir, "output_dir", optional=True)
    if image_dir is not None and output_dir is None:
        output_dir = (base_dir / "outputs").resolve()
    if image_dir is not None and output is not None:
        raise ValueError("'output' is only valid with a single 'image'; use 'output_dir' for a directory")
    if image is not None and output_dir is not None:
        raise ValueError("'output_dir' is only valid with 'image_dir'; use 'output' for a single image")

    overwrite = data.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ValueError("Config field 'overwrite' must be true or false")

    device_map = data.get("device_map", "auto")
    if not isinstance(device_map, str) or not device_map.strip():
        raise ValueError("Config field 'device_map' must be a non-empty string")

    dtype = data.get("dtype", "auto")
    if dtype not in SUPPORTED_DTYPES:
        choices = ", ".join(sorted(SUPPORTED_DTYPES))
        raise ValueError(f"Config field 'dtype' must be one of: {choices}")

    max_tokens = data.get("max_tokens", 8192)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("Config field 'max_tokens' must be a positive integer")

    prompt = data.get("prompt")
    if prompt is None:
        prompt = DEFAULT_PROMPT
    elif not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Config field 'prompt' must be null or a non-empty string")

    system_prompt = data.get("system_prompt")
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    elif not isinstance(system_prompt, str):
        raise ValueError("Config field 'system_prompt' must be null or a string")

    return LocalInferenceConfig(
        model=model,
        image=image,
        image_dir=image_dir,
        output=output,
        output_dir=output_dir,
        overwrite=overwrite,
        device_map=device_map.strip(),
        dtype=dtype,
        max_tokens=max_tokens,
        prompt=prompt,
        system_prompt=system_prompt,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="WeVisDoc Transformers inference")
    parser.add_argument("--model", default=os.getenv("WEVISDOC_MODEL_PATH"), help="Hugging Face model ID or checkpoint directory")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="One page image")
    inputs.add_argument("--image-dir", type=Path, help="Process all supported images in a directory (non-recursive)")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=sorted(SUPPORTED_DTYPES), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for batch Markdown results")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty batch results")
    add_prompt_arguments(parser)
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("Provide --model or WEVISDOC_MODEL_PATH")
    if args.image_dir is not None and args.output is not None:
        parser.error("--output is only valid with --image; use --output-dir with --image-dir")
    return LocalInferenceConfig(
        model=args.model,
        image=args.image,
        image_dir=args.image_dir,
        output=args.output,
        output_dir=args.output_dir if args.image_dir else None,
        overwrite=args.overwrite,
        device_map=args.device_map,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
    )


def _torch_dtype(torch, name):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def _list_images(image_dir):
    """Return supported images directly inside a directory, in stable order."""
    if not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_TYPES
    )
    if not paths:
        raise ValueError(f"No supported images in {image_dir}")
    return paths


def _result_path(image, output_dir):
    """Keep the image filename and append .md (for example page.png.md)."""
    return output_dir / f"{image.name}.md"


class LocalRecognizer:
    """Reusable local OCR for either a page path or an in-memory PIL image."""

    def __init__(self, model, *, device_map="auto", dtype="auto", max_tokens=8192,
                 prompt=DEFAULT_PROMPT, system_prompt=DEFAULT_SYSTEM_PROMPT,
                 local_files_only=False, max_pixels=None, max_memory=None,
                 offload_folder=None):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._torch = torch
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.model = None
        self.processor = None
        try:
            placement = {}
            if max_memory is not None:
                placement['max_memory'] = max_memory
            if offload_folder is not None:
                placement['offload_folder'] = str(offload_folder)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model, dtype=_torch_dtype(torch, dtype), device_map=device_map,
                local_files_only=local_files_only, **placement,
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(
                model, local_files_only=local_files_only)
            if max_pixels is not None:
                size = dict(self.processor.image_processor.size)
                size['longest_edge'] = max_pixels
                size['shortest_edge'] = min(size['shortest_edge'], max_pixels)
                self.processor.image_processor.size = size
        except Exception:
            self.close()
            raise

    def infer(self, image):
        if self.model is None:
            raise RuntimeError("LocalRecognizer is closed")
        if isinstance(image, (str, os.PathLike)):
            messages = build_messages(image, self.prompt, self.system_prompt, backend="local")
        else:
            from PIL import Image

            if not isinstance(image, Image.Image):
                raise ValueError("Expected an image path or PIL image")
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": [
                    {"type": "text", "text": self.system_prompt}]})
            messages.append({"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt}]})
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False)
        tokens = generated[:, inputs["input_ids"].shape[1]:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
        if tokens.shape[1] >= self.max_tokens and tokens[0, -1].item() not in eos_ids:
            raise RuntimeError("Generation truncated; increase max_tokens")
        text = self.processor.batch_decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if not text.strip():
            raise RuntimeError("Model returned empty content")
        return text

    def close(self):
        import gc

        self.model = None
        self.processor = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run(config):
    """Run one local request or a sequential directory batch."""
    try:
        # Delay imports so client-only users do not need GPU dependencies.
        print(f"Loading model: {config.model}", file=sys.stderr)
        recognizer = LocalRecognizer(
            config.model, device_map=config.device_map, dtype=config.dtype,
            max_tokens=config.max_tokens, prompt=config.prompt,
            system_prompt=config.system_prompt)
        infer_one = recognizer.infer

        if config.image is not None:
            text = infer_one(config.image)
            if config.output:
                write_result(config.output, text)
                print(f"Saved result: {config.output}", file=sys.stderr)
            print(text)
            return 0

        image_paths = _list_images(config.image_dir)
        output_dir = config.output_dir or Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = skipped = failed = 0
        for image in image_paths:
            output = _result_path(image, output_dir)
            if not config.overwrite and output.is_file() and output.stat().st_size:
                skipped += 1
                print(f"Skipped existing: {output}", file=sys.stderr)
                continue
            try:
                write_result(output, infer_one(image))
                completed += 1
                print(f"[{completed}/{len(image_paths)}] Saved: {output}", file=sys.stderr)
            except Exception as exc:
                failed += 1
                print(f"Failed: {image}: {exc}", file=sys.stderr)
        print(f"Completed: {completed}; skipped: {skipped}; failed: {failed}", file=sys.stderr)
        return 1 if failed else 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if 'recognizer' in locals():
            recognizer.close()


def main(argv=None):
    return run(parse_args(argv))


def main_from_config(path):
    """Entry point for IDEs and scripts that do not use command-line options."""
    try:
        config = load_config(path)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
