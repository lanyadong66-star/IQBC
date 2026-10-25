"""SR-native, offline page-to-Markdown inference using Yad_OCR weights."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path

from .common.ocr_prompts import DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OCR_MODEL = PROJECT_ROOT / 'weights' / 'ocr' / 'yad_OCR'
OCR_RUNTIME = PROJECT_ROOT / '.runtime' / 'ocr'


def configure_ocr_runtime():
    """Keep runtime writes inside SR, including offline Hugging Face caches."""
    for key, directory in (
        ('HF_HOME', 'huggingface'), ('HF_HUB_CACHE', 'huggingface/hub'),
        ('HUGGINGFACE_HUB_CACHE', 'huggingface/hub'),
        ('HF_ASSETS_CACHE', 'huggingface/assets'),
        ('TRANSFORMERS_CACHE', 'huggingface/transformers'),
        ('HF_MODULES_CACHE', 'huggingface/modules'), ('TORCH_HOME', 'torch'),
        ('XDG_CACHE_HOME', 'cache'), ('CUDA_CACHE_PATH', 'cuda'),
        ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
        ('MPLCONFIGDIR', 'matplotlib'), ('TMPDIR', 'tmp'), ('TEMP', 'tmp'), ('TMP', 'tmp'),
    ):
        os.environ[key] = str(OCR_RUNTIME / directory)
    (OCR_RUNTIME / 'tmp').mkdir(parents=True, exist_ok=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def resolve_ocr_model(model=None):
    path = Path(model) if model is not None else DEFAULT_OCR_MODEL
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def ocr_status(model=None):
    """Check local inference assets without importing Torch or loading weights."""
    path = resolve_ocr_model(model)
    required = ('config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                'tokenizer.json', 'chat_template.json')
    missing = [name for name in required if not (path / name).is_file()]
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding='utf-8'))['weight_map']
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError('Empty weight map')
            missing.extend(name for name in set(weight_map.values())
                           if not (path / name).is_file())
        except (OSError, ValueError, KeyError, TypeError):
            missing.append('valid model.safetensors.index.json')
    elif not (path / 'model.safetensors').is_file():
        missing.append('model.safetensors')
    dependencies = [name for name in ('torch', 'transformers', 'accelerate', 'PIL', 'markdown')
                    if importlib.util.find_spec(name) is None]
    return {'ready': not missing and not dependencies, 'model': str(path),
            'missing_files': sorted(missing), 'missing_dependencies': dependencies,
            'source_ready': True}


class Yad_OCR:
    """Lazy in-memory OCR. No external source project or training framework.

    CUDA uses the original BF16 + Accelerate auto placement policy, with no
    custom VRAM cap. The actual placement is available through device_map.
    """

    def __init__(self, *, model=None, device='cuda', max_tokens=8192, max_pixels=2097152):
        if device not in ('cpu', 'cuda'):
            raise ValueError("OCR device must be 'cpu' or 'cuda'.")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
            raise ValueError('OCR max_tokens must be an integer from 1 to 32768.')
        if type(max_pixels) is not int or not 65536 <= max_pixels <= 4194304:
            raise ValueError('OCR max_pixels must be an integer from 65536 to 4194304.')
        self.model_path = resolve_ocr_model(model)
        self.device = device
        self.max_tokens = max_tokens
        self.max_pixels = max_pixels
        self._model = self._processor = self._torch = None
        self._closed = False

    @property
    def device_map(self):
        """Actual model placement, not an assumption based on checkpoint size."""
        return dict(getattr(self._model, 'hf_device_map', {}))

    def _load_model(self):
        if self._model is not None:
            return
        configure_ocr_runtime()
        status = ocr_status(self.model_path)
        if not status['ready']:
            details = ', '.join(status['missing_files'] + status['missing_dependencies'])
            raise RuntimeError('OCR 未就绪，请检查 weights/ocr/yad_OCR 和 requirements-ocr.txt：' + details)
        import torch

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('OCR CUDA 不可用，请选择 CPU，或检查 PyTorch CUDA 环境。')
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError('OCR 需要支持 Qwen3-VL 的 Transformers，请检查 requirements-ocr.txt。') from error
        self._torch = torch
        try:
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(self.model_path), device_map='auto' if self.device == 'cuda' else 'cpu',
                dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                local_files_only=True, offload_folder=str(OCR_RUNTIME / 'offload'))
            self._model.eval()
            self._processor = AutoProcessor.from_pretrained(
                str(self.model_path), local_files_only=True)
            size = dict(self._processor.image_processor.size)
            size['longest_edge'] = self.max_pixels
            size['shortest_edge'] = min(size['shortest_edge'], self.max_pixels)
            self._processor.image_processor.size = size
        except Exception:
            self.close()
            raise

    def _infer_page(self, page):
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': DEFAULT_SYSTEM_PROMPT}]},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': page}, {'type': 'text', 'text': DEFAULT_PROMPT}]},
        ]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False)
        tokens = generated[:, inputs['input_ids'].shape[1]:]
        eos = self._model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
        if tokens.shape[1] >= self.max_tokens and tokens[0, -1].item() not in eos_ids:
            raise RuntimeError('Generation truncated; increase max_tokens')
        text = self._processor.batch_decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if not text.strip():
            raise RuntimeError('Model returned empty content')
        return text

    def infer_image(self, image):
        if self._closed:
            raise RuntimeError('Yad_OCR is closed.')
        import numpy as np
        from PIL import Image

        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2):
            raise ValueError('OCR expects a BGR uint8 image of at least 2 x 2 pixels.')
        self._load_model()
        # Keep the whole page and convert OpenCV BGR to the processor's RGB.
        return self._infer_page(Image.fromarray(image[:, :, ::-1].copy()))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._model = self._processor = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def extract_markdown(image, **options):
    """Recognize an in-memory page and return unmodified Markdown."""
    with Yad_OCR(**options) as recognizer:
        return recognizer.infer_image(image)
