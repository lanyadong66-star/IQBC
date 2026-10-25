"""Composable image restoration followed by optional layout-aware OCR."""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import os
import tempfile

from .ocr import Yad_OCR


IMAGE_STAGES = (
    ('use_super_resolution', 'x4 超分'),
    ('use_dewarping', '透视矫正'),
    ('use_shadow_removal', '去阴影'),
    ('use_background_fading', '背景淡化'),
    ('use_text_enhancement', '文字增强'),
    ('use_background_removal', '去背景 / 二值化'),
)


@dataclass
class DocumentResult:
    image: object
    markdown: str | None
    stages: list[str]

    def save_markdown(self, output_path, *, overwrite=False):
        """Save the exact model output as UTF-8, without normalizing layout."""
        if self.markdown is None:
            raise ValueError('OCR was not enabled for this result.')
        target = Path(output_path).resolve()
        if target.suffix.lower() != '.md':
            raise ValueError('Markdown output_path must end with .md.')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise FileExistsError(f'Output already exists: {target}')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', newline='', dir=target.parent,
                    delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(self.markdown)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target


def process_document(image, *, use_ocr=False, ocr_model=None, ocr_max_tokens=8192,
                     ocr_max_pixels=2097152,
                     **image_options):
    """Return image + optional Markdown; disabled stages load no models.

    Uses the existing image-stage order. Releases restoration weights before
    loading Yad_OCR, so both models do not have to coexist on the GPU.
    """
    import numpy as np
    from .x4.infer_document_x4 import (
        SRX4Inferencer, _normalize_engine_config, _read_image, _resolve_path,
    )

    if not isinstance(use_ocr, bool):
        raise ValueError('use_ocr must be true or false.')
    options = _normalize_engine_config(image_options)
    if isinstance(image, (str, PathLike)):
        image = _read_image(_resolve_path(image, options['project_root'], 'input_path'))
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2):
        raise ValueError('Expected a BGR uint8 image of at least 2 x 2 pixels.')
    stages = [label for key, label in IMAGE_STAGES if options[key]]
    recognizer = (Yad_OCR(
        model=ocr_model, max_tokens=ocr_max_tokens, max_pixels=ocr_max_pixels,
        device='cuda' if options['num_gpu'] else 'cpu') if use_ocr else None)
    if stages:
        with SRX4Inferencer(**options) as engine:
            result_image = engine.infer_image(image)
    else:
        result_image = image.copy()
    markdown = None
    if use_ocr:
        with recognizer as engine:
            markdown = engine.infer_image(result_image)
        stages.append('OCR 文本提取')
    return DocumentResult(result_image, markdown, stages or ['未启用处理，返回原图'])
