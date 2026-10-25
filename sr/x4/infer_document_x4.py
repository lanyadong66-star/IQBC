"""Reusable inference-only SR x4 + restoration document pipeline."""

from __future__ import annotations

import gc
import os
import sys
import threading
from collections.abc import Mapping
from os import PathLike, path as osp

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir, osp.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.dont_write_bytecode = True

INFERENCE_WEIGHT = 'weights/SR_x4.pth'
RESTORATION_WEIGHT = 'weights/restoration.pkl'
MASK_WEIGHT = 'weights/document_mask.pkl'
_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
_CONFIG_KEYS = {
    'weight', 'num_gpu', 'tile_size', 'tile_pad', 'progress', 'project_root',
    'use_background_removal', 'use_dewarping', 'use_shadow_removal', 'use_super_resolution',
    'use_text_enhancement', 'text_strength', 'use_background_fading', 'background_strength',
    'restoration_weight', 'mask_weight', 'runtime_platform',
}


def _resolve_path(value, project_root, name):
    value = osp.expanduser(os.fspath(value).strip())
    if not value:
        raise ValueError(f'{name} cannot be empty.')
    if sys.platform.startswith('linux') and (
            (len(value) > 2 and value[1] == ':' and value[2] in ('/', chr(92)))
            or value.startswith(chr(92) * 2)):
        raise ValueError(f'Windows path is not valid on Linux: {value}')
    return osp.abspath(value if osp.isabs(value) else osp.join(project_root, value))


def _same_path(first, second):
    return osp.normcase(osp.realpath(first)) == osp.normcase(osp.realpath(second))


def _normalize_tiling(tile_size, tile_pad):
    tile_size, tile_pad = int(tile_size), int(tile_pad)
    if tile_size < 1 or tile_pad < 0 or tile_size % 16 or tile_pad % 16:
        raise ValueError('tile_size must be positive, tile_pad nonnegative, and both divisible by 16.')
    return tile_size, tile_pad


def _normalize_engine_config(config):
    if not isinstance(config, Mapping):
        raise TypeError('Inference config must be a dictionary.')
    unknown = sorted(set(config) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f'Unsupported inference config keys: {unknown}')
    normalized = dict(config)
    project_root = _resolve_path(
        normalized.get('project_root', PROJECT_ROOT), PROJECT_ROOT, 'project_root')
    if not osp.isdir(project_root):
        raise FileNotFoundError(f'project_root does not exist: {project_root}')
    normalized['project_root'] = project_root
    normalized['tile_size'], normalized['tile_pad'] = _normalize_tiling(
        normalized.get('tile_size', 256), normalized.get('tile_pad', 32))
    normalized['num_gpu'] = int(normalized.get('num_gpu', 1))
    if normalized['num_gpu'] < 0:
        raise ValueError('num_gpu cannot be negative.')
    normalized['runtime_platform'] = str(normalized.get('runtime_platform', 'auto')).lower()
    if normalized['runtime_platform'] not in ('auto', 'windows', 'linux'):
        raise ValueError("runtime_platform must be 'auto', 'windows', or 'linux'.")
    for key, default in (
            ('progress', True), ('use_background_removal', False),
            ('use_dewarping', False), ('use_super_resolution', True),
            ('use_text_enhancement', False), ('use_background_fading', False),
            ('use_shadow_removal', False)):
        normalized[key] = normalized.get(key, default)
        if not isinstance(normalized[key], bool):
            raise ValueError(f'{key} must be true or false.')
    for key in ('text_strength', 'background_strength'):
        strength = normalized.get(key, 0.5)
        if (isinstance(strength, bool) or not isinstance(strength, (int, float))
                or not 0 <= strength <= 1):
            raise ValueError(f'{key} must be a number between 0 and 1.')
        normalized[key] = strength
    required_weights = {}
    if normalized['use_super_resolution']:
        required_weights['weight'] = normalized.get('weight', INFERENCE_WEIGHT)
    if any(normalized[key] for key in (
            'use_background_removal', 'use_dewarping', 'use_shadow_removal',
            'use_text_enhancement', 'use_background_fading')):
        required_weights['restoration_weight'] = normalized.get('restoration_weight', RESTORATION_WEIGHT)
    if normalized['use_dewarping']:
        required_weights['mask_weight'] = normalized.get('mask_weight', MASK_WEIGHT)
    for name, value in required_weights.items():
        normalized[name] = _resolve_path(value, project_root, name)
        if not osp.isfile(normalized[name]):
            raise FileNotFoundError(f'{name} checkpoint does not exist: {normalized[name]}')
    return normalized


def _runtime_environment(project_root):
    runtime_root = osp.join(project_root, '.runtime', 'x4')
    for name, folder in (
            ('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch'),
            ('CUDA_CACHE_PATH', 'cuda-cache'), ('MPLCONFIGDIR', 'matplotlib'),
            ('TMPDIR', 'tmp')):
        location = osp.join(runtime_root, folder)
        os.environ[name] = location
    # Only the temporary directory must exist before importing the runtime.
    # Libraries create their own cache directories when they actually need them.
    os.makedirs(os.environ['TMPDIR'], exist_ok=True)
    if os.name == 'nt':
        os.environ['TEMP'] = os.environ['TMPDIR']
        os.environ['TMP'] = os.environ['TMPDIR']


def _build_model_options(config):
    options = {
        'weight': config.get('weight'),
        'num_gpu': config['num_gpu'],
        'use_super_resolution': config['use_super_resolution'],
        'scale': 4,
        'network_g': {'window_size': 16},
        'tile': {'tile_size': config['tile_size'], 'tile_pad': config['tile_pad']},
    }
    if config['use_background_removal']:
        options['background_removal'] = {
            'enabled': True, 'weight': config['restoration_weight']}
    if config['use_dewarping']:
        options['dewarping'] = {
            'enabled': True, 'weight': config['restoration_weight'],
            'mask_weight': config['mask_weight']}
    if config['use_shadow_removal']:
        options['shadow_removal'] = {
            'enabled': True, 'weight': config['restoration_weight']}
    if config['use_text_enhancement']:
        options['text_enhancement'] = {
            'enabled': True, 'weight': config['restoration_weight'],
            'strength': config['text_strength']}
    if config['use_background_fading']:
        options['background_fading'] = {
            'enabled': True, 'weight': config['restoration_weight'],
            'strength': config['background_strength']}
    return options


def _read_image(input_path):
    import cv2
    import numpy as np

    if not osp.isfile(input_path):
        raise FileNotFoundError(f'Input image does not exist: {input_path}')
    image = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode input image: {input_path}')
    return image


def _write_image(output_path, image, overwrite):
    import cv2

    success, encoded = cv2.imencode(osp.splitext(output_path)[1], image)
    if not success:
        raise OSError(f'Cannot encode output image: {output_path}')
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb' if overwrite else 'xb') as output_file:
        output_file.write(encoded.tobytes())


def _directory_images(input_dir):
    with os.scandir(input_dir) as entries:
        paths = sorted(
            entry.path for entry in entries
            if entry.is_file() and not entry.name.startswith('.')
            and osp.splitext(entry.name)[1].lower() in _IMAGE_EXTENSIONS)
    stems = set()
    for path in paths:
        stem = osp.splitext(osp.basename(path))[0].casefold()
        if stem in stems:
            raise ValueError(f'Inference images would use the same output stem: {stem}')
        stems.add(stem)
    if not paths:
        raise FileNotFoundError(f'No supported inference images were found in: {input_dir}')
    return paths


class SRX4Inferencer:
    """BGR uint8 inference: optional SR -> dewarp -> fade -> text -> cleanup.

    Array/file inference returns an array without saving by default.
    Use a context manager or close() to release all loaded networks.
    """

    SCALE = 4
    NAMESPACE = 'x4'

    def __init__(self, weight=INFERENCE_WEIGHT, num_gpu=1, tile_size=256, tile_pad=32,
                 progress=True, project_root=PROJECT_ROOT,
                 use_background_removal=False, use_dewarping=False,
                 restoration_weight=RESTORATION_WEIGHT, mask_weight=MASK_WEIGHT,
                 runtime_platform='auto', *, use_super_resolution=True,
                 use_text_enhancement=False, text_strength=0.5,
                 use_background_fading=False, background_strength=0.5,
                 use_shadow_removal=False):
        self._config = _normalize_engine_config({
            'weight': weight, 'num_gpu': num_gpu, 'tile_size': tile_size,
            'tile_pad': tile_pad, 'progress': progress, 'project_root': project_root,
            'use_background_removal': use_background_removal,
            'use_dewarping': use_dewarping,
            'use_shadow_removal': use_shadow_removal,
            'use_super_resolution': use_super_resolution,
            'use_text_enhancement': use_text_enhancement,
            'text_strength': text_strength,
            'use_background_fading': use_background_fading,
            'background_strength': background_strength,
            'restoration_weight': restoration_weight, 'mask_weight': mask_weight,
            'runtime_platform': runtime_platform,
        })
        self._closed = False
        self._inference_lock = threading.Lock()
        self._model = None
        _runtime_environment(self._config['project_root'])

        import torch
        from sr.models.sr_model import RestorationPipeline

        self._torch = torch
        expected = self._config['runtime_platform']
        actual = 'windows' if sys.platform == 'win32' else sys.platform
        if expected != 'auto' and expected != actual:
            raise RuntimeError(f'runtime_platform={expected!r}, but this machine is {actual!r}.')
        count = self._config['num_gpu']
        if count and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use num_gpu=0 for CPU inference.')
        if count > torch.cuda.device_count():
            raise RuntimeError(f'num_gpu={count}, but only {torch.cuda.device_count()} GPUs are visible.')
        torch.backends.cudnn.benchmark = True
        try:
            self._model = RestorationPipeline(_build_model_options(self._config))
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    @property
    def weight(self):
        return self._config.get('weight') if self._config['use_super_resolution'] else None

    @property
    def scale(self):
        return self.SCALE if self._config['use_super_resolution'] else 1

    @property
    def closed(self):
        return self._closed

    def _require_open(self):
        if self._closed or self._model is None:
            raise RuntimeError('SRX4Inferencer is closed.')

    def set_tiling(self, tile_size, tile_pad=32):
        tile_size, tile_pad = _normalize_tiling(tile_size, tile_pad)
        with self._inference_lock:
            self._require_open()
            self._config.update(tile_size=tile_size, tile_pad=tile_pad)
            self._model.opt['tile'] = {'tile_size': tile_size, 'tile_pad': tile_pad}
            self._model.clear_attention_mask_cache()

    def set_progress(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('enabled must be true or false.')
        with self._inference_lock:
            self._require_open()
            self._config['progress'] = enabled

    def infer_image(self, image, progress_callback=None):
        """Run only enabled stages and return a BGR uint8 image without saving."""
        with self._inference_lock:
            self._require_open()
            return self._model.infer_array(image, progress_callback)

    def infer_file(self, input_path, output_path=None, *, overwrite=False):
        """Return an array; save only when output_path is explicitly supplied."""
        root = self._config['project_root']
        input_path = _resolve_path(input_path, root, 'input_path')
        if output_path is not None:
            output_path = _resolve_path(output_path, root, 'output_path')
            if _same_path(input_path, output_path):
                raise ValueError('output_path must differ from input_path.')
            if osp.exists(output_path) and not overwrite:
                raise FileExistsError(f'Output already exists: {output_path}')
        result = self.infer_image(_read_image(input_path))
        if output_path is not None:
            _write_image(output_path, result, overwrite)
        return result

    def infer_directory(self, input_dir, output_dir):
        """Save final images using original filenames, without dataset/training code."""
        from tqdm import tqdm
        from sr.common.tile_geometry import count_tile_regions

        root = self._config['project_root']
        input_dir = _resolve_path(input_dir, root, 'input_dir')
        output_dir = _resolve_path(output_dir, root, 'output_dir')
        if _same_path(input_dir, output_dir):
            raise ValueError('output_dir must differ from input_dir.')
        paths = _directory_images(input_dir)
        with self._inference_lock:
            self._require_open()
            with tqdm(paths, desc='Inference images', unit='image',
                      disable=not self._config['progress']) as images:
                for path in images:
                    image = _read_image(path)
                    height, width = image.shape[:2]
                    if self._config['use_super_resolution']:
                        height, width = (height + 15) // 16 * 16, (width + 15) // 16 * 16
                    passes = 1 if self._config['use_super_resolution'] else sum(
                        self._config[key] for key in (
                            'use_background_fading', 'use_text_enhancement', 'use_background_removal'))
                    tile_count = (count_tile_regions(height, width, self._config['tile_size'])
                                  * passes if passes else 0)
                    with tqdm(total=tile_count, desc=osp.basename(path), unit='tile',
                              leave=False, disable=not self._config['progress']) as tiles:
                        result = self._model.infer_array(image, tiles.update)
                    _write_image(osp.join(output_dir, osp.basename(path)), result, overwrite=True)
        return {
            'scale': self.scale, 'output_dir': output_dir, 'image_count': len(paths),
            'weight': self.weight,
            'use_background_removal': self._config['use_background_removal'],
            'use_dewarping': self._config['use_dewarping'],
            'use_shadow_removal': self._config['use_shadow_removal'],
            'use_super_resolution': self._config['use_super_resolution'],
            'use_text_enhancement': self._config['use_text_enhancement'],
            'text_strength': self._config['text_strength'],
            'use_background_fading': self._config['use_background_fading'],
            'background_strength': self._config['background_strength'],
        }

    def close(self):
        with self._inference_lock:
            if self._closed:
                return
            self._closed = True
            model, self._model = self._model, None
            if model is not None:
                model.clear_attention_mask_cache()
                model.close_background_remover()
                model.close_perspective_corrector()
            del model
            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def infer_document(image, **options):
    """Return an image, or DocumentResult when use_ocr=True; never auto-save."""
    if any(key in options for key in ('use_ocr', 'ocr_model', 'ocr_max_tokens', 'ocr_max_pixels')):
        from sr.document import process_document

        result = process_document(image, **options)
        return result if options.get('use_ocr', False) else result.image
    with SRX4Inferencer(**options) as inferencer:
        if isinstance(image, (str, PathLike)):
            return inferencer.infer_file(image)
        return inferencer.infer_image(image)


def run_x4_inference(config):
    """Run a batch using the same options as SRX4Inferencer."""
    if not isinstance(config, Mapping):
        raise TypeError('Inference config must be a dictionary.')
    options = dict(config)
    missing = {'input_dir', 'output_dir'} - options.keys()
    if missing:
        raise ValueError(f'Missing inference config keys: {sorted(missing)}')
    input_dir, output_dir = options.pop('input_dir'), options.pop('output_dir')
    ocr_options = {key: options.pop(key) for key in (
        'use_ocr', 'ocr_model', 'ocr_max_tokens', 'ocr_max_pixels') if key in options}
    if not isinstance(ocr_options.get('use_ocr', False), bool):
        raise ValueError('use_ocr must be true or false.')
    options = _normalize_engine_config(options)
    if ocr_options.get('use_ocr', False):
        from sr.document import DocumentResult, IMAGE_STAGES
        from sr.ocr import Yad_OCR

        root = options['project_root']
        input_dir = _resolve_path(input_dir, root, 'input_dir')
        output_dir = _resolve_path(output_dir, root, 'output_dir')
        if _same_path(input_dir, output_dir):
            raise ValueError('output_dir must differ from input_dir.')
        paths = _directory_images(input_dir)
        # Two passes reuse each model across the batch without keeping both
        # restoration and OCR weights resident on the GPU at the same time.
        if any(options[key] for key, _ in IMAGE_STAGES):
            with SRX4Inferencer(**options) as inferencer:
                inferencer.infer_directory(input_dir, output_dir)
        else:
            for path in paths:
                _write_image(osp.join(output_dir, osp.basename(path)), _read_image(path), overwrite=True)
        with Yad_OCR(
                model=ocr_options.get('ocr_model'),
                device='cuda' if options['num_gpu'] else 'cpu',
                max_tokens=ocr_options.get('ocr_max_tokens', 8192),
                max_pixels=ocr_options.get('ocr_max_pixels', 2097152)) as recognizer:
            for path in paths:
                target = osp.join(output_dir, osp.basename(path))
                markdown = recognizer.infer_image(_read_image(target))
                DocumentResult(None, markdown, []).save_markdown(target + '.md', overwrite=True)
        return {'scale': 4 if options['use_super_resolution'] else 1,
                'output_dir': output_dir, 'image_count': len(paths),
                'use_ocr': True, 'markdown_count': len(paths)}
    with SRX4Inferencer(**options) as inferencer:
        return inferencer.infer_directory(input_dir, output_dir)


INFERENCE_X4_CONFIG = {
    'runtime_platform': 'windows',
    'input_dir': r'F:\测试',
    'output_dir': r'F:\测试结果',
    'weight': INFERENCE_WEIGHT,
    'restoration_weight': RESTORATION_WEIGHT,  # 默认 weights/restoration.pkl。
    'mask_weight': MASK_WEIGHT,              # 默认 weights/document_mask.pkl。
    'num_gpu': 1,
    'tile_size': 256,          # 16的倍数，下同
    'tile_pad': 16,
    'progress': True,
    'use_super_resolution': True,            # x4 超分，可独立关闭
    'use_dewarping': False,                  # 透视矫正
    'use_shadow_removal': False,             # 原生模型去阴影；在矫正后、背景淡化前执行
    'use_background_removal': False,         # 二值化
    'use_text_enhancement': False,           # 仅加深候选文字，在二值化之前执行
    'text_strength': 0.5,                    # 文字加深强度，0 到 1
    'use_background_fading': False,         # 背景颜色淡化，保护候选文字，不二值化
    'background_strength': 0.5,             # 背景淡化强度，0 到 1
    'use_ocr': False,                       # 独立 OCR；在已启用的图像步骤之后执行
    'ocr_max_tokens': 8192,                 # 输出上限；截断会报错，不保存残缺 Markdown
    'ocr_max_pixels': 2097152,              # 保持页面比例的 OCR 视觉像素上限，避免 x4 后显存暴涨
    'project_root': PROJECT_ROOT,
}


if __name__ == '__main__':
    run_x4_inference(INFERENCE_X4_CONFIG)
