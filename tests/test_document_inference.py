"""Offline regressions for the unified pipeline; no external restoration imports."""

import os
import itertools
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT))
RUNTIME_ROOT = PROJECT_ROOT / '.runtime' / 'x4'
for key, folder in (
        ('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch'),
        ('CUDA_CACHE_PATH', 'cuda-cache'), ('MPLCONFIGDIR', 'matplotlib'),
        ('TMPDIR', 'tmp'), ('TEMP', 'tmp'), ('TMP', 'tmp')):
    location = RUNTIME_ROOT / folder
    os.environ[key] = str(location)
(RUNTIME_ROOT / 'tmp').mkdir(parents=True, exist_ok=True)
os.environ['CUDA_CACHE_DISABLE'] = '1'

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from sr.x4.infer_document_x4 import (
    _build_model_options, _normalize_engine_config, _runtime_environment,
)
from sr.models.sr_model import RestorationPipeline, _ForegroundCleaner, _PerspectiveCorrector
from sr.x4 import SRX4Inferencer, infer_document


class PipelineTests(unittest.TestCase):
    def make_model(self, dewarp=False, background=False, sr=True, text=False, fade=False, shadow=False):
        model = RestorationPipeline.__new__(RestorationPipeline)
        model.device = torch.device('cpu')
        model.opt = {
            'network_g': {'window_size': 16}, 'scale': 4,
            'tile': {'tile_size': 16, 'tile_pad': 16},
            'use_super_resolution': sr,
        }
        self.events = []

        class Upscaler(torch.nn.Module):
            def forward(inner, tensor):
                self.events.append('sr')
                return F.interpolate(tensor, scale_factor=4, mode='nearest')

        model.net_g = Upscaler()
        if background or text or fade or shadow:
            model._background_remover = types.SimpleNamespace()
        if shadow:
            model.opt['shadow_removal'] = {'enabled': True}

            def remove_shadow(image):
                self.events.append('shadow')
                return image // 2 + 30

            model._background_remover.remove_shadow = remove_shadow
        if fade:
            model.opt['background_fading'] = {'enabled': True, 'strength': 0.5}

            def fade_background(image, strength):
                self.events.append('fade')
                return ((image.astype(np.uint16) + 255) // 2).astype(np.uint8)

            model._background_remover.fade_background = fade_background
        if dewarp:
            model.opt['dewarping'] = {'enabled': True}

            def correct(image):
                self.events.append('dewarp')
                return 255 - image

            model._perspective_corrector = types.SimpleNamespace(dewarp=correct)
        if background:
            model.opt['background_removal'] = {'enabled': True}

            def remove(image):
                self.events.append('background')
                return image

            model._background_remover.remove_background = remove
        if text:
            model.opt['text_enhancement'] = {'enabled': True, 'strength': 0.5}

            def enhance(image, strength):
                self.events.append('text')
                return image // 2

            model._background_remover.enhance_text = enhance
        return model

    def test_all_sixty_four_switch_combinations_and_stage_order(self):
        image = np.arange(17 * 19 * 3, dtype=np.uint8).reshape(17, 19, 3)
        for sr, dewarp, background, text, fade, shadow in itertools.product((False, True), repeat=6):
            with self.subTest(sr=sr, dewarp=dewarp, background=background, text=text, fade=fade, shadow=shadow):
                model = self.make_model(dewarp, background, sr, text, fade, shadow)
                output = model.infer_array(image)
                expected = 255 - image if dewarp else image
                if sr:
                    expected = np.repeat(np.repeat(expected, 4, axis=0), 4, axis=1)
                if shadow:
                    expected = expected // 2 + 30
                if fade:
                    expected = ((expected.astype(np.uint16) + 255) // 2).astype(np.uint8)
                if text:
                    expected = expected // 2
                np.testing.assert_array_equal(output, expected)
                self.assertEqual(output.dtype, np.uint8)
                self.assertEqual(self.events.count('dewarp'), int(dewarp))
                self.assertEqual('sr' in self.events, sr)
                self.assertEqual('background' in self.events, background)
                self.assertEqual('text' in self.events, text)
                self.assertEqual('fade' in self.events, fade)
                self.assertEqual(self.events.count('shadow'), int(shadow))
                actual_stages = [stage for stage, _ in itertools.groupby(self.events)]
                expected_stages = [stage for enabled, stage in (
                    (sr, 'sr'), (dewarp, 'dewarp'), (shadow, 'shadow'), (fade, 'fade'),
                    (text, 'text'), (background, 'background'),
                ) if enabled]
                self.assertEqual(actual_stages, expected_stages)
                for name in ('lq', 'img', 'output'):
                    self.assertFalse(hasattr(model, name))

    def test_disabled_sr_does_not_load_network_or_weight(self):
        with patch('sr.models.sr_model.SuperResolutionNet') as network, \
                patch('sr.models.sr_model._load_checkpoint') as load:
            model = RestorationPipeline({'num_gpu': 0, 'use_super_resolution': False})
            model.clear_attention_mask_cache()
            network.assert_not_called()
            load.assert_not_called()
            self.assertIsNone(model.net_g)

    def test_no_auxiliary_load_when_disabled(self):
        model = self.make_model()
        with patch('sr.models.sr_model._PerspectiveCorrector') as dewarper, \
                patch('sr.models.sr_model._ForegroundCleaner') as remover:
            model.infer_array(np.zeros((17, 19, 3), np.uint8))
            dewarper.assert_not_called()
            remover.assert_not_called()

    def test_tiny_input_padding_and_crop(self):
        model = self.make_model()
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        expected = np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)
        np.testing.assert_array_equal(model.infer_array(image), expected)

    def test_cleanup_after_network_failure(self):
        model = self.make_model()
        model.net_g = Mock(side_effect=RuntimeError('network failed'))
        with self.assertRaisesRegex(RuntimeError, 'network failed'):
            model.infer_array(np.zeros((17, 19, 3), np.uint8))
        self.assertFalse(hasattr(model, 'lq'))
        self.assertFalse(hasattr(model, 'img'))

    def test_reject_invalid_arrays(self):
        model = self.make_model()
        for image in (None, np.zeros((17, 19)), np.zeros((17, 19, 4), np.uint8),
                      np.zeros((0, 19, 3), np.uint8)):
            with self.subTest(shape=getattr(image, 'shape', None)):
                with self.assertRaises(ValueError):
                    model.infer_array(image)

    def test_foreground_pixels_preserved(self):
        image = np.full((3, 4, 3), 42, np.uint8)
        mask = np.zeros((3, 4), bool)
        mask[1, 2] = True
        output = _ForegroundCleaner._apply_mask(image, mask)
        np.testing.assert_array_equal(output[1, 2], image[1, 2])
        self.assertTrue((output[~mask] == 255).all())

    def test_text_enhancement_preserves_background_and_color_ratios(self):
        cleaner = _ForegroundCleaner.__new__(_ForegroundCleaner)
        image = np.full((17, 19, 3), (211, 225, 239), np.uint8)
        mask = np.zeros(image.shape[:2], bool)
        mask[4:9, 6:13] = True
        image[mask] = (180, 120, 60)
        original = image.copy()
        cleaner.predict_foreground_mask = Mock(return_value=mask)
        output = cleaner.enhance_text(image, strength=0.5)
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(output[~mask], image[~mask])
        self.assertTrue((output[mask] < image[mask]).all())
        np.testing.assert_allclose(output[mask].astype(float), image[mask] * 0.675, atol=0.5)
        np.testing.assert_array_equal(cleaner.enhance_text(image, 0), image)
        cleaner.predict_foreground_mask.return_value = np.zeros_like(mask)
        np.testing.assert_array_equal(cleaner.enhance_text(image), image)
        for strength in (-0.1, 1.1, True, '0.5', float('nan'), float('inf')):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                cleaner.enhance_text(image, strength)

    def test_foreground_tiling_covers_every_pixel_without_sr(self):
        image = np.arange(37 * 43 * 3, dtype=np.uint8).reshape(37, 43, 3)
        for text in (False, True):
            model = self.make_model(background=not text, sr=False, text=text)
            callback = Mock()
            output = model.infer_array(image, callback)
            np.testing.assert_array_equal(output, image // 2 if text else image)
            self.assertEqual(callback.call_count, 9)

    def test_background_fading_preserves_foreground_and_keeps_color(self):
        cleaner = _ForegroundCleaner.__new__(_ForegroundCleaner)
        image = np.full((37, 43, 3), (35, 65, 105), np.uint8)
        mask = np.zeros(image.shape[:2], bool)
        mask[8:21, 9:34] = True
        image[mask] = (15, 25, 40)
        original = image.copy()
        cleaner.predict_foreground_mask = Mock(return_value=mask)
        previous = image
        for strength in (0, 0.25, 0.5, 1):
            output = cleaner.fade_background(image, strength)
            np.testing.assert_array_equal(image, original)
            np.testing.assert_array_equal(output[mask], image[mask])
            self.assertTrue((output[~mask] >= previous[~mask]).all())
            self.assertTrue((output[~mask] < 255).all())
            self.assertTrue((output[~mask, 0] < output[~mask, 1]).all())
            self.assertTrue((output[~mask, 1] < output[~mask, 2]).all())
            expected = np.rint(image[~mask].astype(float) + (255 - image[~mask].astype(float)) * 0.85 * strength)
            np.testing.assert_array_equal(output[~mask], expected.astype(np.uint8))
            previous = output
        cleaner.predict_foreground_mask.reset_mock()
        np.testing.assert_array_equal(cleaner.fade_background(image, 0), image)
        cleaner.predict_foreground_mask.assert_not_called()
        cleaner.predict_foreground_mask.return_value = np.ones_like(mask)
        np.testing.assert_array_equal(cleaner.fade_background(image, 1), image)
        cleaner.predict_foreground_mask.return_value = np.zeros_like(mask)
        output = cleaner.fade_background(image)
        self.assertTrue((output > image).all())
        for strength in (-0.1, 1.1, True, '0.5', float('nan'), float('inf')):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                cleaner.fade_background(image, strength)

    def test_background_fading_tiling_and_shared_model(self):
        image = np.full((37, 43, 3), 101, np.uint8)
        for text, background in itertools.product((False, True), repeat=2):
            model = self.make_model(sr=False, fade=True, text=text, background=background)
            callback = Mock()
            with patch('sr.models.sr_model._ForegroundCleaner') as constructor:
                output = model.infer_array(image, callback)
                constructor.assert_not_called()
            expected = np.full_like(image, 178 // 2 if text else 178)
            np.testing.assert_array_equal(output, expected)
            self.assertEqual(callback.call_count, 9 * (1 + int(text) + int(background)))

    def test_dewarp_only_runs_at_native_resolution(self):
        image = np.zeros((17, 19, 3), np.uint8)
        model = self.make_model(dewarp=True, sr=False)
        np.testing.assert_array_equal(model.infer_array(image), 255 - image)
        self.assertEqual(self.events, ['dewarp'])

    def test_background_receives_fully_enhanced_image_and_reuses_model(self):
        image = np.full((37, 43, 3), 200, np.uint8)
        model = self.make_model(background=True, sr=False, text=True)

        def remove(tile):
            self.events.append('background')
            np.testing.assert_array_equal(tile, np.full_like(tile, 100))
            return tile + 10

        model._background_remover.remove_background = remove
        callback = Mock()
        with patch('sr.models.sr_model._ForegroundCleaner') as constructor:
            output = model.infer_array(image, callback)
            constructor.assert_not_called()
        np.testing.assert_array_equal(output, np.full_like(image, 110))
        np.testing.assert_array_equal(image, np.full_like(image, 200))
        self.assertEqual(self.events, ['text'] * 9 + ['background'] * 9)
        self.assertEqual(callback.call_count, 18)

    def test_perspective_corrector_close_is_idempotent(self):
        model = _PerspectiveCorrector.__new__(_PerspectiveCorrector)
        model._device = torch.device('cpu')
        model._model = Mock()
        predictor = model._mask_predictor = Mock()
        model.close()
        model.close()
        predictor.close.assert_called_once()

    def test_shadow_native_prompt_bgr_padding_and_model_output(self):
        cleaner = _ForegroundCleaner.__new__(_ForegroundCleaner)
        cleaner._cv2 = cv2
        cleaner._device = torch.device('cpu')
        cleaner._dtype = torch.float32
        image = np.full((17, 19, 3), (40, 90, 150), np.uint8)
        image[5:8, 4:10] = (10, 20, 30)
        original = image.copy()
        resized = cv2.resize(image, (1024, 1024))
        prompt = cv2.resize(cv2.merge([
            cv2.medianBlur(cv2.dilate(channel, np.ones((7, 7), np.uint8)), 21)
            for channel in cv2.split(resized)
        ]), (19, 17))
        np.testing.assert_array_equal(cleaner._build_shadow_prompt(image), prompt)

        def predict(tensor):
            self.assertEqual(tuple(tensor.shape), (1, 6, 24, 24))
            expected = cv2.copyMakeBorder(np.concatenate((image, prompt), -1), 7, 0, 5, 0,
                                          cv2.BORDER_REPLICATE).astype(np.float32) / 255
            np.testing.assert_array_equal(tensor[0].numpy().transpose(1, 2, 0), expected)
            output = torch.empty((1, 3, 24, 24))
            output[:, 0] = -0.2
            output[:, 1] = 0.5
            output[:, 2] = 1.2
            return output

        cleaner._model = Mock(side_effect=predict)
        output = cleaner.remove_shadow(image)
        np.testing.assert_array_equal(output, np.full_like(image, (0, 127, 255)))
        np.testing.assert_array_equal(image, original)
        cleaner._model.assert_called_once()
        cleaner._model = None
        with self.assertRaises(RuntimeError):
            cleaner.remove_shadow(image)

    def test_shadow_large_original_uses_model_illumination_without_tiles(self):
        cleaner = _ForegroundCleaner.__new__(_ForegroundCleaner)
        cleaner._cv2 = cv2
        cleaner._device = torch.device('cpu')
        cleaner._dtype = torch.float32
        cleaner.SHADOW_MAX_SIDE = 16  # Exercise the 1600-pixel branch cheaply.
        image = np.full((17, 31, 3), (60, 90, 120), np.uint8)
        cleaner._build_shadow_prompt = Mock(return_value=np.full_like(image, 220))
        cleaner._model = Mock(return_value=torch.full((1, 3, 16, 16), 0.8))
        output = cleaner.remove_shadow(image)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, np.uint8)
        np.testing.assert_allclose(output, 204, atol=1)
        cleaner._build_shadow_prompt.assert_called_once_with(image)
        cleaner._model.assert_called_once()
        self.assertEqual(tuple(cleaner._model.call_args.args[0].shape), (1, 6, 16, 16))
        cleaner._model.return_value = torch.zeros((1, 3, 16, 16))
        self.assertTrue(np.isfinite(cleaner.remove_shadow(image)).all())

    def test_shadow_reuses_foreground_model_and_precedes_other_photometric_steps(self):
        model = self.make_model(sr=False, dewarp=True, shadow=True, fade=True, text=True, background=True)
        image = np.full((37, 43, 3), 200, np.uint8)
        callback = Mock()
        with patch('sr.models.sr_model._ForegroundCleaner') as constructor:
            output = model.infer_array(image, callback)
            constructor.assert_not_called()
        np.testing.assert_array_equal(output, np.full_like(image, 78))
        self.assertEqual(self.events, ['dewarp', 'shadow'] + ['fade'] * 9 + ['text'] * 9 + ['background'] * 9)
        self.assertEqual(callback.call_count, 27)


class PublicAPITests(unittest.TestCase):
    def fake_inferencer(self):
        model = SRX4Inferencer.__new__(SRX4Inferencer)
        model._closed = False
        model._inference_lock = threading.Lock()
        model._config = _normalize_engine_config({'progress': False})
        model._model = Mock()
        model._model.opt = {}
        model._model.infer_array.side_effect = lambda image, callback=None: image.copy()
        return model

    def test_project_local_weights_and_independent_switches(self):
        base = {'weight': 'weights/SR_x4.pth'}
        for dewarp in (False, True):
            for background in (False, True):
                config = _normalize_engine_config(dict(
                    base, use_dewarping=dewarp,
                    use_background_removal=background))
                options = _build_model_options(config)
                self.assertEqual('dewarping' in options, dewarp)
                self.assertEqual('background_removal' in options, background)
                for name in ('restoration_weight', 'mask_weight'):
                    if name in config:
                        self.assertTrue(Path(config[name]).is_relative_to(PROJECT_ROOT))

    def test_reject_nonboolean_switch(self):
        for name in ('use_dewarping', 'use_background_removal', 'use_super_resolution',
                     'use_text_enhancement', 'use_background_fading', 'use_shadow_removal'):
            with self.assertRaises(ValueError):
                _normalize_engine_config({'weight': 'ignored', name: 'true'})

    def test_foreground_modes_can_run_together_and_strength_validated(self):
        config = _normalize_engine_config({
            'use_super_resolution': False,
            'use_background_removal': True, 'use_text_enhancement': True})
        self.assertTrue(config['use_text_enhancement'])
        options = _build_model_options(config)
        self.assertIn('background_removal', options)
        self.assertIn('text_enhancement', options)
        for strength in (-1, 2, '0.5', True, float('nan'), float('inf')):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                _normalize_engine_config({'text_strength': strength})

    def test_native_resolution_api_needs_only_enabled_weights(self):
        for mode in ('use_background_removal', 'use_dewarping', 'use_text_enhancement',
                     'use_background_fading', 'use_shadow_removal'):
            config = _normalize_engine_config({
                'use_super_resolution': False, 'weight': 'not-installed/sr.pth', mode: True})
            options = _build_model_options(config)
            self.assertFalse(options['use_super_resolution'])
            self.assertIn('restoration_weight', config)
        config = _normalize_engine_config({'use_super_resolution': False})
        self.assertNotIn('weight', config)
        self.assertNotIn('restoration_weight', config)
        model = self.fake_inferencer()
        model._config = config
        self.assertEqual(model.scale, 1)
        self.assertIsNone(model.weight)

    def test_shadow_only_needs_restoration_checkpoint_and_shares_cleaner(self):
        config = _normalize_engine_config({
            'use_super_resolution': False, 'use_shadow_removal': True, 'num_gpu': 0,
            'weight': 'missing/sr.pth', 'mask_weight': 'missing/mask.pkl'})
        self.assertNotIn('restoration_weight', _normalize_engine_config({'use_super_resolution': False}))
        options = _build_model_options(config)
        self.assertEqual(options['shadow_removal']['weight'], config['restoration_weight'])
        self.assertNotIn('dewarping', options)
        with patch('sr.models.sr_model._ForegroundCleaner') as constructor:
            constructor.return_value.remove_shadow.side_effect = lambda image: image.copy()
            model = RestorationPipeline(options)
            image = np.full((17, 19, 3), 180, np.uint8)
            np.testing.assert_array_equal(model.infer_array(image), image)
            np.testing.assert_array_equal(model.infer_array(image), image)
            constructor.assert_called_once_with(model_path=config['restoration_weight'], device=torch.device('cpu'))

    def test_background_fading_options_and_strength_validation(self):
        config = _normalize_engine_config({
            'use_super_resolution': False, 'use_background_fading': True,
            'background_strength': 0.8, 'weight': 'not-installed/sr.pth'})
        options = _build_model_options(config)
        self.assertEqual(options['background_fading']['strength'], 0.8)
        self.assertEqual(options['background_fading']['weight'], config['restoration_weight'])
        self.assertNotIn('text_enhancement', options)
        self.assertNotIn('background_removal', options)
        for strength in (-1, 2, '0.5', True, float('nan'), float('inf')):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                _normalize_engine_config({'background_strength': strength})
        with patch('sr.models.sr_model._ForegroundCleaner') as constructor:
            model = RestorationPipeline(options)
            self.assertIs(model._get_background_remover(), constructor.return_value)
            self.assertIs(model._get_background_remover(), constructor.return_value)
            constructor.assert_called_once()

    def test_infer_file_unicode_and_optional_save(self):
        model = self.fake_inferencer()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT / 'tmp') as directory:
            directory = Path(directory)
            source = directory / '测试输入.png'
            output = directory / '输出.png'
            image = np.full((17, 19, 3), 42, np.uint8)
            cv2.imencode('.png', image)[1].tofile(str(source))
            result = model.infer_file(source)
            np.testing.assert_array_equal(result, image)
            self.assertEqual(list(directory.iterdir()), [source])
            model.infer_file(source, output)
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                model.infer_file(source, output)
            model.infer_file(source, output, overwrite=True)
            with self.assertRaises(ValueError):
                model.infer_file(source, source, overwrite=True)

    def test_array_api_lock_and_closed_state(self):
        model = self.fake_inferencer()
        image = np.zeros((17, 19, 3), np.uint8)
        np.testing.assert_array_equal(model.infer_image(image), image)
        model._closed = True
        with self.assertRaises(RuntimeError):
            model.infer_image(image)

    def test_one_shot_function_routes_array_and_path(self):
        import importlib
        module = importlib.import_module('sr.x4.infer_document_x4')
        with patch.object(module, 'SRX4Inferencer') as constructor:
            model = constructor.return_value.__enter__.return_value
            image = np.zeros((17, 19, 3), np.uint8)
            infer_document(image, progress=False)
            model.infer_image.assert_called_once_with(image)
            infer_document(Path('page.png'), progress=False)
            model.infer_file.assert_called_once_with(Path('page.png'))

    def test_wrapper_close_releases_all_networks_once(self):
        model = self.fake_inferencer()
        model._torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        networks = model._model
        model.close()
        model.close()
        networks.close_perspective_corrector.assert_called_once()
        networks.close_background_remover.assert_called_once()
        self.assertTrue(model.closed)

    def test_directory_unicode_filtering_and_original_filenames(self):
        model = self.fake_inferencer()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT / 'tmp') as directory:
            source = Path(directory) / 'input'
            output = Path(directory) / 'output'
            source.mkdir()
            image = np.full((17, 19, 3), 42, np.uint8)
            cv2.imencode('.png', image)[1].tofile(str(source / '中文.png'))
            cv2.imencode('.png', image)[1].tofile(str(source / 'page.png'))
            (source / 'ignore.txt').touch()
            result = model.infer_directory(source, output)
            self.assertEqual(result['image_count'], 2)
            self.assertEqual({path.name for path in output.iterdir()}, {'中文.png', 'page.png'})
            with self.assertRaises(ValueError):
                model.infer_directory(source, source)

    def test_directory_rejects_empty_and_duplicate_stems_before_writing(self):
        model = self.fake_inferencer()
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT / 'tmp') as directory:
            source = Path(directory) / 'input'
            output = Path(directory) / 'output'
            source.mkdir()
            with self.assertRaises(FileNotFoundError):
                model.infer_directory(source, output)
            (source / 'page.jpg').touch()
            (source / 'page.png').touch()
            with self.assertRaises(ValueError):
                model.infer_directory(source, output)
            self.assertFalse(output.exists())

    def test_tiling_progress_and_closed_model(self):
        model = self.fake_inferencer()
        model.set_tiling(32, 16)
        self.assertEqual(model._model.opt['tile'], {'tile_size': 32, 'tile_pad': 16})
        model._model.clear_attention_mask_cache.assert_called_once()
        model.set_progress(True)
        self.assertTrue(model._config['progress'])
        with self.assertRaises(ValueError):
            model.set_tiling(17, 16)
        with self.assertRaises(ValueError):
            model.set_progress('true')
        model._closed = True
        with self.assertRaises(RuntimeError):
            model.set_tiling(32, 16)

    def test_disabled_auxiliary_paths_are_not_validated(self):
        config = _normalize_engine_config({
            'restoration_weight': 'not-installed/restoration.pkl',
            'mask_weight': 'not-installed/document_mask.pkl',
        })
        options = _build_model_options(config)
        self.assertNotIn('dewarping', options)
        self.assertNotIn('background_removal', options)

    def test_training_keys_are_not_supported(self):
        for key in ('experiment_name', 'is_train', 'train', 'model_type'):
            with self.assertRaises(ValueError):
                _normalize_engine_config({key: 'unused'})

    def test_no_training_framework_imported(self):
        self.assertFalse(any(name == 'basicsr' or name.startswith('basicsr.')
                             for name in sys.modules))
        self.assertFalse((PROJECT_ROOT / 'sr' / 'common' / 'document_training_core.py').exists())
        self.assertFalse((PROJECT_ROOT / 'sr' / 'x4' / 'train_document_x4_finetune.py').exists())

    def test_public_package_exports(self):
        import sr
        self.assertIs(sr.SRX4Inferencer, SRX4Inferencer)
        self.assertIs(sr.infer_document, infer_document)

    def test_runtime_creates_only_the_required_temporary_directory(self):
        with patch.dict(os.environ), patch('sr.x4.infer_document_x4.os.makedirs') as mkdir:
            _runtime_environment(str(PROJECT_ROOT))
            mkdir.assert_called_once_with(str(RUNTIME_ROOT / 'tmp'), exist_ok=True)

    def test_renamed_default_weight_and_auxiliary_paths(self):
        config = _normalize_engine_config({
            'use_dewarping': True, 'use_background_removal': True})
        expected = {
            'weight': 'SR_x4.pth', 'restoration_weight': 'restoration.pkl',
            'mask_weight': 'document_mask.pkl',
        }
        for key, name in expected.items():
            self.assertEqual(Path(config[key]).name, name)
            self.assertEqual(Path(config[key]).parent, PROJECT_ROOT / 'weights')

    def test_project_relative_default_is_independent_of_working_directory(self):
        with patch('os.getcwd', return_value=str(PROJECT_ROOT / 'tests')):
            config = _normalize_engine_config({})
        self.assertEqual(Path(config['weight']), PROJECT_ROOT / 'weights' / 'SR_x4.pth')


if __name__ == '__main__':
    unittest.main()
