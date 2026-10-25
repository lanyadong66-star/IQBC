"""Offline OCR orchestration and Yad_OCR adapter regressions (no weight loads)."""

import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sr.ocr import DEFAULT_OCR_MODEL, Yad_OCR, configure_ocr_runtime, ocr_status

configure_ocr_runtime()

import cv2
import numpy as np
import torch

from sr import DocumentResult, infer_document, process_document, run_x4_inference
from sr.document import IMAGE_STAGES

MARKDOWN = '# 标题\r\n\r\n段落  \r\n第二行\r\n\r\n<table><tr><td colspan="2">表格</td></tr></table>\r\n\\( E = mc^2 \\)\r\n'


class DocumentOCRTests(unittest.TestCase):
    def test_all_128_combinations_and_models_released_before_ocr(self):
        image = np.full((17, 19, 3), (20, 60, 90), np.uint8)
        keys = [key for key, _ in IMAGE_STAGES]
        for values in itertools.product((False, True), repeat=len(keys) + 1):
            image_values, use_ocr = values[:-1], values[-1]
            options = dict(zip(keys, image_values))
            with self.subTest(values=values), \
                    patch('sr.x4.infer_document_x4.SRX4Inferencer') as restore, \
                    patch('sr.document.Yad_OCR') as ocr:
                expected = image + 1 if any(image_values) else image
                model = restore.return_value.__enter__.return_value
                model.infer_image.return_value = expected

                def recognize(actual):
                    np.testing.assert_array_equal(actual, expected)
                    if any(image_values):
                        restore.return_value.__exit__.assert_called_once()
                    return MARKDOWN

                ocr.return_value.__enter__.return_value.infer_image.side_effect = recognize
                result = process_document(image, use_ocr=use_ocr, num_gpu=0, **options)
                np.testing.assert_array_equal(result.image, expected)
                self.assertEqual(result.markdown, MARKDOWN if use_ocr else None)
                if not any(image_values):
                    restore.assert_not_called()
                if not use_ocr:
                    ocr.assert_not_called()
                stages = [label for (key, label) in IMAGE_STAGES if options[key]]
                if use_ocr:
                    stages.append('OCR 文本提取')
                self.assertEqual(result.stages, stages or ['未启用处理，返回原图'])

    def test_one_shot_opt_in_and_old_return_type(self):
        image = np.zeros((17, 19, 3), np.uint8)
        with patch('sr.document.Yad_OCR') as ocr:
            ocr.return_value.__enter__.return_value.infer_image.return_value = MARKDOWN
            result = infer_document(image, use_super_resolution=False, use_ocr=True, num_gpu=0)
            self.assertIsInstance(result, DocumentResult)
            self.assertEqual(result.markdown, MARKDOWN)
            result = infer_document(image, use_super_resolution=False, use_ocr=False, num_gpu=0)
            self.assertIsInstance(result, np.ndarray)

    def test_markdown_written_exactly_without_newline_or_layout_changes(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / '.runtime' / 'ocr' / 'tmp') as directory:
            output = Path(directory) / '中文.png.md'
            result = DocumentResult(None, MARKDOWN, [])
            result.save_markdown(output)
            self.assertEqual(output.read_bytes(), MARKDOWN.encode('utf-8'))
            with self.assertRaises(FileExistsError):
                result.save_markdown(output)
            result.save_markdown(output, overwrite=True)
            with self.assertRaises(ValueError):
                result.save_markdown(output.with_suffix('.txt'))
            with self.assertRaises(ValueError):
                DocumentResult(None, None, []).save_markdown(output)

    def test_directory_ocr_writes_png_and_markdown_with_original_filename(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / '.runtime' / 'ocr' / 'tmp') as directory:
            source, output = Path(directory) / 'input', Path(directory) / 'output'
            source.mkdir()
            image = np.full((17, 19, 3), 220, np.uint8)
            cv2.imencode('.png', image)[1].tofile(str(source / '中文.png'))
            with patch('sr.ocr.Yad_OCR') as ocr:
                ocr.return_value.__enter__.return_value.infer_image.return_value = MARKDOWN
                result = run_x4_inference(dict(
                    input_dir=str(source), output_dir=str(output), num_gpu=0,
                    use_super_resolution=False, use_ocr=True))
            self.assertEqual(result['markdown_count'], 1)
            self.assertTrue((output / '中文.png').is_file())
            self.assertEqual((output / '中文.png.md').read_bytes(), MARKDOWN.encode('utf-8'))

    def test_invalid_ocr_options_and_image_types(self):
        for options in ({'device': 'remote'}, {'max_tokens': True}, {'max_tokens': 0},
                        {'max_tokens': 32769}, {'max_pixels': True}, {'max_pixels': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Yad_OCR(**options)
        for image in (None, np.zeros((4, 4)), np.zeros((1, 4, 3), np.uint8)):
            with self.subTest(image=image), self.assertRaises(ValueError):
                Yad_OCR(device='cpu').infer_image(image)
        with self.assertRaises(ValueError):
            process_document(np.zeros((2, 2, 3), np.uint8), use_ocr=1, use_super_resolution=False)

    def test_adapter_converts_bgr_to_rgb_preserves_page_and_reuses_recognizer(self):
        adapter = Yad_OCR(device='cpu')
        image = np.full((17, 19, 3), (10, 40, 200), np.uint8)
        with patch.object(adapter, '_load_model'), \
                patch.object(adapter, '_infer_page', return_value=MARKDOWN) as infer:
            self.assertEqual(adapter.infer_image(image), MARKDOWN)
            page = infer.call_args.args[0]
            self.assertEqual(page.size, (19, 17))
            self.assertEqual(page.getpixel((0, 0)), (200, 40, 10))
        adapter.close()
        adapter.close()
        self.assertIsNone(adapter._model)
        with self.assertRaises(RuntimeError):
            adapter.infer_image(image)

    def test_cache_and_temporary_paths_stay_in_project(self):
        configure_ocr_runtime()
        for key in ('HF_HOME', 'HF_HUB_CACHE', 'HF_MODULES_CACHE', 'TORCH_HOME',
                    'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR',
                    'TORCHINDUCTOR_CACHE_DIR', 'TEMP', 'TMP', 'TMPDIR'):
            self.assertTrue(Path(os.environ[key]).is_relative_to(PROJECT_ROOT), key)
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')

    def test_readiness_lists_missing_model_files(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / '.runtime' / 'ocr' / 'tmp') as directory:
            status = ocr_status(directory)
            self.assertFalse(status['ready'])
            self.assertIn('config.json', status['missing_files'])


class NativeOCRTests(unittest.TestCase):
    def test_in_memory_messages_offline_loading_and_truncation(self):
        model = MagicMock()
        model.device = 'cpu'
        model.hf_device_map = {'': 'cpu'}
        model.generation_config.eos_token_id = 9
        model.generate.return_value = torch.tensor([[1, 2, 3, 9]])
        processor = MagicMock()
        processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': 16777216}
        processor.apply_chat_template.return_value.to.return_value = {
            'input_ids': torch.tensor([[1, 2]])}
        processor.batch_decode.return_value = [MARKDOWN]
        transformers = types.SimpleNamespace(
            AutoProcessor=Mock(), Qwen3VLForConditionalGeneration=Mock())
        transformers.AutoProcessor.from_pretrained.return_value = processor
        transformers.Qwen3VLForConditionalGeneration.from_pretrained.return_value = model
        with patch.dict(sys.modules, {'transformers': transformers}), \
                patch('sr.ocr.ocr_status', return_value={'ready': True}):
            with Yad_OCR(device='cpu', max_tokens=2, max_pixels=1048576) as recognizer:
                image = np.full((100, 100, 3), 255, np.uint8)
                self.assertEqual(recognizer.infer_image(image), MARKDOWN)
                messages = processor.apply_chat_template.call_args.args[0]
                self.assertEqual(messages[-1]['content'][0]['image'].size, (100, 100))
                self.assertTrue(transformers.AutoProcessor.from_pretrained.call_args.kwargs['local_files_only'])
                self.assertEqual(processor.image_processor.size['longest_edge'], 1048576)
                self.assertEqual(recognizer.device_map, {'': 'cpu'})
                self.assertEqual(recognizer.infer_image(image), MARKDOWN)
                transformers.Qwen3VLForConditionalGeneration.from_pretrained.assert_called_once()
                model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
                with self.assertRaisesRegex(RuntimeError, 'truncated'):
                    recognizer.infer_image(image)
                model.generate.return_value = torch.tensor([[1, 2, 9]])
                processor.batch_decode.return_value = ['  ']
                with self.assertRaisesRegex(RuntimeError, 'empty'):
                    recognizer.infer_image(image)
            self.assertIsNone(recognizer._model)

    def test_cuda_restores_original_auto_bf16_policy_without_artificial_memory_cap(self):
        model, processor = MagicMock(), MagicMock()
        processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': 16777216}
        model.hf_device_map = {'model.visual': 0, 'model.language_model.layers.35': 'cpu'}
        transformers = types.SimpleNamespace(AutoProcessor=Mock(), Qwen3VLForConditionalGeneration=Mock())
        transformers.AutoProcessor.from_pretrained.return_value = processor
        transformers.Qwen3VLForConditionalGeneration.from_pretrained.return_value = model
        with patch.dict(sys.modules, {'transformers': transformers}), \
                patch('sr.ocr.ocr_status', return_value={'ready': True}), \
                patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.empty_cache'):
            with Yad_OCR(device='cuda') as recognizer:
                recognizer._load_model()
                kwargs = transformers.Qwen3VLForConditionalGeneration.from_pretrained.call_args.kwargs
                self.assertEqual(kwargs['device_map'], 'auto')
                self.assertIs(kwargs['dtype'], torch.bfloat16)
                self.assertNotIn('max_memory', kwargs)
                self.assertNotIn('quantization_config', kwargs)
                self.assertTrue(kwargs['local_files_only'])
                self.assertTrue(Path(kwargs['offload_folder']).is_relative_to(PROJECT_ROOT))
                self.assertEqual(recognizer.device_map, model.hf_device_map)

    def test_model_and_code_are_native_to_sr(self):
        import sr
        import sr.ocr

        self.assertIs(sr.Yad_OCR, Yad_OCR)
        self.assertEqual(Yad_OCR.__name__, 'Yad_OCR')
        self.assertIn('Yad_OCR', sr.__all__)
        self.assertEqual(DEFAULT_OCR_MODEL.name, 'yad_OCR')
        self.assertTrue(DEFAULT_OCR_MODEL.is_relative_to(PROJECT_ROOT / 'weights'))
        self.assertTrue(Path(sr.ocr.__file__).is_relative_to(PROJECT_ROOT / 'sr'))
        self.assertFalse(any(name == 'yad_ocr' or name.startswith('yad_ocr.') for name in sys.modules))
        self.assertTrue(ocr_status()['ready'])

    def test_yad_checkpoint_index_matches_renamed_shards_and_tensor_headers(self):
        index = json.loads((DEFAULT_OCR_MODEL / 'model.safetensors.index.json').read_text(encoding='utf-8'))
        shards = {'yad_OCR-00001-of-00002.safetensors', 'yad_OCR-00002-of-00002.safetensors'}
        self.assertEqual(set(index['weight_map'].values()), shards)
        self.assertEqual({path.name for path in DEFAULT_OCR_MODEL.glob('*.safetensors')}, shards)
        actual_map = {}
        for shard in shards:
            with (DEFAULT_OCR_MODEL / shard).open('rb') as stream:
                header_size = int.from_bytes(stream.read(8), 'little')
                header = json.loads(stream.read(header_size))
            for name in header:
                if name != '__metadata__':
                    self.assertNotIn(name, actual_map)
                    actual_map[name] = shard
        self.assertEqual(index['weight_map'], actual_map)


if __name__ == '__main__':
    unittest.main()
