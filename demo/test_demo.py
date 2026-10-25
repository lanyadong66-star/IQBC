"""Demo regressions: options, device mapping, HTTP validation and local saving."""

import base64
import itertools
import json
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import app

import cv2
import numpy as np


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((17, 19, 3), 230, np.uint8)
        self.image[4:7, 3:16] = (30, 45, 60)
        self.png = cv2.imencode('.png', self.image)[1].tobytes()
        self.payload = {
            'image': base64.b64encode(self.png).decode('ascii'),
            'filename': '测试输入.png',
            'super_resolution': True,
        }

    def test_decode_valid_image_and_filename(self):
        image, filename, switches, device, tile, strength, background_strength = app._decode_request(self.payload)
        np.testing.assert_array_equal(image, self.image)
        self.assertEqual(filename, '测试输入.png')
        self.assertTrue(switches['super_resolution'])
        self.assertFalse(switches['background_removal'])
        self.assertEqual((device, tile), ('cuda', 256))
        self.assertEqual(strength, 0.5)
        self.assertFalse(switches['background_fading'])
        self.assertFalse(switches['shadow_removal'])
        self.assertEqual(background_strength, 0.5)

    def test_reject_invalid_payloads_and_switches(self):
        cases = [None, [], {}, {'image': 'not base64'}]
        for key, value in (
            ('device', 'invalid'), ('tile_size', 7), ('super_resolution', 1),
            ('background_removal', 'true'), ('tile_size', 128.0),
            ('text_enhancement', 1), ('text_strength', True), ('text_strength', -1),
            ('text_strength', 2), ('text_strength', '0.5'), ('text_strength', float('nan')),
            ('background_fading', 1), ('background_strength', True), ('background_strength', -1),
            ('background_strength', 2), ('background_strength', '0.5'),
            ('background_strength', float('nan')), ('background_strength', float('inf')),
            ('ocr', 1), ('ocr', 'true'),
            ('shadow_removal', 1), ('shadow_removal', 'true'),
        ):
            cases.append(dict(self.payload, **{key: value}))
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                app._decode_request(payload)

    def test_reject_tiny_or_oversized_images(self):
        for shape in ((1, 19, 3), (17, 1, 3)):
            encoded = cv2.imencode('.png', np.zeros(shape, np.uint8))[1]
            with self.assertRaises(ValueError):
                app._decode_request(dict(self.payload, image=base64.b64encode(encoded).decode()))
        with patch.object(app, 'MAX_IMAGE_PIXELS', 10), self.assertRaises(ValueError):
            app._decode_request(self.payload)

    def test_filename_has_no_directory_component(self):
        _, filename, *_ = app._decode_request(dict(self.payload, filename='../../x/测试.png'))
        self.assertEqual(filename, '测试.png')

    def test_all_sixty_four_switch_combinations(self):
        keys = ('super_resolution', 'dewarping', 'shadow_removal', 'background_removal', 'text_enhancement', 'background_fading')
        for values in itertools.product((False, True), repeat=len(keys)):
            switches = dict(zip(keys, values))
            with self.subTest(switches=switches), \
                    patch('sr.SRX4Inferencer') as sr_class:
                sr_class.return_value.__enter__.return_value.infer_image.side_effect = (
                    lambda image: np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)
                    if switches['super_resolution'] else image.copy())
                output, stages = app._run_inference(self.image, switches, 'cuda', 128)
                expected = (68, 76, 3) if switches['super_resolution'] else self.image.shape
                self.assertEqual(output.shape, expected)
                self.assertEqual(output.dtype, np.uint8)
                self.assertTrue(stages)
                if any(values):
                    self.assertEqual(sr_class.call_args.kwargs['num_gpu'], 1)
                    self.assertEqual(sr_class.call_args.kwargs['use_super_resolution'], switches['super_resolution'])
                    self.assertEqual(sr_class.call_args.kwargs['use_dewarping'], switches['dewarping'])
                    self.assertEqual(sr_class.call_args.kwargs['use_shadow_removal'], switches['shadow_removal'])
                    self.assertEqual(sr_class.call_args.kwargs['use_background_removal'], switches['background_removal'])
                    use_text = switches['text_enhancement']
                    self.assertEqual(sr_class.call_args.kwargs['use_text_enhancement'], use_text)
                    self.assertEqual(sr_class.call_args.kwargs['use_background_fading'], switches['background_fading'])
                    self.assertEqual(stages, [label for enabled, label in (
                        (switches['super_resolution'], 'x4 超分'),
                        (switches['dewarping'], '透视矫正'),
                        (switches['shadow_removal'], '去阴影'),
                        (switches['background_fading'], '背景淡化'),
                        (use_text, '文字增强'),
                        (switches['background_removal'], '去背景 / 二值化'),
                    ) if enabled])
                else:
                    sr_class.assert_not_called()
                    np.testing.assert_array_equal(output, self.image)

    def test_cpu_is_zero_gpu_count(self):
        switches = dict(super_resolution=True, dewarping=False, background_removal=False, text_enhancement=False)
        with patch('sr.SRX4Inferencer') as constructor:
            constructor.return_value.__enter__.return_value.infer_image.return_value = self.image
            app._run_inference(self.image, switches, 'cpu', 128)
            self.assertEqual(constructor.call_args.kwargs['num_gpu'], 0)

    def test_gpu_unavailable_is_actionable(self):
        switches = dict(super_resolution=True, dewarping=False, background_removal=False, text_enhancement=False)
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CPU'):
                app._run_inference(self.image, switches, 'cuda', 128)

    def test_text_strength_forwarded_and_background_keeps_enhancement(self):
        payload = dict(self.payload, background_removal=True, text_enhancement=True, text_strength=0.8)
        image, _, switches, device, tile_size, strength, _ = app._decode_request(payload)
        self.assertTrue(switches['text_enhancement'])
        with patch('sr.SRX4Inferencer') as constructor:
            constructor.return_value.__enter__.return_value.infer_image.return_value = image
            _, stages = app._run_inference(image, switches, device, tile_size, strength)
            self.assertTrue(constructor.call_args.kwargs['use_text_enhancement'])
            self.assertEqual(constructor.call_args.kwargs['text_strength'], 0.8)
            self.assertEqual(stages, ['x4 超分', '文字增强', '去背景 / 二值化'])

    def test_background_fading_is_independent_and_strength_forwarded(self):
        payload = dict(self.payload, super_resolution=False,
                       background_fading=True, background_strength=0.75)
        image, _, switches, device, tile_size, text_strength, strength = app._decode_request(payload)
        with patch('sr.SRX4Inferencer') as constructor:
            constructor.return_value.__enter__.return_value.infer_image.return_value = image
            output, stages = app._run_inference(image, switches, device, tile_size, text_strength, strength)
            self.assertEqual(stages, ['背景淡化'])
            self.assertEqual(output.shape, image.shape)
            self.assertTrue(constructor.call_args.kwargs['use_background_fading'])
            self.assertEqual(constructor.call_args.kwargs['background_strength'], 0.75)
            self.assertFalse(constructor.call_args.kwargs['use_super_resolution'])
            self.assertFalse(constructor.call_args.kwargs['use_background_removal'])
            self.assertFalse(constructor.call_args.kwargs['use_text_enhancement'])

    def test_runtime_cache_paths_are_project_local(self):
        import os

        for key in ('TEMP', 'TMP', 'TMPDIR', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'XDG_CACHE_HOME'):
            self.assertTrue(Path(os.environ[key]).is_relative_to(app.PROJECT_ROOT))


class HTTPTests(unittest.TestCase):
    setUp = DemoTests.setUp

    @classmethod
    def setUpClass(cls):
        cls.server = app.ThreadingHTTPServer(('127.0.0.1', 0), app.DemoHandler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=5)

    def request(self, path, payload=None, headers=None):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        try:
            request_headers = {'Content-Type': 'application/json'}
            request_headers.update(headers or {})
            body = json.dumps(payload).encode('utf-8') if payload is not None else None
            connection.request('POST' if body is not None else 'GET', path, body, request_headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, raw
        finally:
            connection.close()

    def test_status_and_page(self):
        code, body = self.request('/api/status')
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(body)['ready'])
        code, body = self.request('/')
        self.assertEqual(code, 200)
        self.assertIn(b'super-resolution', body)
        self.assertIn(b'text-enhancement', body)
        self.assertIn(b'background-fading', body)
        self.assertIn(b'shadow-removal', body)
        self.assertLess(body.index(b'id="dewarping"'), body.index(b'id="shadow-removal"'))
        self.assertLess(body.index(b'id="shadow-removal"'), body.index(b'id="background-fading"'))
        self.assertIn(b'background-strength', body)
        self.assertNotIn(b'CLAHE', body)
        self.assertIn(b'result-stack', body)
        self.assertIn(b'document-output', body)
        self.assertIn(b'enable-ocr', body)
        self.assertNotIn(b'id="ocr-source"', body)
        self.assertNotIn(b'id="run-ocr"', body)
        self.assertNotIn(b'ocr-section', body)
        self.assertNotIn('OCR 文本提取'.encode('utf-8'), body)
        self.assertIn(b'sandbox=""', body)
        self.assertIn(b'download-markdown', body)

    def test_ocr_original_layout_preview_and_exact_markdown_download(self):
        markdown = '# 中文标题\r\n\r\n段落  \r\n换行\r\n<table><tr><td colspan="2">表格</td></tr></table>\r\n\\( x^2 \\)'
        with patch.object(app, '_run_ocr', return_value=markdown) as ocr:
            code, body = self.request('/api/ocr', dict(self.payload, source='original',
                                                     ocr_device='cpu', ocr_max_tokens=4096))
        self.assertEqual(code, 200)
        response = json.loads(body)
        self.assertEqual(response['markdown'], markdown)
        self.assertIn('<table>', response['markdown_html'])
        self.assertIn('Content-Security-Policy', response['markdown_html'])
        self.assertIn('\\( x^2 \\)', response['markdown_html'])
        np.testing.assert_array_equal(ocr.call_args.args[0], self.image)
        self.assertEqual(ocr.call_args.args[1]['device'], 'cpu')
        code, body = self.request(response['markdown_url'])
        self.assertEqual(code, 200)
        self.assertEqual(body, markdown.encode('utf-8'))
        self.assertEqual(self.request('/api/markdown/wrong')[0], 404)

    def test_ocr_consumes_final_processed_image_not_original(self):
        processed = self.image // 2
        with patch.object(app, '_run_inference', return_value=(processed, ['test'])):
            code, body = self.request('/api/infer', self.payload)
            self.assertEqual(code, 200)
            result_id = json.loads(body)['result_id']
        with patch.object(app, '_run_ocr', return_value='# 结果') as ocr:
            code, body = self.request('/api/ocr', {'source': 'result', 'result_id': result_id})
        self.assertEqual(code, 200)
        np.testing.assert_array_equal(ocr.call_args.args[0], processed)
        self.assertEqual(json.loads(body)['source'], 'result')
        self.assertEqual(self.request('/api/ocr', {'source': 'result', 'result_id': 'old'})[0], 400)

    def test_infer_api_can_chain_ocr_and_preserves_image_when_ocr_fails(self):
        with patch.object(app, '_run_inference', return_value=(self.image, ['test'])), \
                patch.object(app, '_run_ocr', return_value='# 标题'):
            code, body = self.request('/api/infer', dict(self.payload, ocr=True))
        self.assertEqual(code, 200)
        response = json.loads(body)
        self.assertEqual(response['markdown'], '# 标题')
        self.assertEqual(response['stages'][-1], 'OCR 文本提取')
        with patch.object(app, '_run_inference', return_value=(self.image, ['test'])), \
                patch.object(app, '_run_ocr', side_effect=RuntimeError('OCR unavailable')):
            code, body = self.request('/api/infer', dict(self.payload, ocr=True))
        response = json.loads(body)
        self.assertEqual(code, 200)
        self.assertIn('image', response)
        self.assertIn('ocr_error', response)
        self.assertNotIn('markdown', response)

    def test_unified_ocr_only_uses_original_image_without_restoration_model(self):
        with patch('sr.SRX4Inferencer') as image_model, \
                patch.object(app, '_run_ocr', return_value='# 原图识别') as ocr:
            code, body = self.request('/api/infer', dict(
                self.payload, super_resolution=False, dewarping=False,
                background_fading=False, text_enhancement=False,
                background_removal=False, ocr=True, device='cpu', ocr_device='cpu'))
        self.assertEqual(code, 200)
        image_model.assert_not_called()
        np.testing.assert_array_equal(ocr.call_args.args[0], self.image)
        response = json.loads(body)
        self.assertEqual(response['markdown'], '# 原图识别')
        np.testing.assert_array_equal(
            cv2.imdecode(np.frombuffer(base64.b64decode(response['image']), np.uint8), cv2.IMREAD_COLOR),
            self.image)

    def test_unified_chain_passes_processed_image_and_independent_ocr_options(self):
        processed = self.image // 2
        with patch.object(app, '_run_inference', return_value=(processed, ['去背景'])) as image_run, \
                patch.object(app, '_run_ocr', return_value='# 处理后识别') as ocr:
            code, body = self.request('/api/infer', dict(
                self.payload, background_removal=True, ocr=True,
                ocr_device='cpu', ocr_max_tokens=4096, ocr_max_pixels=1048576))
        self.assertEqual(code, 200)
        self.assertTrue(image_run.call_args.args[1]['super_resolution'])
        self.assertTrue(image_run.call_args.args[1]['background_removal'])
        np.testing.assert_array_equal(ocr.call_args.args[0], processed)
        self.assertEqual(ocr.call_args.args[1], {
            'device': 'cpu', 'max_tokens': 4096, 'max_pixels': 1048576})
        response = json.loads(body)
        self.assertEqual(response['markdown'], '# 处理后识别')
        code, downloaded = self.request(response['markdown_url'])
        self.assertEqual(code, 200)
        self.assertEqual(downloaded, response['markdown'].encode('utf-8'))

    def test_shadow_only_and_shadow_then_ocr_http_flow(self):
        processed = self.image // 2 + 80
        for use_ocr in (False, True):
            with self.subTest(use_ocr=use_ocr), patch('sr.SRX4Inferencer') as model, \
                    patch.object(app, '_run_ocr', return_value='# 去阴影后识别') as ocr:
                model.return_value.__enter__.return_value.infer_image.return_value = processed
                code, body = self.request('/api/infer', dict(
                    self.payload, super_resolution=False, shadow_removal=True, ocr=use_ocr, device='cpu'))
                self.assertEqual(code, 200)
                response = json.loads(body)
                self.assertEqual(response['stages'], ['去阴影'] + (['OCR 文本提取'] if use_ocr else []))
                self.assertTrue(model.call_args.kwargs['use_shadow_removal'])
                self.assertFalse(model.call_args.kwargs['use_super_resolution'])
                np.testing.assert_array_equal(
                    cv2.imdecode(np.frombuffer(base64.b64decode(response['image']), np.uint8), cv2.IMREAD_COLOR),
                    processed)
                if use_ocr:
                    np.testing.assert_array_equal(ocr.call_args.args[0], processed)
                    model.return_value.__exit__.assert_called_once()
                else:
                    ocr.assert_not_called()

    def test_reject_invalid_ocr_options_before_inference(self):
        for options in ({'source': 'remote'}, {'ocr_device': 'remote'}, {'ocr_max_tokens': True},
                        {'ocr_max_tokens': 0}, {'ocr_max_tokens': 32769},
                        {'ocr_max_pixels': True}, {'ocr_max_pixels': 1}):
            with self.subTest(options=options), patch.object(app, '_run_ocr') as ocr:
                code, _ = self.request('/api/ocr', dict(self.payload, **options))
                self.assertEqual(code, 400)
                ocr.assert_not_called()

    def test_model_html_is_confined_to_sandboxed_preview(self):
        source = '<script>parent.document.body.innerHTML="bad"</script><table><tr><td>ok</td></tr></table>'
        preview = app.render_markdown(source)
        self.assertIn("default-src 'none'", preview)
        self.assertIn("form-action 'none'", preview)
        self.assertIn('<table>', preview)
        self.assertNotIn('<script>', preview)
        hostile = '<meta http-equiv="refresh" content="0;url=https://outside.example"><img src="https://outside.example/x"><table onclick="alert(1)"><tr><td colspan="2" style="text-align:center;background:url(https://outside.example)">ok</td></tr></table>'
        preview = app.render_markdown(hostile)
        self.assertNotIn('outside.example', preview)
        self.assertNotIn('onclick', preview)
        self.assertIn('colspan="2"', preview)
        self.assertIn('text-align:center', preview)

    def test_reject_unknown_route_and_foreign_origins(self):
        code, _ = self.request('/api/missing')
        self.assertEqual(code, 404)
        code, _ = self.request('/api/infer', self.payload, {'Origin': 'https://untrusted.example'})
        self.assertEqual(code, 403)
        code, _ = self.request('/api/status', headers={'Host': 'untrusted.example'})
        self.assertEqual(code, 403)

    def test_bad_image_has_a_json_error(self):
        code, body = self.request('/api/infer', {'image': 'bad'})
        self.assertEqual(code, 400)
        self.assertIn('error', json.loads(body))

    def test_infer_and_explicit_project_local_save(self):
        temp_parent = app.DEMO_DIR / '.runtime' / 'tmp'
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as directory:
            target_dir = Path(directory) / 'results'
            with patch.object(app, 'RESULTS_DIR', target_dir), \
                    patch.object(app, '_run_inference', return_value=(self.image, ['test'])):
                code, body = self.request('/api/infer', self.payload)
                self.assertEqual(code, 200)
                result = json.loads(body)
                self.assertFalse(target_dir.exists())
                code, body = self.request('/api/save', {'result_id': result['result_id']})
                self.assertEqual(code, 200)
                saved = Path(json.loads(body)['path'])
                self.assertTrue(saved.is_relative_to(app.DEMO_DIR))
                self.assertEqual(saved.read_bytes(), self.png)
                code, _ = self.request('/api/save', {'result_id': 'wrong'})
                self.assertEqual(code, 400)

    def test_background_fading_http_parameters(self):
        payload = dict(self.payload, super_resolution=False,
                       background_fading=True, background_strength=0.65)
        with patch.object(app, '_run_inference', return_value=(self.image, ['背景淡化'])) as infer:
            code, body = self.request('/api/infer', payload)
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(body)['stages'], ['背景淡化'])
            args = infer.call_args.args
            self.assertTrue(args[1]['background_fading'])
            self.assertFalse(args[1]['background_removal'])
            self.assertEqual(args[-1], 0.65)


if __name__ == '__main__':
    unittest.main()
