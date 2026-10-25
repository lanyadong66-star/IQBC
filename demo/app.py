"""Local, in-memory web demo for the SR_main document restoration pipeline."""

from __future__ import annotations

import base64
import binascii
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse


DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
HTML_PATH = DEMO_DIR / 'index.html'
RESULTS_DIR = DEMO_DIR / 'results'
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_REQUEST_BYTES = 36 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
ALLOWED_HOSTS = {'127.0.0.1', 'localhost'}
INFERENCE_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sr-inference')
RESULT_LOCK = threading.Lock()
LATEST_RESULT = {}


def _configure_runtime():
    """Keep all library caches and temporary files inside this project."""
    runtime = PROJECT_ROOT / '.runtime' / 'x4'
    for key, folder in (
        ('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch'),
        ('CUDA_CACHE_PATH', 'cuda-cache'), ('MPLCONFIGDIR', 'matplotlib'),
        ('TMPDIR', 'tmp'), ('TEMP', 'tmp'), ('TMP', 'tmp'),
    ):
        os.environ[key] = str(runtime / folder)
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(MAX_IMAGE_PIXELS)
    (runtime / 'tmp').mkdir(parents=True, exist_ok=True)


_configure_runtime()

from sr.ocr import configure_ocr_runtime, extract_markdown, ocr_status
from markdown_preview import render_markdown

configure_ocr_runtime()


def _json_response(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler.send_header('Cache-Control', 'no-store')
    handler.end_headers()
    handler.wfile.write(body)


def _decode_request(payload):
    import cv2
    import numpy as np

    if not isinstance(payload, dict):
        raise ValueError('请求格式不正确。')
    encoded_image = payload.get('image')
    if not isinstance(encoded_image, str) or not encoded_image:
        raise ValueError('请先选择一张图片。')
    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError('图片数据无法读取，请重新上传。') from error
    if len(image_bytes) > 26 * 1024 * 1024:
        raise ValueError('图片文件最大为 26 MB。')
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('无法解码图片，请上传 PNG、JPG、TIFF、BMP 或 WebP 图片。')
    height, width = image.shape[:2]
    if min(height, width) < 2:
        raise ValueError('图片至少需要 2 × 2 像素。')
    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError('演示版单张图片限制为 400 万像素，以避免推理时占用过多内存。')
    if max(height, width) >= 32767:
        raise ValueError('图片边长需小于 32767 像素。')

    switches = {}
    for key in ('super_resolution', 'dewarping', 'shadow_removal', 'background_removal',
                'text_enhancement', 'background_fading', 'ocr'):
        value = payload.get(key, False)
        if not isinstance(value, bool):
            raise ValueError('处理选项必须为布尔值。')
        switches[key] = value
    text_strength = payload.get('text_strength', 0.5)
    if (isinstance(text_strength, bool) or not isinstance(text_strength, (int, float))
            or not 0 <= text_strength <= 1):
        raise ValueError('文字加深强度必须为 0 到 1 之间的数字。')
    background_strength = payload.get('background_strength', 0.5)
    if (isinstance(background_strength, bool) or not isinstance(background_strength, (int, float))
            or not 0 <= background_strength <= 1):
        raise ValueError('背景淡化强度必须为 0 到 1 之间的数字。')
    device = payload.get('device', 'cuda')
    if device not in ('cuda', 'cpu'):
        raise ValueError('设备选项不正确。')
    tile_size = payload.get('tile_size', 256)
    if type(tile_size) is not int or tile_size not in (128, 256, 384, 512):
        raise ValueError('切块尺寸选项不正确。')
    filename = payload.get('filename', 'document.png')
    if not isinstance(filename, str):
        filename = 'document.png'
    filename = Path(filename.replace('\\', '/')).name[:120] or 'document.png'
    return image, filename, switches, device, tile_size, text_strength, background_strength


def _ocr_options(payload):
    device = payload.get('ocr_device', payload.get('device', 'cuda'))
    if device not in ('cpu', 'cuda'):
        raise ValueError('OCR 设备选项不正确。')
    max_tokens = payload.get('ocr_max_tokens', 8192)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
        raise ValueError('OCR 输出上限必须为 1 到 32768 之间的整数。')
    max_pixels = payload.get('ocr_max_pixels', 2097152)
    if type(max_pixels) is not int or not 65536 <= max_pixels <= 4194304:
        raise ValueError('OCR 视觉像素上限必须为 65536 到 4194304 之间的整数。')
    return {'device': device, 'max_tokens': max_tokens, 'max_pixels': max_pixels}


def _run_ocr(image, options):
    try:
        return extract_markdown(image, **options)
    except RuntimeError as error:
        if 'out of memory' in str(error).lower():
            raise RuntimeError('OCR 显存不足，请降低 OCR 视觉分辨率或切换 OCR 设备为 CPU；调整图像分块不影响 OCR。') from error
        if 'Generation truncated' in str(error):
            raise RuntimeError('OCR 输出达到 Token 上限，未提供残缺文件。请提高 OCR 输出 Token 上限后重试。') from error
        raise


def _markdown_response(markdown, filename, result_id):
    if not isinstance(markdown, str) or not markdown.strip():
        raise RuntimeError('OCR 返回空内容，请检查输入图片。')
    return {'markdown': markdown, 'markdown_html': render_markdown(markdown),
            'markdown_filename': filename + '.md', 'ocr_result_id': result_id,
            'markdown_url': '/api/markdown/' + result_id}


def _run_inference(image, switches, device, tile_size, text_strength=0.5, background_strength=0.5):
    import torch

    from sr import SRX4Inferencer

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    stages = [label for enabled, label in (
        (switches['super_resolution'], 'x4 超分'),
        (switches['dewarping'], '透视矫正'),
        (switches.get('shadow_removal', False), '去阴影'),
        (switches.get('background_fading', False), '背景淡化'),
        (switches['text_enhancement'], '文字增强'),
        (switches['background_removal'], '去背景 / 二值化'),
    ) if enabled]
    if not stages:
        return image.copy(), ['未启用处理，返回原图']
    with SRX4Inferencer(
        num_gpu=1 if device == 'cuda' else 0,
        tile_size=tile_size, tile_pad=16, progress=False,
        use_super_resolution=switches['super_resolution'],
        use_dewarping=switches['dewarping'],
        use_shadow_removal=switches.get('shadow_removal', False),
        use_background_removal=switches['background_removal'],
        use_text_enhancement=switches['text_enhancement'], text_strength=text_strength,
        use_background_fading=switches.get('background_fading', False),
        background_strength=background_strength,
    ) as engine:
        return engine.infer_image(image), stages


class DemoHandler(BaseHTTPRequestHandler):
    server_version = 'SRDemo/1.0'

    def _host_is_local(self):
        try:
            parsed = urlparse('//' + self.headers.get('Host', ''))
            return parsed.hostname in ALLOWED_HOSTS
        except ValueError:
            return False

    def _origin_is_local(self):
        origin = self.headers.get('Origin')
        if origin is None:
            return True
        parsed = urlparse(origin)
        return (parsed.scheme == 'http' and parsed.netloc == self.headers.get('Host'))

    def do_GET(self):
        if not self._host_is_local() or not self._origin_is_local():
            _json_response(self, 403, {'error': '仅允许从本机访问此演示页面。'})
            return
        route = urlparse(self.path).path
        if route in ('/', '/index.html'):
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if route == '/api/status':
            weight_names = ('SR_x4.pth', 'restoration.pkl', 'document_mask.pkl')
            weights = {
                name: (PROJECT_ROOT / 'weights' / name).is_file()
                for name in weight_names
            }
            _json_response(self, 200, {
                'ready': all(weights.values()),
                'weights': weights,
                'max_image_pixels': MAX_IMAGE_PIXELS,
                'ocr': ocr_status(),
            })
            return
        if route.startswith('/api/markdown/'):
            requested_id = route.rsplit('/', 1)[-1]
            with RESULT_LOCK:
                if (not LATEST_RESULT.get('markdown')
                        or requested_id != LATEST_RESULT.get('ocr_id')):
                    _json_response(self, 404, {'error': 'OCR 结果已失效，请重新提取。'})
                    return
                body = LATEST_RESULT['markdown'].encode('utf-8')
                filename = LATEST_RESULT['markdown_filename']
            self.send_response(200)
            self.send_header('Content-Type', 'text/markdown; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Content-Disposition', "attachment; filename=ocr.md; filename*=UTF-8''" + quote(filename, safe=''))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        _json_response(self, 404, {'error': '页面不存在。'})

    def do_POST(self):
        if not self._host_is_local() or not self._origin_is_local():
            _json_response(self, 403, {'error': '仅允许从本机访问此演示页面。'})
            return
        route = urlparse(self.path).path
        if route not in ('/api/infer', '/api/save', '/api/ocr'):
            _json_response(self, 404, {'error': '接口不存在。'})
            return
        try:
            content_length = int(self.headers.get('Content-Length', '0'))
            if content_length < 1 or content_length > MAX_REQUEST_BYTES:
                raise ValueError('上传请求无效或过大，请检查图片大小。')
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError('请求格式不正确。')
            if route == '/api/ocr':
                started = time.perf_counter()
                options = _ocr_options(payload)
                source = payload.get('source', 'original')
                if source == 'result':
                    with RESULT_LOCK:
                        if (not LATEST_RESULT.get('id')
                                or payload.get('result_id') != LATEST_RESULT['id']):
                            raise ValueError('图像处理结果已失效，请重新处理或选择原图。')
                        image = LATEST_RESULT['image_array']
                        filename = LATEST_RESULT['filename']
                        source_id = LATEST_RESULT['id']
                elif source == 'original':
                    image, filename, *_ = _decode_request(payload)
                    source_id = None
                else:
                    raise ValueError('OCR 输入必须为原图或处理结果。')
                markdown = INFERENCE_WORKER.submit(_run_ocr, image, options).result()
                ocr_id = uuid.uuid4().hex
                response = _markdown_response(markdown, filename, ocr_id)
                with RESULT_LOCK:
                    if source_id is not None and source_id != LATEST_RESULT.get('id'):
                        raise ValueError('识别期间图像结果已更新，请对新结果重新提取。')
                    LATEST_RESULT.update(markdown=markdown, ocr_id=ocr_id,
                                         markdown_filename=response['markdown_filename'])
                response.update(seconds=round(time.perf_counter() - started, 2),
                                device=options['device'], source=source)
                _json_response(self, 200, response)
                return
            if route == '/api/save':
                if not isinstance(payload, dict):
                    raise ValueError('请求格式不正确。')
                with RESULT_LOCK:
                    if not LATEST_RESULT.get('id') or payload.get('result_id') != LATEST_RESULT['id']:
                        raise ValueError('结果已失效，请重新运行推理。')
                    if not RESULTS_DIR.resolve().is_relative_to(DEMO_DIR.resolve()):
                        raise ValueError('保存目录必须位于 demo 文件夹内。')
                    RESULTS_DIR.mkdir(exist_ok=True)
                    target = RESULTS_DIR / (LATEST_RESULT['id'] + '.png')
                    if not target.exists():
                        target.write_bytes(LATEST_RESULT['png'])
                _json_response(self, 200, {'path': str(target)})
                return
            started = time.perf_counter()
            image, filename, switches, device, tile_size, strength, background_strength = _decode_request(payload)
            ocr_options = _ocr_options(payload) if switches['ocr'] else None
            result, stages = INFERENCE_WORKER.submit(
                _run_inference, image, switches, device, tile_size, strength, background_strength).result()
            import cv2

            ok, encoded = cv2.imencode('.png', result)
            if not ok:
                raise RuntimeError('无法编码输出图片。')
            result_id = uuid.uuid4().hex
            png = encoded.tobytes()
            with RESULT_LOCK:
                LATEST_RESULT.clear()
                LATEST_RESULT.update(id=result_id, png=png, image_array=result, filename=filename)
            response = {
                'image': base64.b64encode(png).decode('ascii'),
                'result_id': result_id,
                'filename': filename,
                'width': int(result.shape[1]),
                'height': int(result.shape[0]),
                'stages': stages,
                'device': device,
                'seconds': round(time.perf_counter() - started, 2),
            }
            if switches['ocr']:
                try:
                    markdown = INFERENCE_WORKER.submit(_run_ocr, result, ocr_options).result()
                    ocr_id = uuid.uuid4().hex
                    response.update(_markdown_response(markdown, filename, ocr_id))
                    with RESULT_LOCK:
                        if LATEST_RESULT.get('id') == result_id:
                            LATEST_RESULT.update(markdown=markdown, ocr_id=ocr_id,
                                                 markdown_filename=filename + '.md')
                    response['stages'] = stages + ['OCR 文本提取']
                except Exception as error:
                    response['ocr_error'] = str(error)
                response['seconds'] = round(time.perf_counter() - started, 2)
            _json_response(self, 200, response)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            _json_response(self, 400, {'error': str(error)})
        except (FileNotFoundError, RuntimeError) as error:
            message = str(error)
            if 'out of memory' in message.lower():
                message = '显存或内存不足。请选择 128 分块，或切换到 CPU 后重试。'
            _json_response(self, 503, {'error': message})
        except Exception as error:
            if 'max_image_pixels' in str(error).lower():
                _json_response(self, 400, {'error': '图片超过 400 万像素，请缩小后再上传。'})
                return
            _json_response(self, 500, {'error': f'推理失败：{error}'})

    def log_message(self, format_string, *args):
        print('[demo] ' + format_string % args, flush=True)


def main():
    host = '127.0.0.1'
    port = int(os.environ.get('SR_DEMO_PORT', '8765'))
    server = ThreadingHTTPServer((host, port), DemoHandler)
    server.daemon_threads = True
    print(f'SR demo is ready: http://127.0.0.1:{port}', flush=True)
    print('本机访问；上传图片只在内存中处理，不会写入磁盘。', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n正在关闭演示服务。', flush=True)
    finally:
        server.server_close()
        INFERENCE_WORKER.shutdown(wait=True)


if __name__ == '__main__':
    main()
