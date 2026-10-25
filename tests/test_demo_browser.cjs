/* Browser regressions use a self-closing local server and mocked model responses. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const vm = require('node:vm');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const qa = path.join(root, '.runtime', 'ocr', 'qa', 'shadow-browser');
const source = '# OCR 标题\n\n第一段  \n第二行\n\n<table><tr><th>项目</th><th>数量</th></tr><tr><td>Book</td><td>2</td></tr></table>\n\n\\( E = mc^2 \\)\n';
const preview = '<!doctype html><meta charset="utf-8"><style>body{padding:20px;font:15px/1.8 sans-serif}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>OCR 标题</h1><p>第一段<br>第二行</p><table><tr><th>项目</th><th>数量</th></tr><tr><td>Book</td><td>2</td></tr></table><p>\\( E = mc^2 \\)</p><script>parent.document.body.dataset.ocrUnsafe="yes"</script>';

async function liveOCR(page) {
  await page.goto(process.env.SR_DEMO_URL || 'http://127.0.0.1:8766');
  await page.locator('#file-input').setInputFiles(path.join(root, '.runtime', 'ocr', 'qa', 'table-input.png'));
  await page.locator('#super-resolution').uncheck();
  await page.locator('#enable-ocr').check();
  await page.locator('#run').click();
  await page.waitForFunction(() => !document.getElementById('run').disabled, undefined, { timeout: 300000 });
  assert.equal(await page.locator('#ocr-status').getAttribute('class'), 'ocr-status',
    await page.locator('#ocr-status').textContent());
  assert.equal(await page.locator('#download-markdown').isVisible(), true);
  const markdown = await page.locator('#ocr-raw').textContent();
  assert.match(markdown, /20261023/);
  assert.match(markdown, /Book/);
  assert.match(markdown, /<table>|\|[^\n]*Quantity/i);
  await page.frameLocator('#ocr-preview').locator('table').waitFor({ state: 'visible' });
  await downloadMarkdown(page, markdown, 'live-table.png.md');
  await page.screenshot({ path: path.join(qa, 'live-desktop.png'), fullPage: true });
  return { live: true, markdown, downloadExact: true };
}

async function downloadMarkdown(page, markdown, filename) {
  const downloading = page.waitForEvent('download');
  await page.locator('#download-markdown').click();
  const download = await downloading;
  assert.match(download.suggestedFilename(), /\.md$/);
  const destination = path.join(qa, 'downloads', filename);
  await download.saveAs(destination);
  assert.deepEqual(fs.readFileSync(destination), Buffer.from(markdown, 'utf8'));
}

async function checkLayout(page, width) {
  await page.setViewportSize({ width, height: 1000 });
  await page.locator('#ocr-preview').scrollIntoViewIfNeeded();
  const frameLayout = await page.frameLocator('#ocr-preview').locator('body').evaluate(body => ({
    viewport: window.innerWidth, body: body.getBoundingClientRect().width,
    heading: body.querySelector('h1').getBoundingClientRect().width,
  }));
  const frameBox = await page.locator('#ocr-preview').boundingBox();
  assert.equal(frameLayout.viewport, Math.round(frameBox.width));
  assert.ok(frameLayout.heading > frameLayout.viewport - 100);
  const layout = await page.evaluate(() => {
    const rect = selector => {
      const box = document.querySelector(selector).getBoundingClientRect();
      return { left: box.left, right: box.right, top: box.top, bottom: box.bottom, width: box.width };
    };
    return { previews: rect('.previews'), output: rect('#document-output'), settings: rect('.settings'),
      overflow: document.documentElement.scrollWidth > window.innerWidth,
      accent: getComputedStyle(document.documentElement).getPropertyValue('--accent').trim(),
      overlaps: [...document.querySelectorAll('.preview-head')].some(head => {
        const [label, size] = [...head.children].map(child => child.getBoundingClientRect());
        return label.right > size.left || size.right > head.getBoundingClientRect().right;
      }) };
  });
  assert.equal(layout.overflow, false, 'horizontal overflow at ' + width + 'px');
  assert.equal(layout.overlaps, false, 'preview header overlap at ' + width + 'px');
  assert.equal(Math.round(layout.output.top - layout.previews.bottom), width <= 760 ? 12 : 14);
  assert.equal(Math.round(layout.output.left), Math.round(layout.previews.left));
  assert.equal(Math.round(layout.output.width), Math.round(layout.previews.width));
  assert.equal(layout.accent, '#3a7ca8');
  if (width > 760) assert.ok(layout.settings.right < layout.previews.left);
  else assert.ok(layout.settings.bottom < layout.previews.top);
  return { width, gap: Math.round(layout.output.top - layout.previews.bottom), overflow: false };
}

async function testUnified(page, html, errors) {
  const server = http.createServer((request, response) => {
    if (request.url === '/api/status') {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ ready: true, ocr: { ready: true } }));
    } else if (request.url === '/') {
      response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      response.end(html);
    } else {
      response.writeHead(404);
      response.end();
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const origin = 'http://127.0.0.1:' + server.address().port;
  const calls = [];
  const apiRequests = [];
  let mode = 'success', release = null, gate = null;
  page.on('request', request => {
    if (request.url().startsWith(origin + '/api/')) apiRequests.push(new URL(request.url()).pathname);
  });
  try {
    await page.route('**/*', route => {
      const url = route.request().url();
      if (/^https?:/.test(url) && !url.startsWith(origin + '/')) return route.abort();
      return route.continue();
    });
    await page.goto(origin);
    await page.locator('#ocr-ready').filter({ hasText: '已就绪' }).waitFor();
    assert.equal(await page.locator('.settings #enable-ocr').count(), 1);
    assert.equal(await page.locator('#dewarping').locator('..').evaluate(label =>
      label.nextElementSibling.querySelector('input').id), 'shadow-removal');
    assert.equal(await page.locator('#ocr-source, #run-ocr, .ocr-section').count(), 0);
    assert.equal(await page.getByText('OCR 文本提取', { exact: true }).count(), 0);
    assert.equal(await page.locator('#run').isDisabled(), true);
    assert.equal(await page.locator('#ocr-device').isDisabled(), true);
    const png = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      canvas.width = 640; canvas.height = 384;
      const draw = canvas.getContext('2d');
      draw.fillStyle = '#fff'; draw.fillRect(0, 0, 640, 384);
      draw.fillStyle = '#203345'; draw.font = 'bold 28px sans-serif'; draw.fillText('文档识别测试', 48, 64);
      draw.font = '18px sans-serif'; draw.fillText('日期：2026-10-23', 48, 108);
      draw.fillText('第一段内容，保持原有阅读顺序。', 48, 150);
      draw.fillText('第二行内容。', 48, 180);
      draw.strokeStyle = '#708598';
      for (const y of [212, 260, 308]) { draw.beginPath(); draw.moveTo(48, y); draw.lineTo(592, y); draw.stroke(); }
      for (const x of [48, 440, 592]) { draw.beginPath(); draw.moveTo(x, 212); draw.lineTo(x, 308); draw.stroke(); }
      draw.fillText('项目', 64, 244); draw.fillText('数量', 460, 244);
      draw.fillText('Book', 64, 292); draw.fillText('2', 460, 292);
      return canvas.toDataURL('image/png').split(',')[1];
    });
    await page.route('**/api/infer', async route => {
      const payload = route.request().postDataJSON();
      calls.push(payload);
      if (gate) await gate;
      if (mode === 'http-error') return route.fulfill({ status: 500, json: { error: '模拟处理失败' } });
      const steps = [['super_resolution', 'x4 超分'], ['dewarping', '透视矫正'],
        ['shadow_removal', '去阴影'],
        ['background_fading', '背景淡化'], ['text_enhancement', '文字增强'], ['background_removal', '去背景 / 二值化']];
      const data = { image: png, result_id: 'browser-result', width: 640, height: 384,
        stages: steps.filter(([key]) => payload[key]).map(([, label]) => label), device: 'cpu', seconds: 0.1 };
      if (payload.ocr && mode === 'ocr-error') data.ocr_error = '模拟 OCR 失败，图像仍可保存';
      else if (payload.ocr) Object.assign(data, { markdown: source, markdown_html: preview,
        markdown_filename: 'input.png.md', ocr_result_id: 'browser-ocr' });
      if (payload.ocr && !data.ocr_error) data.stages.push('OCR');
      await route.fulfill({ json: data });
    });
    await page.locator('#file-input').setInputFiles({ name: 'input.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') });
    await page.waitForFunction(() => !document.getElementById('run').disabled);
    await page.locator('#super-resolution').uncheck();
    await page.locator('#enable-ocr').check();
    assert.equal(await page.locator('#ocr-device').isEnabled(), true);
    await page.locator('#ocr-device').selectOption('cpu');
    await page.locator('#ocr-max-tokens').selectOption('16384');
    await page.locator('#ocr-max-pixels').selectOption('4194304');
    gate = new Promise(resolve => { release = resolve; });
    const pending = page.waitForRequest('**/api/infer');
    await page.locator('#run').click();
    await pending;
    assert.equal(await page.locator('#run').isDisabled(), true);
    assert.equal(await page.locator('#enable-ocr').isDisabled(), true);
    assert.equal(await page.locator('#ocr-device').isDisabled(), true);
    assert.match(await page.locator('#status').textContent(), /识别原图/);
    release(); gate = null;
    await page.waitForFunction(() => !document.getElementById('run').disabled);
    assert.equal(calls.length, 1);
    for (const key of ['super_resolution', 'dewarping', 'shadow_removal', 'background_fading', 'text_enhancement', 'background_removal']) assert.equal(calls[0][key], false);
    assert.equal(calls[0].ocr, true);
    assert.equal(calls[0].image, png);
    assert.equal(calls[0].ocr_device, 'cpu');
    assert.equal(calls[0].ocr_max_tokens, 16384);
    assert.equal(calls[0].ocr_max_pixels, 4194304);
    assert.equal(await page.locator('#ocr-device').isEnabled(), true);
    assert.equal(await page.locator('#text-strength').isDisabled(), true);
    await page.frameLocator('#ocr-preview').locator('table').waitFor();
    assert.equal(await page.locator('#ocr-preview').getAttribute('sandbox'), '');
    assert.equal(await page.locator('body').getAttribute('data-ocr-unsafe'), null);
    await page.locator('#ocr-source-tab').click();
    assert.equal(await page.locator('#ocr-raw').textContent(), source);
    await downloadMarkdown(page, source, 'input.png.md');
    await page.locator('#ocr-source-tab').press('ArrowLeft');
    assert.equal(await page.locator('#ocr-preview-tab').getAttribute('aria-selected'), 'true');
    await page.locator('#ocr-preview').scrollIntoViewIfNeeded();
    await page.locator('#ocr-preview').screenshot({ path: path.join(qa, 'ocr-only-preview.png') });
    await page.screenshot({ path: path.join(qa, 'desktop-ocr-only.png'), fullPage: true });

    await page.locator('#super-resolution').check();
    await page.locator('#dewarping').check();
    await page.locator('#shadow-removal').check();
    await page.locator('#background-removal').check();
    assert.equal(await page.locator('#download-markdown').isVisible(), false);
    assert.equal(await page.locator('#output-canvas img').count(), 0);
    await page.locator('#run').click();
    await page.waitForFunction(() => !document.getElementById('run').disabled);
    assert.equal(calls.length, 2);
    assert.equal(calls[1].super_resolution, true);
    assert.equal(calls[1].dewarping, true);
    assert.equal(calls[1].shadow_removal, true);
    assert.equal(calls[1].background_removal, true);
    assert.equal(calls[1].ocr, true);
    assert.match(await page.locator('#stages').textContent(), /x4 超分.*透视矫正.*去阴影.*去背景.*OCR/);
    await page.frameLocator('#ocr-preview').locator('table').waitFor();
    const layouts = [];
    for (const width of [320, 375, 540, 600, 760, 768, 1000, 1024, 1440]) {
      layouts.push(await checkLayout(page, width));
      if ([375, 768, 1440].includes(width)) {
        await page.locator('#ocr-preview').screenshot({ path: path.join(qa, 'preview-' + width + 'px.png') });
        await page.screenshot({ path: path.join(qa, 'chained-' + width + 'px.png'), fullPage: true });
      }
    }
    await page.locator('#ocr-preview').screenshot({ path: path.join(qa, 'markdown-preview.png') });
    await page.locator('#ocr-max-tokens').selectOption('8192');
    assert.equal(await page.locator('#download-markdown').isVisible(), false);
    assert.equal(await page.locator('#ocr-raw').textContent(), '');

    await page.locator('#enable-ocr').uncheck();
    assert.equal(await page.locator('#ocr-device').isDisabled(), true);
    await page.locator('#super-resolution').uncheck();
    await page.locator('#dewarping').uncheck();
    await page.locator('#background-removal').uncheck();
    await page.locator('#run').click();
    await page.waitForFunction(() => !document.getElementById('run').disabled);
    assert.equal(calls[2].ocr, false);
    assert.equal(calls[2].shadow_removal, true);
    assert.equal(calls[2].super_resolution, false);
    assert.equal(calls[2].dewarping, false);
    assert.equal(calls[2].background_removal, false);
    assert.equal(await page.locator('#stages').textContent(), '去阴影');
    assert.equal(Object.hasOwn(calls[2], 'ocr_device'), false);
    assert.equal(await page.locator('#download-markdown').isVisible(), false);
    assert.equal(await page.locator('#ocr-device').isDisabled(), true);
    assert.equal(await page.locator('#output-canvas img').count(), 1);

    mode = 'ocr-error';
    await page.locator('#enable-ocr').check();
    await page.locator('#run').click();
    await page.waitForFunction(() => !document.getElementById('run').disabled);
    assert.match(await page.locator('#ocr-status').textContent(), /模拟 OCR 失败/);
    assert.equal(await page.locator('#output-canvas img').count(), 1);
    assert.equal(await page.locator('#save').isVisible(), true);
    assert.equal(await page.locator('#download-markdown').isVisible(), false);
    assert.equal(await page.locator('#ocr-device').isEnabled(), true);
    mode = 'http-error';
    await page.locator('#run').click();
    await page.waitForFunction(() => !document.getElementById('run').disabled);
    assert.match(await page.locator('#status').textContent(), /模拟处理失败/);
    assert.equal(await page.locator('#output-canvas img').count(), 0);
    assert.equal(await page.locator('#save').isVisible(), false);
    assert.equal(apiRequests.includes('/api/ocr'), false);
    assert.equal(apiRequests.filter(url => url === '/api/infer').length, 5);
    await page.locator('#remove-file').click();
    assert.equal(await page.locator('#run').isDisabled(), true);
    assert.deepEqual(errors, []);
    return { ok: true, requests: calls.length, separateOcrRequest: false,
      downloadExact: true, layouts, pageErrors: errors, artifacts: qa };
  } finally {
    if (release) release();
    await page.close();
    const closed = new Promise(resolve => server.close(resolve));
    server.closeAllConnections();
    await closed;
  }
}

async function main() {
  for (const directory of ['downloads', 'tmp', 'cache', 'crashes', 'appdata', 'localappdata', 'user']) {
    fs.mkdirSync(path.join(qa, directory), { recursive: true });
  }
  const html = fs.readFileSync(path.join(root, 'demo', 'index.html'), 'utf8');
  new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
  const context = await chromium.launchPersistentContext(path.join(qa, 'profile'), {
    channel: 'msedge', headless: true, viewport: { width: 1440, height: 1000 },
    acceptDownloads: true, downloadsPath: path.join(qa, 'downloads'),
    env: { ...process.env, TEMP: path.join(qa, 'tmp'), TMP: path.join(qa, 'tmp'), TMPDIR: path.join(qa, 'tmp'),
      APPDATA: path.join(qa, 'appdata'), LOCALAPPDATA: path.join(qa, 'localappdata'), USERPROFILE: path.join(qa, 'user') },
    args: ['--disable-breakpad', '--disable-crash-reporter', '--no-first-run', '--no-default-browser-check',
      '--disable-background-networking', '--disable-component-update', '--disable-sync',
      '--disk-cache-dir=' + path.join(qa, 'cache'), '--crash-dumps-dir=' + path.join(qa, 'crashes')],
  });
  context.setDefaultTimeout(15000);
  try {
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const result = process.argv.includes('--live') ? await liveOCR(page) : await testUnified(page, html, errors);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await context.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
