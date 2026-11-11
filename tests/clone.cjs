const assert = require('node:assert/strict');
require('node:dns').setDefaultResultOrder('ipv4first');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE ||
  'C:/Users/12495/AppData/Local/OpenAI/Codex/runtimes/cua_node/db1cc9e06a6c48d6/bin/node_modules/playwright');

const base = process.env.VOICE_LAB_URL || 'http://localhost:8001';
const output = path.resolve(__dirname, '..', '.test-output', 'clone');

async function noOverflow(page) {
  const size = await page.evaluate(() => ({
    width: innerWidth, content: document.documentElement.scrollWidth
  }));
  assert(size.content <= size.width + 1, JSON.stringify(size));
}

async function main() {
  await fs.mkdir(output, { recursive: true });
  const state = await (await fetch(`${base}/api/clone/status`)).json();
  assert(state.qualities.includes('fast'));
  assert(state.recognition);
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    headless: true
  });
  const errors = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1080 }, acceptDownloads: true });
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.waitForFunction(() => document.getElementById('cloneEngine').textContent.includes('AI'));
    await page.locator('#fileInput').setInputFiles('C:/Users/12495/Desktop/素材.mp3');
    await page.waitForFunction(() => !document.getElementById('prepareClone').disabled);
    await page.locator('#prepareClone').click();
    await page.waitForFunction(() => !document.getElementById('cloneReferenceEditor').hidden &&
      !document.getElementById('cloneTranscript').disabled &&
      document.getElementById('cloneTranscript').value.length > 0, null, { timeout: 60000 });
    await page.locator('#cloneName').fill('QA clone voice');
    await page.locator('#saveClone').click();
    await page.waitForFunction(() => document.getElementById('cloneProgress').textContent.includes('已保存') &&
      !document.getElementById('cloneText').disabled);
    const text = '今天我们换一种方式讲故事。声音的细节、说话的节奏，还有每句话之间的停顿，都值得认真对待。';
    await page.locator('#cloneText').fill(text);
    await page.locator('#cloneQuality').selectOption('fast');
    await page.locator('#generateClone').click();
    await page.waitForFunction(() => !document.getElementById('cloneResult').hidden, null, { timeout: 120000 });
    await page.locator('#cloneResultAudio').evaluate(audio => audio.pause());
    const timing = await page.locator('#cloneTiming').textContent();
    const downloaded = page.waitForEvent('download');
    await page.locator('#downloadClone').click();
    await (await downloaded).saveAs(path.join(output, 'browser-generated.wav'));
    const bytes = await fs.readFile(path.join(output, 'browser-generated.wav'));
    assert.equal(bytes.toString('ascii', 0, 4), 'RIFF');
    assert(bytes.length > 24000);
    await noOverflow(page);
    await page.screenshot({ path: path.join(output, 'desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await noOverflow(page);
    await page.screenshot({ path: path.join(output, 'mobile.png'), fullPage: true });

    const video = await context.newPage();
    video.on('pageerror', error => errors.push(error.message));
    await video.goto(`${base}/video.html`);
    await video.locator('#useCloneAudio').click();
    await video.waitForFunction(() => document.getElementById('voiceName').textContent.includes('QA clone voice'));
    await video.locator('#openTts').click();
    await video.waitForFunction(() => [...document.getElementById('ttsVoice').options].some(option => option.value.startsWith('clone:')));
    const option = await video.locator('#ttsVoice option').allTextContents();
    assert(option.some(name => name.includes('AI 克隆')));

    await page.reload();
    await page.waitForFunction(() => document.getElementById('cloneName').value === 'QA clone voice');
    assert.equal(await page.locator('#cloneText').inputValue(), text);
    await page.locator('[aria-label="删除克隆音色：QA clone voice"]').click();
    await page.waitForFunction(() => document.getElementById('cloneReferenceEditor').hidden);
    assert.equal(await page.locator('.clone-profile').filter({ hasText: 'QA clone voice' }).count(), 0);
    await page.locator('#fileInput').setInputFiles({ name: 'broken.wav', mimeType: 'audio/wav', buffer: Buffer.from('invalid') });
    await page.waitForFunction(() => document.getElementById('referenceStatus').textContent.includes('无法'));
    assert(await page.locator('#prepareClone').isDisabled());
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ timing, errors, bytes: bytes.length, output }));
    console.log('PASS: real audio import, local ASR, profile persistence/deletion, clone generation/download, desktop/mobile, video transfer, invalid audio.');
    await context.close();
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
