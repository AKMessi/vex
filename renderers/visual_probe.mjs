import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {measureVisualScene} from './visual_telemetry.mjs';

const requestPath = path.resolve(process.argv[2]);
const request = JSON.parse(await fs.readFile(requestPath,'utf8'));
const nodeRoot = process.env.VEX_REMOTION_NODE_ROOT || path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require = createRequire(path.join(nodeRoot,'package.json'));
const {ensureBrowser} = await import(pathToFileURL(require.resolve('@remotion/renderer')).href);
const puppeteer = (await import(pathToFileURL(require.resolve('puppeteer-core')).href)).default;
const status = await ensureBrowser({logLevel:'error'});
if (!status.path) throw new Error('No Chromium executable for visual measurement');
const browser = await puppeteer.launch({executablePath:status.path,headless:true,args:['--no-sandbox','--disable-setuid-sandbox','--allow-file-access-from-files']});
try {
  const page = await browser.newPage();
  await page.setViewport({width:request.width,height:request.height});
  await page.setRequestInterception(true);
  page.on('request',r => /^(file:|data:|about:)/.test(r.url()) ? r.continue() : r.abort());
  await page.goto(pathToFileURL(path.resolve(request.html_path)).href,{waitUntil:'load',timeout:30000});
  await page.evaluate(() => document.fonts.ready);
  const samples = [];
  for (const fraction of request.fractions || [.03,.42,.68,.94]) {
    await page.evaluate((time) => {for (const timeline of Object.values(window.__timelines || {})) timeline.seek(time);},fraction*request.duration_sec);
    samples.push(await page.evaluate(measureVisualScene,Math.round(fraction*request.duration_sec*request.fps)));
  }
  await fs.writeFile(request.output_path,JSON.stringify({version:'vex-browser-telemetry-v1',available:true,samples},null,2));
} finally { await browser.close(); }
