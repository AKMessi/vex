import path from 'node:path';
import {bundle} from '@remotion/bundler';
import {openBrowser, renderMedia, selectComposition} from '@remotion/renderer';

const [root, outputLocation, width, height, fps, frames] = process.argv.slice(2);
const serveUrl = await bundle({
  entryPoint: path.join(root, 'scene.jsx'),
  publicDir: root,
  webpackOverride: config => ({
    ...config,
    resolve: {...config.resolve, modules: ['/opt/vex/node_modules', ...(config.resolve?.modules || [])]},
  }),
});
const browserExecutable = '/usr/bin/chromium';
const browser = await openBrowser('chrome', {browserExecutable, logLevel: 'error'});
try {
  const composition = await selectComposition({serveUrl, id: 'VexScene', browserExecutable, puppeteerInstance: browser});
  const expected = {width: Number(width), height: Number(height), fps: Number(fps), durationInFrames: Number(frames)};
  for (const [key, value] of Object.entries(expected)) {
    if (composition[key] !== value) throw new Error('VexScene has a different canvas or frame clock.');
  }
  await renderMedia({serveUrl, composition, codec: 'h264', outputLocation, muted: true,
    concurrency: 1, browserExecutable, puppeteerInstance: browser});
} finally {
  await browser.close({silent: true});
}
