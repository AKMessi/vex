import path from 'node:path';
import {bundle} from '@remotion/bundler';
import {openBrowser, renderMedia, selectComposition} from '@remotion/renderer';

const [root, outputLocation, width, height, fps, frames] = process.argv.slice(2);
console.error('Remotion: bundling the scene');
const serveUrl = await bundle({
  entryPoint: path.join(root, 'scene.jsx'),
  publicDir: root,
  webpackOverride: config => ({
    ...config,
    resolve: {...config.resolve, modules: ['/opt/vex/node_modules', ...(config.resolve?.modules || [])]},
  }),
});
const browserExecutable = '/usr/bin/chromium-headless-shell';
const chromeMode = 'headless-shell';
const chromiumOptions = {gl: 'swangle'};
const browserOptions = {browserExecutable, chromeMode, chromiumOptions, logLevel: 'warn'};
console.error('Remotion: opening Chromium Headless Shell');
const browser = await openBrowser('chrome', browserOptions);
try {
  console.error('Remotion: reading the composition');
  const composition = await selectComposition({serveUrl, id: 'VexScene', ...browserOptions,
    puppeteerInstance: browser, timeoutInMilliseconds: 30000});
  const expected = {width: Number(width), height: Number(height), fps: Number(fps), durationInFrames: Number(frames)};
  for (const [key, value] of Object.entries(expected)) {
    if (composition[key] !== value) throw new Error('VexScene has a different canvas or frame clock.');
  }
  console.error('Remotion: rendering the scene');
  await renderMedia({serveUrl, composition, codec: 'h264', outputLocation, muted: true,
    concurrency: 1, disallowParallelEncoding: true, offthreadVideoThreads: 1,
    ffmpegOverride: ({args}) => ['-filter_threads', '1', ...args.slice(0, -1), '-threads', '2', args.at(-1)],
    ...browserOptions, puppeteerInstance: browser, timeoutInMilliseconds: 30000});
  console.error('Remotion: media rendered');
} finally {
  await browser.close({silent: true});
}
