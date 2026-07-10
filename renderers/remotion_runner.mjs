import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const COMPOSITION_ID = 'VexAutoVisual';

const runnerDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(runnerDir, '..');

const loadRemotionApis = async (nodeRoot) => {
  const requireFromRoot = createRequire(path.join(nodeRoot, 'package.json'));
  const bundlerPath = requireFromRoot.resolve('@remotion/bundler');
  const rendererPath = requireFromRoot.resolve('@remotion/renderer');
  const bundlerModule = await import(pathToFileURL(bundlerPath).href);
  const rendererModule = await import(pathToFileURL(rendererPath).href);
  return {
    bundle: bundlerModule.bundle,
    renderMedia: rendererModule.renderMedia,
    selectComposition: rendererModule.selectComposition,
  };
};

const parseTimeout = () => {
  const raw = String(process.env.VEX_REMOTION_TIMEOUT_MS || '').trim();
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 30000;
  }
  return Math.max(10000, Math.round(parsed));
};

const parseConcurrency = () => {
  const raw = String(process.env.VEX_REMOTION_CONCURRENCY || '').trim();
  if (!raw) {
    return null;
  }
  const parsed = Number(raw);
  if (Number.isFinite(parsed) && parsed > 0) {
    return parsed;
  }
  return raw;
};

const readJson = async (filePath) => {
  const payload = await fs.readFile(filePath, 'utf8');
  return JSON.parse(payload);
};

const writeJson = async (filePath, payload) => {
  await fs.writeFile(filePath, JSON.stringify(payload, null, 2), 'utf8');
};

const main = async () => {
  const jobDirArg = process.argv[2];
  if (!jobDirArg) {
    throw new Error('Usage: node remotion_runner.mjs <job-dir>');
  }

  const jobDir = path.resolve(jobDirArg);
  const nodeRoot = path.resolve(process.env.VEX_REMOTION_NODE_ROOT || repoRoot);
  const entryPoint = path.join(jobDir, 'entry.jsx');
  const inputPropsPath = path.join(jobDir, 'input_props.json');
  const outputLocation = path.join(jobDir, 'visual.mp4');
  const resultPath = path.join(jobDir, 'remotion_result.json');
  const bundleOutDir = path.join(jobDir, 'bundle');
  const nodeModules = path.join(nodeRoot, 'node_modules');
  const timeoutInMilliseconds = parseTimeout();
  const concurrency = parseConcurrency();
  const bundleProgress = [];
  const renderProgress = [];
  const browserLogs = [];

  const inputProps = await readJson(inputPropsPath);
  await fs.rm(bundleOutDir, {recursive: true, force: true});
  const {bundle, renderMedia, selectComposition} = await loadRemotionApis(nodeRoot);

  const serveUrl = await bundle({
    entryPoint,
    rootDir: nodeRoot,
    outDir: bundleOutDir,
    publicDir: null,
    onProgress: (progress) => {
      bundleProgress.push(Math.round(progress));
    },
    webpackOverride: (webpackConfig) => {
      const existingResolve = webpackConfig.resolve || {};
      const existingModules = existingResolve.modules || [];
      return {
        ...webpackConfig,
        resolve: {
          ...existingResolve,
          modules: [nodeModules, ...existingModules],
        },
      };
    },
  });

  const browserExecutable = String(
    process.env.REMOTION_BROWSER_EXECUTABLE || '',
  ).trim();
  const browserOptions = browserExecutable
    ? {browserExecutable}
    : {};

  const composition = await selectComposition({
    serveUrl,
    id: COMPOSITION_ID,
    inputProps,
    logLevel: 'warn',
    timeoutInMilliseconds,
    ...browserOptions,
  });

  await renderMedia({
    serveUrl,
    composition,
    codec: 'h264',
    outputLocation,
    inputProps,
    muted: true,
    enforceAudioTrack: false,
    imageFormat: 'jpeg',
    logLevel: 'warn',
    overwrite: true,
    timeoutInMilliseconds,
    concurrency,
    onBrowserLog: (log) => {
      browserLogs.push({
        type: log.type,
        text: log.text,
        stackTrace: log.stackTrace,
      });
    },
    onProgress: (progress) => {
      renderProgress.push({
        progress: Number(progress.progress || 0),
        renderedFrames: progress.renderedFrames ?? null,
        encodedFrames: progress.encodedFrames ?? null,
        stitchStage: progress.stitchStage ?? null,
      });
    },
    ...browserOptions,
  });

  await writeJson(resultPath, {
    ok: true,
    composition_id: composition.id,
    width: composition.width,
    height: composition.height,
    fps: composition.fps,
    duration_in_frames: composition.durationInFrames,
    output_location: outputLocation,
    serve_url: serveUrl,
    bundle_progress: bundleProgress,
    render_progress_samples: renderProgress.slice(-20),
    browser_logs: browserLogs.slice(-40),
    concurrency,
  });
};

main().catch(async (error) => {
  const jobDirArg = process.argv[2];
  if (jobDirArg) {
    const jobDir = path.resolve(jobDirArg);
    await writeJson(path.join(jobDir, 'remotion_result.json'), {
      ok: false,
      error: error && error.stack ? String(error.stack) : String(error),
    }).catch(() => {});
  }
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
