import fs from 'node:fs/promises';
import crypto from 'node:crypto';
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
    openBrowser: rendererModule.openBrowser,
    renderMedia: rendererModule.renderMedia,
    renderStill: rendererModule.renderStill,
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

const parseOpenGlRenderer = () => {
  const requested = String(process.env.VEX_REMOTION_GL || '').trim().toLowerCase();
  const supported = new Set(['swangle', 'angle', 'egl', 'swiftshader', 'vulkan', 'angle-egl']);
  return supported.has(requested) ? requested : 'swangle';
};

const readJson = async (filePath) => {
  const payload = await fs.readFile(filePath, 'utf8');
  return JSON.parse(payload);
};

const writeJson = async (filePath, payload) => {
  await fs.writeFile(filePath, JSON.stringify(payload, null, 2), 'utf8');
};

const pathExists = async (filePath) => {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
};

const readIfPresent = async (filePath) => {
  try {
    return await fs.readFile(filePath);
  } catch (error) {
    if (error && error.code === 'ENOENT') {
      return Buffer.alloc(0);
    }
    throw error;
  }
};

const bundleFingerprint = async ({entryPoint, nodeRoot}) => {
  const runtimeModule = path.join(path.dirname(entryPoint), 'remotion_scene_graph.jsx');
  const dependencyLock = path.join(nodeRoot, 'package-lock.json');
  const packageManifest = path.join(nodeRoot, 'package.json');
  const hash = crypto.createHash('sha256');
  hash.update('vex-remotion-bundle-v2\0');
  for (const filePath of [entryPoint, runtimeModule, packageManifest, dependencyLock]) {
    hash.update(path.basename(filePath));
    hash.update('\0');
    hash.update(await readIfPresent(filePath));
    hash.update('\0');
  }
  return hash.digest('hex');
};

const resolveBundle = async ({
  bundle,
  entryPoint,
  nodeRoot,
  nodeModules,
  jobDir,
  bundleProgress,
}) => {
  const fingerprint = await bundleFingerprint({entryPoint, nodeRoot});
  const configuredCache = String(process.env.VEX_REMOTION_BUNDLE_CACHE_DIR || '').trim();
  const cacheRoot = path.resolve(configuredCache || path.join(jobDir, '.remotion-bundle-cache'));
  const cacheTarget = path.join(cacheRoot, fingerprint);
  const markerPath = path.join(cacheTarget, 'vex_bundle_manifest.json');
  if (await pathExists(markerPath)) {
    const marker = await readJson(markerPath).catch(() => ({}));
    if (marker.fingerprint === fingerprint && marker.version === 2) {
      bundleProgress.push(100);
      return {
        serveUrl: cacheTarget,
        fingerprint,
        cacheHit: true,
        cacheRoot,
      };
    }
  }
  if (await pathExists(cacheTarget)) {
    const resolvedTarget = path.resolve(cacheTarget);
    const resolvedRoot = `${path.resolve(cacheRoot)}${path.sep}`;
    if (!resolvedTarget.startsWith(resolvedRoot) || path.basename(resolvedTarget) !== fingerprint) {
      throw new Error('Refusing to replace an invalid Remotion bundle cache path.');
    }
    await fs.rm(resolvedTarget, {recursive: true, force: true});
  }

  await fs.mkdir(cacheRoot, {recursive: true});
  const temporaryTarget = path.join(
    cacheRoot,
    `.build-${fingerprint}-${process.pid}-${crypto.randomBytes(5).toString('hex')}`,
  );
  const generatedServeUrl = await bundle({
    entryPoint,
    rootDir: nodeRoot,
    outDir: temporaryTarget,
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
  await writeJson(path.join(temporaryTarget, 'vex_bundle_manifest.json'), {
    version: 2,
    fingerprint,
    created_at: new Date().toISOString(),
  });
  try {
    await fs.rename(temporaryTarget, cacheTarget);
  } catch (error) {
    if (await pathExists(markerPath)) {
      await fs.rm(temporaryTarget, {recursive: true, force: true});
    } else {
      throw error;
    }
  }
  return {
    serveUrl: cacheTarget,
    generatedServeUrl,
    fingerprint,
    cacheHit: false,
    cacheRoot,
  };
};

const parseRenderMode = (request) => {
  const mode = String(request.render_mode || 'final').trim().toLowerCase();
  return new Set(['final', 'preview', 'stills']).has(mode) ? mode : 'final';
};

const previewScale = (request) => {
  const value = Number(request.preview_scale);
  return Number.isFinite(value) ? Math.max(0.25, Math.min(1, value)) : 0.5;
};

const stillFractions = (request) => {
  const source = Array.isArray(request.sample_fractions)
    ? request.sample_fractions
    : [0.08, 0.42, 0.82];
  return [...new Set(source
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
    .map((value) => Math.max(0, Math.min(1, value)).toFixed(4)))]
    .map(Number)
    .slice(0, 12);
};

const safeCandidateId = (value, index) => {
  const cleaned = String(value || '')
    .replace(/[^A-Za-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80);
  return cleaned || `candidate_${String(index + 1).padStart(2, '0')}`;
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
  const requestPath = path.join(jobDir, 'render_request.json');
  const resultPath = path.join(jobDir, 'remotion_result.json');
  const nodeModules = path.join(nodeRoot, 'node_modules');
  const timeoutInMilliseconds = parseTimeout();
  const concurrency = parseConcurrency();
  const openGlRenderer = parseOpenGlRenderer();
  const bundleProgress = [];
  const renderProgress = [];
  const browserLogs = [];

  const inputProps = await readJson(inputPropsPath);
  const request = await readJson(requestPath).catch(() => ({}));
  const renderMode = parseRenderMode(request);
  const requestedMediaContract = request.media_contract && typeof request.media_contract === 'object'
    ? request.media_contract
    : {};
  const transparent = renderMode === 'final' && request.transparent === true;
  const mediaContract = renderMode === 'preview'
    ? {
      version: 'vex-remotion-media-contract-v1',
      codec: 'h264',
      pixel_format: 'yuv420p',
      encoded_pixel_format: 'yuv420p',
      prores_profile: null,
      color_space: 'bt709',
      color_primaries: null,
      color_transfer: null,
      color_range: 'tv',
      image_format: 'png',
      has_alpha: false,
      filename: 'visual.mp4',
    }
    : {
      version: 'vex-remotion-media-contract-v1',
      codec: 'prores',
      pixel_format: transparent ? 'yuva444p10le' : 'yuv422p10le',
      encoded_pixel_format: transparent ? 'yuva444p12le' : 'yuv422p10le',
      prores_profile: transparent ? '4444' : 'hq',
      color_space: 'bt709',
      color_primaries: 'bt709',
      color_transfer: 'bt709',
      color_range: 'tv',
      image_format: 'png',
      has_alpha: transparent,
      filename: 'visual.mov',
    };
  const contractKeys = [
    'version',
    'codec',
    'pixel_format',
    'encoded_pixel_format',
    'prores_profile',
    'color_space',
    'color_primaries',
    'color_transfer',
    'color_range',
    'image_format',
    'has_alpha',
    'filename',
  ];
  const mismatches = contractKeys.filter(
    (key) => requestedMediaContract[key] !== undefined
      && requestedMediaContract[key] !== mediaContract[key],
  );
  if (mismatches.length) {
    throw new Error(`Remotion media contract mismatch: ${mismatches.join(', ')}`);
  }
  const outputLocation = path.join(jobDir, mediaContract.filename);
  const candidateBatchFile = String(request.candidate_input_props_file || '').trim();
  const candidateBatchPath = candidateBatchFile
    ? path.join(jobDir, path.basename(candidateBatchFile))
    : null;
  const candidateBatchPayload = candidateBatchPath
    ? await readJson(candidateBatchPath)
    : [];
  const candidateBatch = Array.isArray(candidateBatchPayload)
    ? candidateBatchPayload.slice(0, 8)
    : [];
  const {
    bundle,
    openBrowser,
    renderMedia,
    renderStill,
    selectComposition,
  } = await loadRemotionApis(nodeRoot);
  const bundleResolution = await resolveBundle({
    bundle,
    entryPoint,
    nodeRoot,
    nodeModules,
    jobDir,
    bundleProgress,
  });
  const serveUrl = bundleResolution.serveUrl;

  const browserExecutable = String(
    process.env.REMOTION_BROWSER_EXECUTABLE || '',
  ).trim();
  const browserOptions = browserExecutable
    ? {browserExecutable}
    : {};
  const chromiumOptions = {gl: openGlRenderer};
  const browser = await openBrowser('chrome', {
    chromiumOptions,
    logLevel: 'warn',
    ...browserOptions,
  });
  let composition;
  let stillFramePaths = [];
  let candidateStillResults = [];
  try {
    if (renderMode === 'stills') {
      const outputDir = path.join(jobDir, 'remotion_preview_frames');
      await fs.mkdir(outputDir, {recursive: true});
      const fractions = stillFractions(request);
      stillFramePaths = [];
      const renderItems = candidateBatch.length
        ? candidateBatch
        : [{candidate_id: 'primary', input_props: inputProps}];
      candidateStillResults = [];
      for (let candidateIndex = 0; candidateIndex < renderItems.length; candidateIndex += 1) {
        const renderItem = renderItems[candidateIndex] || {};
        const candidateId = safeCandidateId(renderItem.candidate_id, candidateIndex);
        const candidateInputProps = renderItem.input_props || inputProps;
        const candidateComposition = await selectComposition({
          serveUrl,
          id: COMPOSITION_ID,
          inputProps: candidateInputProps,
          logLevel: 'warn',
          timeoutInMilliseconds,
          chromiumOptions,
          puppeteerInstance: browser,
          ...browserOptions,
        });
        composition = composition || candidateComposition;
        const candidateOutputDir = path.join(outputDir, candidateId);
        await fs.mkdir(candidateOutputDir, {recursive: true});
        const candidateFrames = [];
        for (let frameIndex = 0; frameIndex < fractions.length; frameIndex += 1) {
          const fraction = fractions[frameIndex];
          const frame = Math.max(
            0,
            Math.min(
              candidateComposition.durationInFrames - 1,
              Math.round((candidateComposition.durationInFrames - 1) * fraction),
            ),
          );
          const output = path.join(
            candidateOutputDir,
            `frame_${String(frameIndex + 1).padStart(2, '0')}_${String(Math.round(fraction * 100)).padStart(2, '0')}.png`,
          );
          await renderStill({
            serveUrl,
            composition: candidateComposition,
            output,
            frame,
            inputProps: candidateInputProps,
            imageFormat: 'png',
            logLevel: 'warn',
            overwrite: true,
            timeoutInMilliseconds,
            chromiumOptions,
            puppeteerInstance: browser,
            onBrowserLog: (log) => {
              browserLogs.push({
                type: log.type,
                text: log.text,
                stackTrace: log.stackTrace,
              });
            },
            ...browserOptions,
          });
          candidateFrames.push(output);
          stillFramePaths.push(output);
        }
        candidateStillResults.push({
          candidate_id: candidateId,
          width: candidateComposition.width,
          height: candidateComposition.height,
          fps: candidateComposition.fps,
          duration_in_frames: candidateComposition.durationInFrames,
          frame_paths: candidateFrames,
        });
      }
    } else {
      composition = await selectComposition({
        serveUrl,
        id: COMPOSITION_ID,
        inputProps,
        logLevel: 'warn',
        timeoutInMilliseconds,
        chromiumOptions,
        puppeteerInstance: browser,
        ...browserOptions,
      });
      await renderMedia({
        serveUrl,
        composition,
        codec: mediaContract.codec,
        outputLocation,
        inputProps,
        muted: true,
        enforceAudioTrack: false,
        imageFormat: mediaContract.image_format,
        pixelFormat: mediaContract.pixel_format,
        proResProfile: mediaContract.prores_profile || undefined,
        colorSpace: mediaContract.color_space,
        scale: renderMode === 'preview' ? previewScale(request) : 1,
        crf: renderMode === 'preview' ? 22 : undefined,
        logLevel: 'warn',
        overwrite: true,
        timeoutInMilliseconds,
        concurrency,
        chromiumOptions,
        puppeteerInstance: browser,
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
    }
  } finally {
    await browser.close({silent: true});
  }

  await writeJson(resultPath, {
    ok: true,
    composition_id: composition.id,
    width: composition.width,
    height: composition.height,
    fps: composition.fps,
    duration_in_frames: composition.durationInFrames,
    output_location: renderMode === 'stills' ? null : outputLocation,
    still_frame_paths: stillFramePaths,
    candidate_stills: candidateStillResults,
    render_mode: renderMode,
    preview_scale: renderMode === 'preview' ? previewScale(request) : 1,
    media_contract: mediaContract,
    serve_url: serveUrl,
    bundle_fingerprint: bundleResolution.fingerprint,
    bundle_cache_hit: bundleResolution.cacheHit,
    bundle_cache_root: bundleResolution.cacheRoot,
    bundle_progress: bundleProgress,
    render_progress_samples: renderProgress.slice(-20),
    browser_logs: browserLogs.slice(-40),
    concurrency,
    open_gl_renderer: openGlRenderer,
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
