import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const [runtime, output, fonts] = process.argv.slice(2);
const requireRuntime = createRequire(path.join(runtime, 'package.json'));
const {build} = requireRuntime('esbuild');
await build({
  entryPoints: [fileURLToPath(new URL('./studio_player.jsx', import.meta.url))],
  outfile: output,
  nodePaths: [path.join(runtime, 'node_modules')],
  bundle: true,
  minify: true,
  platform: 'browser',
  target: ['es2022'],
  format: 'iife',
  globalName: 'VexStudioPlayer',
  define: {'process.env.NODE_ENV': '"production"', __VEX_LIVE_PLAYER__: 'true'},
  plugins: [{
    name: 'studio-fonts',
    setup(builder) {
      builder.onResolve({filter: /^\.\/visual_fonts\.mjs$/}, () => ({path: fonts}));
    },
  }],
});
