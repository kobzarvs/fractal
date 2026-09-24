import { spawnSync } from 'node:child_process';
import { mkdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = resolve(root, 'public/wasm');
mkdirSync(output, { recursive: true });

function run(command, args, env = process.env) {
  const result = spawnSync(command, args, { cwd: root, env, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited with ${result.status}`);
}

const optimizer = spawnSync('wasm-opt', ['--version'], { encoding: 'utf8' });
if (optimizer.status !== 0) throw new Error('Install Binaryen (wasm-opt) before building the WASM kernels.');
const target = resolve(root, 'target/wasm-simd');
run('cargo', ['build', '--locked', '--release', '-p', 'fractal-core', '--target', 'wasm32-unknown-unknown', '--target-dir', target], {
  ...process.env,
  // Fixed 256 MiB arena: reserve the entire configured capacity at startup.
  // Browser/GPU views remain valid for the lifetime of the instance.
  RUSTFLAGS: '-C target-feature=+simd128 -C link-arg=--initial-memory=268435456 -C link-arg=--max-memory=268435456',
});
const raw = resolve(target, 'wasm32-unknown-unknown/release/fractal_core.wasm');
const binary = resolve(output, 'core-simd.wasm');
run('wasm-opt', [raw, '-O3', '--enable-bulk-memory', '--enable-sign-ext', '--enable-nontrapping-float-to-int', '--enable-simd', '-o', binary]);
console.log(`WASM SIMD: ${statSync(binary).size.toLocaleString()} bytes`);
