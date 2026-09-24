/** The numerical oracle never dispatches rendering. Runtime instances supply
 * real WebGL bindings for these imports instead. Fail if called accidentally. */
export function computeOnlyImports(): WebAssembly.Imports {
  const unavailable = () => -1;
  return { gpu: { ring_target: unavailable, temporal_targets: unavailable,
    draw: unavailable, reference_texture: unavailable } };
}
