// Evaluate a shot-local clock independent of render order.
export function filmLayerPose(keys, progress, view) {
  let pose = [0, 0, 1, 0, 1];
  if (keys.length > 0) {
    const next = keys.findIndex((key) => key[0] >= progress);
    if (next <= 0) {
      pose = (next === -1 ? keys[keys.length - 1] : keys[0]).slice(1);
    } else {
      const left = keys[next - 1];
      const right = keys[next];
      const t = (progress - left[0]) / (right[0] - left[0]);
      const eased = t * t * (3 - 2 * t);
      pose = left.slice(1).map((value, i) => value + (right[i + 1] - value) * eased);
    }
  }
  const [x, y, scale, angle, opacity] = pose;
  const cx = view[0] / 2;
  const cy = view[1] / 2;
  return {
    transform: `translate(${x} ${y}) translate(${cx} ${cy}) rotate(${angle}) scale(${scale}) translate(${-cx} ${-cy})`,
    opacity,
  };
}
