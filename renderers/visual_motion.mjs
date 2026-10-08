// Pure seekable motion evaluator shared by Remotion and Hyperframes.
export const motionClamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
export const springCurve = (t, config = {}) => {
  const mass = Math.max(0.1, Number(config.mass || 1));
  const stiffness = Math.max(1, Number(config.stiffness || 140));
  const damping = Math.max(0.1, Number(config.damping || 18));
  const omega = Math.sqrt(stiffness / mass);
  const zeta = damping / (2 * Math.sqrt(stiffness * mass));
  if (zeta < 1) {
    const damped = omega * Math.sqrt(1 - zeta * zeta);
    return 1 - Math.exp(-zeta * omega * t) * (Math.cos(damped * t) + zeta * omega / damped * Math.sin(damped * t));
  }
  if (Math.abs(zeta - 1) < 0.000001) return 1 - Math.exp(-omega * t) * (1 + omega * t);
  const root = Math.sqrt(zeta * zeta - 1);
  const a = -omega * (zeta - root), b = -omega * (zeta + root);
  return 1 - (b * Math.exp(a * t) - a * Math.exp(b * t)) / (b - a);
};
export const motionEase = (value, name = 'linear', config = {}) => {
  const p = motionClamp(value);
  if (p === 0 || p === 1) return p;
  if (name === 'ease_in') return p ** 3;
  if (name === 'ease_out') return 1 - (1 - p) ** 3;
  if (name === 'ease_in_out') return p * p * (3 - 2 * p);
  if (name.startsWith('spring_')) {
    const preset = name === 'spring_gentle' ? {stiffness: 75, damping: 12} : {stiffness: 180, damping: 20};
    const settings = {...preset, ...config};
    return springCurve(p, settings) / Math.max(springCurve(1, settings), 0.001);
  }
  return p;
};
export const evaluateTrack = (tracks, property, time, fallback = 0) => {
  const track = (tracks || []).find((item) => item.property === property);
  const keys = (track?.keyframes || []).filter((key) => Number.isFinite(Number(key.t)) && Number.isFinite(Number(key.value))).slice().sort((a,b) => a.t - b.t);
  if (!keys.length) return fallback;
  if (time <= keys[0].t) return Number(keys[0].value);
  if (time >= keys[keys.length - 1].t) return Number(keys[keys.length - 1].value);
  const i = keys.findIndex((key) => key.t >= time);
  const left = keys[i - 1], right = keys[i];
  const p = motionEase((time - left.t) / Math.max(right.t - left.t, 0.000001), right.easing || left.easing, track.spring || {});
  return Number(left.value) + (Number(right.value) - Number(left.value)) * p;
};
