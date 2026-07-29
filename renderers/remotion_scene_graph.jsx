import React from 'react';
import {fitText} from '@remotion/layout-utils';
import {interpolate} from 'remotion';

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const text = (value) => String(value || '').trim();
const list = (value) => (Array.isArray(value) ? value : []);
const number = (value, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const colorFor = (value, palette) => {
  const colors = {
    background: palette.bg,
    surface: palette.surface,
    ink: palette.ink,
    text: palette.text,
    muted: palette.muted,
    accent: palette.accent,
    accent_secondary: palette.accent2,
    grid: `${palette.accent2}55`,
  };
  return colors[text(value)] || text(value) || 'transparent';
};

const easingValue = (value, easing) => {
  const progress = clamp(value, 0, 1);
  if (easing === 'ease_in') return progress ** 3;
  if (easing === 'ease_out' || easing === 'spring_snappy') return 1 - (1 - progress) ** 3;
  if (easing === 'ease_in_out' || easing === 'spring_gentle') return progress * progress * (3 - 2 * progress);
  return progress;
};

const trackValue = (tracks, property, progress, fallback) => {
  const track = list(tracks).find((item) => text(item.property) === property);
  const keyframes = list(track?.keyframes)
    .filter((item) => Number.isFinite(Number(item?.t)) && Number.isFinite(Number(item?.value)))
    .sort((a, b) => Number(a.t) - Number(b.t));
  if (!keyframes.length) return fallback;
  if (progress <= Number(keyframes[0].t)) return Number(keyframes[0].value);
  if (progress >= Number(keyframes[keyframes.length - 1].t)) return Number(keyframes[keyframes.length - 1].value);
  const rightIndex = keyframes.findIndex((item) => Number(item.t) >= progress);
  const left = keyframes[Math.max(0, rightIndex - 1)];
  const right = keyframes[rightIndex];
  const span = Math.max(Number(right.t) - Number(left.t), 0.0001);
  const local = easingValue(
    (progress - Number(left.t)) / span,
    text(right.easing || left.easing || 'linear'),
  );
  return interpolate(local, [0, 1], [Number(left.value), Number(right.value)], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
};

const anchorAdjusted = (layout) => {
  const width = clamp(number(layout?.width, 0.1), 0.001, 1);
  const height = clamp(number(layout?.height, 0.1), 0.001, 1);
  let x = number(layout?.x, 0);
  let y = number(layout?.y, 0);
  const anchor = text(layout?.anchor) || 'top_left';
  if (anchor === 'center') {
    x -= width / 2;
    y -= height / 2;
  } else if (anchor === 'top_right') {
    x -= width;
  } else if (anchor === 'bottom_left') {
    y -= height;
  } else if (anchor === 'bottom_right') {
    x -= width;
    y -= height;
  }
  return {
    x,
    y,
    width,
    height,
    anchor: 'top_left',
    z_index: clamp(Math.round(number(layout?.z_index, 4)), 0, 100),
  };
};

const clampRectToSafeArea = (rect, safeArea) => {
  const left = clamp(number(safeArea?.left, 0.04), 0, 0.2);
  const right = 1 - clamp(number(safeArea?.right, 0.04), 0, 0.2);
  const top = clamp(number(safeArea?.top, 0.04), 0, 0.2);
  const bottom = 1 - clamp(number(safeArea?.bottom, 0.04), 0, 0.2);
  const width = Math.min(rect.width, Math.max(right - left, 0.001));
  const height = Math.min(rect.height, Math.max(bottom - top, 0.001));
  return {
    ...rect,
    width,
    height,
    x: clamp(rect.x, left, Math.max(left, right - width)),
    y: clamp(rect.y, top, Math.max(top, bottom - height)),
  };
};

const overlap = (a, b, gap = 0) => ({
  x: Math.min(a.x + a.width + gap, b.x + b.width + gap) - Math.max(a.x - gap, b.x - gap),
  y: Math.min(a.y + a.height + gap, b.y + b.height + gap) - Math.max(a.y - gap, b.y - gap),
});

const solveSeparation = (rects, targetIds, axis, gap, safeArea) => {
  const ids = targetIds.filter((target) => rects.has(target));
  for (let pass = 0; pass < 5; pass += 1) {
    let changed = false;
    for (let leftIndex = 0; leftIndex < ids.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < ids.length; rightIndex += 1) {
        const leftId = ids[leftIndex];
        const rightId = ids[rightIndex];
        const left = rects.get(leftId);
        const right = rects.get(rightId);
        const intersection = overlap(left, right, gap / 2);
        if (intersection.x <= 0 || intersection.y <= 0) continue;
        const useX = axis === 'x' || (axis === 'both' && intersection.x <= intersection.y);
        const shift = ((useX ? intersection.x : intersection.y) + gap) / 2;
        const leftNext = {...left};
        const rightNext = {...right};
        if (useX) {
          leftNext.x -= shift;
          rightNext.x += shift;
        } else {
          leftNext.y -= shift;
          rightNext.y += shift;
        }
        rects.set(leftId, clampRectToSafeArea(leftNext, safeArea));
        rects.set(rightId, clampRectToSafeArea(rightNext, safeArea));
        changed = true;
      }
    }
    if (!changed) break;
  }
};

const distributionGroups = (rects, targets, axis) => {
  const useX = axis !== 'y';
  const ordered = targets.slice().sort((leftId, rightId) => {
    const left = rects.get(leftId);
    const right = rects.get(rightId);
    const leftOrthogonal = useX ? left.y + left.height / 2 : left.x + left.width / 2;
    const rightOrthogonal = useX ? right.y + right.height / 2 : right.x + right.width / 2;
    return leftOrthogonal - rightOrthogonal;
  });
  const groups = [];
  ordered.forEach((target) => {
    const rect = rects.get(target);
    const orthogonalCenter = useX ? rect.y + rect.height / 2 : rect.x + rect.width / 2;
    const orthogonalSize = useX ? rect.height : rect.width;
    const current = groups[groups.length - 1];
    if (
      !current
      || Math.abs(orthogonalCenter - current.center)
        > Math.max(0.04, Math.min(orthogonalSize, current.size) * 0.45)
    ) {
      groups.push({center: orthogonalCenter, size: orthogonalSize, targets: [target]});
      return;
    }
    current.targets.push(target);
    current.center = (
      current.center * (current.targets.length - 1) + orthogonalCenter
    ) / current.targets.length;
    current.size = Math.max(current.size, orthogonalSize);
  });
  return groups.map((group) => group.targets);
};

export const solveSceneGraphLayout = (graph) => {
  const nodes = list(graph?.nodes);
  const safeArea = graph?.canvas?.safe_area || {};
  const rects = new Map(nodes.map((node) => [
    text(node.node_id),
    anchorAdjusted(node.layout || {}),
  ]));
  const constraints = list(graph?.constraints)
    .slice()
    .sort((a, b) => number(b.priority, 100) - number(a.priority, 100));

  constraints.forEach((constraint) => {
    const targets = list(constraint.targets).map(text).filter((target) => rects.has(target));
    const type = text(constraint.type);
    const axis = ['x', 'y'].includes(text(constraint.axis)) ? text(constraint.axis) : 'both';
    const gap = clamp(number(constraint.gap, 0.02), 0, 0.5);
    if (!targets.length) return;

    if (type === 'keep_inside_safe_area') {
      targets.forEach((target) => rects.set(target, clampRectToSafeArea(rects.get(target), safeArea)));
      return;
    }

    if (type === 'align' && targets.length >= 2) {
      const centerX = targets.reduce((sum, target) => sum + rects.get(target).x + rects.get(target).width / 2, 0) / targets.length;
      const centerY = targets.reduce((sum, target) => sum + rects.get(target).y + rects.get(target).height / 2, 0) / targets.length;
      targets.forEach((target) => {
        const rect = {...rects.get(target)};
        if (axis !== 'y') rect.x = centerX - rect.width / 2;
        if (axis !== 'x') rect.y = centerY - rect.height / 2;
        rects.set(target, clampRectToSafeArea(rect, safeArea));
      });
      return;
    }

    if (type === 'distribute' && targets.length >= 3) {
      const useX = axis !== 'y';
      distributionGroups(rects, targets, axis).forEach((group) => {
        if (group.length < 3) return;
        const sorted = group.slice().sort((left, right) => {
          const a = rects.get(left);
          const b = rects.get(right);
          return useX
            ? (a.x + a.width / 2) - (b.x + b.width / 2)
            : (a.y + a.height / 2) - (b.y + b.height / 2);
        });
        const first = rects.get(sorted[0]);
        const last = rects.get(sorted[sorted.length - 1]);
        const start = useX ? first.x + first.width / 2 : first.y + first.height / 2;
        const end = useX ? last.x + last.width / 2 : last.y + last.height / 2;
        sorted.forEach((target, index) => {
          const rect = {...rects.get(target)};
          const center = start + ((end - start) * index) / Math.max(sorted.length - 1, 1);
          if (useX) rect.x = center - rect.width / 2;
          else rect.y = center - rect.height / 2;
          rects.set(target, clampRectToSafeArea(rect, safeArea));
        });
      });
      return;
    }

    if (['avoid_overlap', 'minimum_gap'].includes(type)) {
      solveSeparation(rects, targets, axis, gap, safeArea);
    }
  });

  nodes.forEach((node) => {
    const nodeId = text(node.node_id);
    rects.set(nodeId, clampRectToSafeArea(rects.get(nodeId), safeArea));
  });
  return rects;
};

const fontSizeFor = (node, rect, base, typography) => {
  const content = text(node?.content?.text);
  const style = node?.style || {};
  const minimum = text(node.role) === 'title'
    ? Math.max(30, number(typography?.minimum_font_px, 18) * 1.7)
    : content && !node.decorative
      ? Math.max(18, number(typography?.minimum_font_px, 18))
      : 12;
  const requested = clamp(number(style.font_size, text(node.role) === 'title' ? 68 : 30), minimum, 128);
  if (!content) return requested;
  const withinWidth = Math.max(40, rect.width * base.width - Math.max(24, base.width * 0.024));
  const result = fitText({
    text: content,
    withinWidth,
    fontFamily: text(typography?.font_family) || 'Arial, sans-serif',
    fontWeight: String(clamp(number(style.font_weight, 750), 300, 950)),
  });
  return clamp(result.fontSize, minimum, requested);
};

const motionStyleFor = (node, tracks, progress, palette, base) => {
  const style = node.style || {};
  const translateX = trackValue(tracks, 'translate_x', progress, 0) * base.width;
  const translateY = trackValue(tracks, 'translate_y', progress, 0) * base.height;
  const translateZ = trackValue(tracks, 'translate_z', progress, 0) * Math.min(base.width, base.height);
  const scale = trackValue(tracks, 'scale', progress, 1);
  const rotation = trackValue(tracks, 'rotation', progress, 0);
  const rotationX = trackValue(tracks, 'rotation_x', progress, 0);
  const rotationY = trackValue(tracks, 'rotation_y', progress, 0);
  const skewX = trackValue(tracks, 'skew_x', progress, 0);
  const skewY = trackValue(tracks, 'skew_y', progress, 0);
  const opacity = clamp(trackValue(tracks, 'opacity', progress, number(style.opacity, 1)), 0, 1);
  const blur = Math.max(0, trackValue(tracks, 'blur', progress, number(style.blur, 0)));
  const emphasis = clamp(trackValue(tracks, 'emphasis', progress, 0), 0, 1);
  const clipProgress = clamp(trackValue(tracks, 'clip_progress', progress, 1), 0, 1);
  return {
    opacity,
    filter: blur > 0 ? `blur(${blur}px)` : undefined,
    transform: `perspective(${Math.max(300, base.width * 0.7)}px) translate3d(${translateX}px, ${translateY}px, ${translateZ}px) rotateX(${rotationX}deg) rotateY(${rotationY}deg) rotate(${rotation}deg) skew(${skewX}deg, ${skewY}deg) scale(${scale * (1 + emphasis * 0.045)})`,
    transformOrigin: 'center',
    clipPath: clipProgress < 0.999 ? `inset(0 ${(1 - clipProgress) * 100}% 0 0)` : undefined,
    boxShadow: emphasis > 0
      ? `0 0 ${16 + emphasis * 36}px ${colorFor(style.stroke || 'accent', palette)}55`
      : undefined,
  };
};

const positionedStyle = (node, rect, tracks, progress, palette, base, typography) => {
  const style = node.style || {};
  return {
    boxSizing: 'border-box',
    position: 'absolute',
    left: rect.x * base.width,
    top: rect.y * base.height,
    width: Math.max(2, rect.width * base.width),
    height: Math.max(2, rect.height * base.height),
    zIndex: rect.z_index,
    color: colorFor(style.color || (style.fill === 'text' ? 'text' : 'ink'), palette),
    fontFamily: text(typography?.font_family) || 'Arial, sans-serif',
    fontSize: fontSizeFor(node, rect, base, typography),
    fontWeight: clamp(number(style.font_weight, text(node.role) === 'title' ? typography?.title_weight : typography?.body_weight), 300, 950),
    letterSpacing: `${number(typography?.letter_spacing_em, -0.02)}em`,
    lineHeight: number(typography?.line_height, 1.12),
    overflowWrap: 'break-word',
    ...motionStyleFor(node, tracks, progress, palette, base),
  };
};

const telemetryProps = (node, rect) => ({
  'data-vex-sg-node': text(node.node_id),
  'data-vex-primitive': text(node.primitive),
  'data-vex-requested-backend': text(node.backend),
  'data-vex-runtime-backend': ['dom', 'svg'].includes(text(node.backend)) ? text(node.backend) : text(node.fallback_backend),
  'data-vex-binding': text(node.binding?.id) || undefined,
  'data-vex-required-label': text(node.content?.text) || undefined,
  'data-vex-layout': [rect.x, rect.y, rect.width, rect.height].map((value) => value.toFixed(5)).join(','),
  'data-vex-measure-text': node.telemetry?.measure_text ? 'true' : 'false',
});

const NodeFrame = ({node, rect, tracks, progress, palette, base, typography, children, overflow = 'hidden'}) => {
  const style = node.style || {};
  const strokeWidth = Math.max(1, number(style.stroke_width, 2));
  const borderColor = colorFor(style.stroke || 'accent', palette);
  const fill = colorFor(style.fill || 'surface', palette);
  const radius = Math.max(0, number(style.radius, text(node.primitive) === 'semantic_token' ? 9 : 14));
  return <div
    {...telemetryProps(node, rect)}
    style={{
      ...positionedStyle(node, rect, tracks, progress, palette, base, typography),
      display: 'grid',
      placeItems: 'center',
      padding: text(node.content?.text) ? Math.max(10, Math.round(base.width * 0.012)) : 0,
      textAlign: 'center',
      backgroundColor: fill,
      border: `${strokeWidth}px solid ${borderColor}`,
      borderRadius: radius,
      overflow,
      isolation: 'isolate',
    }}
  >
    {children}
  </div>;
};

const GraphNode = (props) => {
  const {node, progress, tracks, palette} = props;
  const reveal = clamp(trackValue(tracks, 'progress', progress, 1), 0, 1);
  const label = text(node.content?.text);
  const role = text(node.role);
  return <NodeFrame {...props}>
    <svg aria-hidden="true" viewBox="0 0 100 100" preserveAspectRatio="none" style={{position: 'absolute', inset: 0, width: '100%', height: '100%', zIndex: -1}}>
      <defs>
        <linearGradient id={`node-gradient-${node.node_id}`} x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor={colorFor(node.style?.stroke || 'accent', palette)} stopOpacity="0.18" />
          <stop offset="1" stopColor={colorFor('accent_secondary', palette)} stopOpacity="0.03" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width={`${reveal * 100}`} height="100" fill={`url(#node-gradient-${node.node_id})`} />
      <path d="M0 92 H100" stroke={colorFor(node.style?.stroke || 'accent', palette)} strokeWidth="3" pathLength="1" strokeDasharray="1" strokeDashoffset={1 - reveal} />
    </svg>
    {role === 'resolved_outcome' && !label
      ? <div style={{width: '34%', aspectRatio: 1, maxWidth: 84, borderRadius: '50%', border: `5px solid ${palette.accent2}`, display: 'grid', placeItems: 'center'}}>
        <i style={{width: '34%', aspectRatio: 1, borderRadius: '50%', backgroundColor: palette.accent}} />
      </div>
      : <strong style={{position: 'relative', maxWidth: '94%', fontWeight: 'inherit'}}>{label}</strong>}
  </NodeFrame>;
};

const KineticText = ({node, rect, tracks, progress, palette, base, typography}) => {
  const reveal = clamp(trackValue(tracks, 'clip_progress', progress, trackValue(tracks, 'opacity', progress, 1)), 0, 1);
  return <div
    {...telemetryProps(node, rect)}
    style={{
      ...positionedStyle(node, rect, tracks, progress, palette, base, typography),
      display: 'flex',
      alignItems: 'center',
      overflow: 'visible',
      color: colorFor(node.style?.fill === 'text' ? 'text' : node.style?.color || 'text', palette),
      textAlign: 'left',
    }}
  >
    <div style={{position: 'relative', maxWidth: '100%'}}>
      <strong style={{fontWeight: 'inherit'}}>{text(node.content?.text)}</strong>
      <i aria-hidden="true" style={{position: 'absolute', left: 0, bottom: -8, width: `${reveal * 42}%`, height: 5, background: `linear-gradient(90deg, ${palette.accent}, ${palette.accent2})`}} />
    </div>
  </div>;
};

const TextBlock = ({node, rect, tracks, progress, palette, base, typography}) => <div
  {...telemetryProps(node, rect)}
  style={{
    ...positionedStyle(node, rect, tracks, progress, palette, base, typography),
    display: 'flex',
    alignItems: 'center',
    color: colorFor(node.style?.fill === 'text' ? 'text' : node.style?.color || 'text', palette),
    textAlign: text(node.style?.text_align) || 'left',
  }}
>
  {text(node.content?.text)}
</div>;

const MetricMark = (props) => {
  const {node, palette, progress, tracks} = props;
  const reveal = clamp(trackValue(tracks, 'progress', progress, 1), 0, 1);
  return <NodeFrame {...props}>
    <div style={{position: 'absolute', inset: 0, background: `radial-gradient(circle at 70% 20%, ${palette.accent2}2A, transparent 46%)`}} />
    <strong style={{position: 'relative', fontSize: '1.2em', fontWeight: 900, color: palette.ink}}>{text(node.content?.text)}</strong>
    <div aria-hidden="true" style={{position: 'absolute', left: '8%', right: '8%', bottom: '10%', height: 5, backgroundColor: `${palette.accent}22`}}>
      <i style={{display: 'block', width: `${reveal * 100}%`, height: '100%', backgroundColor: palette.accent}} />
    </div>
  </NodeFrame>;
};

const normalizedChartData = (value) => list(value)
  .map((item, index) => {
    if (typeof item === 'number') return {label: String(index + 1), value: item};
    if (!item || typeof item !== 'object') return null;
    const numeric = Number(item.value);
    return Number.isFinite(numeric) ? {label: text(item.label) || String(index + 1), value: numeric} : null;
  })
  .filter(Boolean)
  .slice(0, 12);

const DataChart = (props) => {
  const {node, rect, tracks, progress, palette, base, typography} = props;
  const data = normalizedChartData(node.content?.data);
  const reveal = clamp(trackValue(tracks, 'progress', progress, 1), 0, 1);
  const maximum = Math.max(...data.map((item) => Math.abs(item.value)), 1);
  const style = positionedStyle(node, rect, tracks, progress, palette, base, typography);
  return <div {...telemetryProps(node, rect)} data-vex-chart-values={data.map((item) => item.value).join(',')} style={{...style, overflow: 'hidden'}}>
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{position: 'absolute', inset: 0, width: '100%', height: '100%'}}>
      <line x1="8" x2="96" y1="88" y2="88" stroke={palette.muted} strokeOpacity="0.5" strokeWidth="0.8" />
      <line x1="8" x2="8" y1="8" y2="88" stroke={palette.muted} strokeOpacity="0.5" strokeWidth="0.8" />
      {data.map((item, index) => {
        const slot = 84 / Math.max(data.length, 1);
        const barWidth = slot * 0.64;
        const height = (Math.abs(item.value) / maximum) * 68 * reveal;
        return <rect
          key={`${item.label}-${index}`}
          x={10 + index * slot}
          y={88 - height}
          width={barWidth}
          height={height}
          rx="1"
          fill={index % 2 ? palette.accent2 : palette.accent}
        />;
      })}
      {!data.length ? <path d="M12 76 C32 72 38 48 54 54 S78 28 92 22" fill="none" stroke={palette.accent2} strokeOpacity="0.45" strokeWidth="2" pathLength="1" strokeDasharray="0.08 0.06" strokeDashoffset={1 - reveal} /> : null}
    </svg>
    <strong style={{position: 'absolute', left: '10%', top: '8%', right: '8%', color: palette.text, textAlign: 'left'}}>
      {text(node.content?.text)}
    </strong>
    {!data.length ? <span style={{position: 'absolute', left: '10%', bottom: '8%', color: palette.muted, fontSize: '0.45em', fontWeight: 700}}>Evidence mark · no invented scale</span> : null}
  </div>;
};

const VectorIcon = ({node, rect, tracks, progress, palette, base, typography}) => {
  const style = positionedStyle(node, rect, tracks, progress, palette, base, typography);
  const stroke = colorFor(node.style?.stroke || 'accent', palette);
  const pathProgress = clamp(trackValue(tracks, 'stroke_progress', progress, trackValue(tracks, 'progress', progress, 1)), 0, 1);
  return <div {...telemetryProps(node, rect)} style={{...style, display: 'grid', placeItems: 'center'}}>
    <svg viewBox="0 0 100 100" style={{width: '82%', height: '82%', overflow: 'visible'}}>
      <circle cx="50" cy="50" r="35" fill="none" stroke={stroke} strokeWidth="6" pathLength="1" strokeDasharray="1" strokeDashoffset={1 - pathProgress} />
      <path d="M34 51 L46 63 L69 36" fill="none" stroke={palette.text} strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" pathLength="1" strokeDasharray="1" strokeDashoffset={1 - pathProgress} />
    </svg>
  </div>;
};

const VectorShape = (props) => {
  const {node, rect, tracks, progress, palette, base, typography} = props;
  const style = positionedStyle(node, rect, tracks, progress, palette, base, typography);
  const pathProgress = clamp(trackValue(tracks, 'stroke_progress', progress, trackValue(tracks, 'progress', progress, 1)), 0, 1);
  return <div {...telemetryProps(node, rect)} style={style}>
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{width: '100%', height: '100%'}}>
      <rect x="3" y="3" width="94" height="94" rx={clamp(number(node.style?.radius, 4), 0, 30)} fill={colorFor(node.style?.fill || 'surface', palette)} fillOpacity={number(node.style?.fill_opacity, 0.16)} stroke={colorFor(node.style?.stroke || 'accent', palette)} strokeWidth={Math.max(1, number(node.style?.stroke_width, 2))} pathLength="1" strokeDasharray="1" strokeDashoffset={1 - pathProgress} />
    </svg>
  </div>;
};

const VectorPathNode = ({node, rect, tracks, progress, palette, base, typography}) => {
  const style = positionedStyle(node, rect, tracks, progress, palette, base, typography);
  const pathProgress = clamp(trackValue(tracks, 'path_progress', progress, trackValue(tracks, 'progress', progress, 1)), 0, 1);
  return <div {...telemetryProps(node, rect)} style={style}>
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{width: '100%', height: '100%', overflow: 'visible'}}>
      <path d="M2 52 C28 10 66 90 98 48" fill="none" stroke={colorFor(node.style?.stroke || 'accent', palette)} strokeWidth={Math.max(2, number(node.style?.stroke_width, 4))} strokeLinecap="round" pathLength="1" strokeDasharray="1" strokeDashoffset={1 - pathProgress} />
    </svg>
  </div>;
};

const MaskedMedia = ({node, rect, tracks, progress, palette, base, typography}) => {
  const asset = node.content?.asset || {};
  const dataUri = text(asset.data_uri);
  const style = positionedStyle(node, rect, tracks, progress, palette, base, typography);
  const maskProgress = clamp(trackValue(tracks, 'mask_progress', progress, 1), 0, 1);
  if (dataUri.startsWith('data:image/')) {
    return <img
      {...telemetryProps(node, rect)}
      src={dataUri}
      alt={text(node.content?.text)}
      style={{
        ...style,
        objectFit: text(asset.fit) || 'cover',
        clipPath: `inset(0 ${(1 - maskProgress) * 100}% 0 0 round ${Math.max(0, number(node.style?.radius, 12))}px)`,
      }}
    />;
  }
  return <div {...telemetryProps(node, rect)} style={{...style, display: 'grid', placeItems: 'center', border: `2px solid ${palette.accent2}`, background: `repeating-linear-gradient(135deg, ${palette.accent2}18 0 8px, transparent 8px 16px)`, color: palette.text}}>
    <strong>{text(node.content?.text)}</strong>
  </div>;
};

const ParticleField = ({node, rect, tracks, progress, palette, base, typography}) => {
  const style = positionedStyle(node, rect, tracks, progress, palette, base, typography);
  const repeat = clamp(Math.round(number(node.repeat, 12)), 1, 24);
  const fieldProgress = clamp(trackValue(tracks, 'progress', progress, 1), 0, 1);
  return <div {...telemetryProps(node, rect)} style={{...style, overflow: 'visible'}}>
    {Array.from({length: repeat}).map((_, index) => {
      const phase = ((index * 47) % 100) / 100;
      return <i key={index} style={{
        position: 'absolute',
        left: `${(index * 37) % 96}%`,
        top: `${(index * 53) % 92}%`,
        width: 4 + index % 5,
        height: 4 + index % 5,
        borderRadius: '50%',
        backgroundColor: index % 2 ? palette.accent : palette.accent2,
        opacity: 0.18 + fieldProgress * 0.62,
        transform: `translate3d(${Math.sin(phase * Math.PI * 2 + progress * 4) * 12}px, ${(1 - fieldProgress) * (22 + index % 6 * 7)}px, 0)`,
      }} />;
    })}
  </div>;
};

const GroupNode = (props) => <NodeFrame {...props} overflow="visible">
  {text(props.node.content?.text)}
</NodeFrame>;

const NODE_RENDERERS = {
  data_chart: DataChart,
  graph_node: GraphNode,
  group: GroupNode,
  kinetic_text_run: KineticText,
  mask_group: VectorShape,
  masked_media: MaskedMedia,
  metric_mark: MetricMark,
  particle_field: ParticleField,
  semantic_token: GraphNode,
  text_block: TextBlock,
  vector_icon: VectorIcon,
  vector_path: VectorPathNode,
  vector_shape: VectorShape,
};

const boundaryPoint = (source, target, inset = 0) => {
  const sourceCenterX = source.x + source.width / 2;
  const sourceCenterY = source.y + source.height / 2;
  const targetCenterX = target.x + target.width / 2;
  const targetCenterY = target.y + target.height / 2;
  const dx = targetCenterX - sourceCenterX;
  const dy = targetCenterY - sourceCenterY;
  const length = Math.max(Math.hypot(dx, dy), 0.0001);
  const distance = 0.5 / Math.max(Math.abs(dx) / Math.max(source.width, 0.001), Math.abs(dy) / Math.max(source.height, 0.001), 0.0001);
  return {
    x: sourceCenterX + dx * distance + (dx / length) * inset,
    y: sourceCenterY + dy * distance + (dy / length) * inset,
  };
};

const relationGeometry = (relation, rects, base) => {
  const source = rects.get(text(relation.source_id));
  const target = rects.get(text(relation.target_id));
  if (!source || !target) return null;
  const start = boundaryPoint(source, target, 0.006);
  const end = boundaryPoint(target, source, 0.012);
  const x1 = start.x * base.width;
  const y1 = start.y * base.height;
  const x2 = end.x * base.width;
  const y2 = end.y * base.height;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const labelClearance = Math.hypot(dx, dy);
  const routing = text(relation.routing);
  if (routing === 'direct') {
    return {path: `M ${x1} ${y1} L ${x2} ${y2}`, labelX: (x1 + x2) / 2, labelY: (y1 + y2) / 2, labelClearance};
  }
  if (routing === 'orthogonal') {
    const horizontalFirst = Math.abs(dx) >= Math.abs(dy);
    if (horizontalFirst) {
      const middleX = x1 + dx / 2;
      return {
        path: `M ${x1} ${y1} C ${middleX} ${y1}, ${middleX} ${y1}, ${middleX} ${(y1 + y2) / 2} C ${middleX} ${y2}, ${middleX} ${y2}, ${x2} ${y2}`,
        labelX: middleX,
        labelY: (y1 + y2) / 2,
        labelClearance,
      };
    }
    const middleY = y1 + dy / 2;
    return {
      path: `M ${x1} ${y1} C ${x1} ${middleY}, ${x1} ${middleY}, ${(x1 + x2) / 2} ${middleY} C ${x2} ${middleY}, ${x2} ${middleY}, ${x2} ${y2}`,
      labelX: (x1 + x2) / 2,
      labelY: middleY,
      labelClearance,
    };
  }
  const control = Math.max(40, Math.min(Math.abs(dx) * 0.46, 180));
  return {
    path: `M ${x1} ${y1} C ${x1 + Math.sign(dx || 1) * control} ${y1}, ${x2 - Math.sign(dx || 1) * control} ${y2}, ${x2} ${y2}`,
    labelX: (x1 + x2) / 2,
    labelY: (y1 + y2) / 2 - Math.min(24, Math.abs(dy) * 0.08),
    labelClearance,
  };
};

const RoutedRelations = ({graph, rects, tracks, progress, palette, base}) => {
  const relations = list(graph.relations).map((relation, index) => {
    const geometry = relationGeometry(relation, rects, base);
    if (!geometry) return null;
    const relationTracks = tracks.filter((track) => text(track.target_id) === text(relation.relation_id));
    const hasProgress = relationTracks.some((track) => ['progress', 'path_progress', 'stroke_progress'].includes(text(track.property)));
    const visible = hasProgress
      ? clamp(
        trackValue(
          relationTracks,
          'path_progress',
          progress,
          trackValue(relationTracks, 'stroke_progress', progress, trackValue(relationTracks, 'progress', progress, 0)),
        ),
        0,
        1,
      )
      : clamp((progress - (0.22 + index * 0.07)) / 0.2, 0, 1);
    return {relation, geometry, visible, index};
  }).filter(Boolean);
  return <svg
    data-vex-relation-layer="true"
    viewBox={`0 0 ${base.width} ${base.height}`}
    style={{position: 'absolute', inset: 0, width: base.width, height: base.height, overflow: 'visible', pointerEvents: 'none', zIndex: 3}}
  >
    <defs>
      {relations.map(({relation, index}) => {
        const color = colorFor(relation.style?.stroke || (index % 2 ? 'accent_secondary' : 'accent'), palette);
        return <marker key={relation.relation_id} id={`arrow-${relation.relation_id}`} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L8,4 L0,8 z" fill={color} />
        </marker>;
      })}
    </defs>
    {relations.map(({relation, geometry, visible, index}) => {
      const color = colorFor(relation.style?.stroke || (index % 2 ? 'accent_secondary' : 'accent'), palette);
      const strokeWidth = Math.max(2, number(relation.style?.stroke_width, 4));
      const edgeId = text(relation.binding?.id) || text(relation.relation_id);
      return <g key={relation.relation_id} data-vex-required-edge={edgeId} data-vex-relation-path={geometry.path} data-vex-route={text(relation.routing)}>
        <path d={geometry.path} fill="none" stroke={`${color}25`} strokeWidth={strokeWidth} strokeLinecap="round" />
        <path
          d={geometry.path}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          pathLength="1"
          strokeDasharray="1"
          strokeDashoffset={1 - visible}
          markerEnd={visible > 0.96 ? `url(#arrow-${relation.relation_id})` : undefined}
        />
        {text(relation.type) && geometry.labelClearance >= 110 ? <text x={geometry.labelX} y={geometry.labelY - 8} fill={palette.muted} stroke={palette.bg} strokeWidth="5" paintOrder="stroke" fontSize="13" fontWeight="800" textAnchor="middle" opacity={visible}>
          {text(relation.type).replaceAll('_', ' ').toUpperCase()}
        </text> : null}
      </g>;
    })}
  </svg>;
};

export const SceneGraphLayer = ({graph, palette, base, frame, durationInFrames}) => {
  const progress = clamp(frame / Math.max(durationInFrames - 1, 1), 0, 1);
  const rects = solveSceneGraphLayout(graph);
  const tracks = list(graph?.motion_graph?.tracks);
  const typography = graph?.design_system?.typography || {};
  const nodes = list(graph?.nodes).slice().sort((left, right) => number(left.layout?.z_index, 4) - number(right.layout?.z_index, 4));
  const activePhases = list(graph?.motion_graph?.phases)
    .filter((phase) => progress >= number(phase.start, 0) && progress <= number(phase.end, 1))
    .map((phase) => text(phase.phase_id))
    .join(',');
  return <div
    data-vex-scene-graph={text(graph?.scene_graph_id)}
    data-vex-scene-graph-version={text(graph?.version)}
    data-vex-scene-graph-signature={text(graph?.signature)}
    data-vex-source-program-signature={text(graph?.source_program_signature)}
    data-vex-layout-solved="true"
    data-vex-active-phases={activePhases}
    style={{position: 'absolute', inset: 0}}
  >
    <RoutedRelations graph={graph} rects={rects} tracks={tracks} progress={progress} palette={palette} base={base} />
    {nodes.map((node) => {
      const NodeRenderer = NODE_RENDERERS[text(node.primitive)] || VectorShape;
      const nodeTracks = tracks.filter((track) => text(track.target_id) === text(node.node_id));
      return <NodeRenderer
        key={node.node_id}
        node={node}
        rect={rects.get(text(node.node_id))}
        tracks={nodeTracks}
        progress={progress}
        palette={palette}
        base={base}
        typography={typography}
      />;
    })}
  </div>;
};
