import React from 'react';
import {fitText} from '@remotion/layout-utils';
import {
  AbsoluteFill,
  Composition,
  interpolate,
  registerRoot,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

const PALETTES = {
  editorial_clean: {bg: '#101418', surface: '#F5F1E8', surfaceDark: '#1C252D', text: '#F8FAFC', muted: '#AEB9C5', ink: '#111827', accent: '#E11D48', accent2: '#0891B2', accent3: '#F59E0B'},
  bold_tech: {bg: '#090B10', surface: '#EAF8FF', surfaceDark: '#131B25', text: '#F8FAFC', muted: '#B4C0CC', ink: '#07111F', accent: '#22C55E', accent2: '#38BDF8', accent3: '#F97316'},
  documentary_kinetic: {bg: '#15120E', surface: '#F3EFE3', surfaceDark: '#29231A', text: '#FFF7ED', muted: '#D5C7B1', ink: '#1C1917', accent: '#DC2626', accent2: '#2563EB', accent3: '#D97706'},
  product_ui: {bg: '#0C1116', surface: '#F8FAFC', surfaceDark: '#17202A', text: '#F8FAFC', muted: '#B6C2CF', ink: '#0F172A', accent: '#2563EB', accent2: '#14B8A6', accent3: '#F43F5E'},
  cinematic_night: {bg: '#06080C', surface: '#E5E7EB', surfaceDark: '#111827', text: '#F9FAFB', muted: '#9CA3AF', ink: '#030712', accent: '#FBBF24', accent2: '#06B6D4', accent3: '#EF4444'},
  signal_lab: {bg: '#0A0F0D', surface: '#ECFDF5', surfaceDark: '#10231E', text: '#F0FDF4', muted: '#A7F3D0', ink: '#052E2B', accent: '#10B981', accent2: '#F59E0B', accent3: '#3B82F6'},
  magazine_luxe: {bg: '#14110F', surface: '#F7F0E3', surfaceDark: '#27211C', text: '#FEF3C7', muted: '#D6C2A4', ink: '#1C1917', accent: '#B91C1C', accent2: '#0F766E', accent3: '#A16207'},
};

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const text = (value) => String(value || '').trim();
const list = (value) => (Array.isArray(value) ? value : []);

const paletteFor = (program) => {
  const base = PALETTES[text(program.style_pack)] || PALETTES.editorial_clean;
  const directed = program.creative_direction?.art_direction?.palette || {};
  const theme = program.theme || {};
  const open = program.open_visual_program?.palette || {};
  return {
    ...base,
    bg: text(open.background) || text(directed.background) || text(theme.background) || base.bg,
    surface: text(open.surface) || text(directed.panel_fill) || text(theme.panel_fill) || base.surface,
    surfaceDark: text(directed.panel_fill) || text(theme.panel_dark) || base.surfaceDark,
    text: text(open.ink) || text(directed.text_primary) || text(theme.text_primary) || base.text,
    muted: text(open.muted) || text(directed.text_secondary) || text(theme.text_secondary) || base.muted,
    accent: text(open.accent) || text(directed.accent) || text(theme.accent) || base.accent,
    accent2: text(open.accent_secondary) || text(directed.accent_secondary) || text(theme.accent_secondary) || base.accent2,
    accent3: text(directed.glow) || base.accent3,
    ink: text(open.ink) || text(directed.ink) || base.ink,
  };
};

const dimensionsFor = (orientation) => {
  if (orientation === 'portrait') return {width: 720, height: 1280};
  if (orientation === 'square') return {width: 900, height: 900};
  return {width: 1280, height: 720};
};

const measuredSize = (value, withinWidth, maxSize, minSize = 20, weight = 800) => {
  if (!text(value)) return minSize;
  const result = fitText({
    text: text(value),
    withinWidth,
    fontFamily: 'Arial',
    fontWeight: String(weight),
  });
  return clamp(result.fontSize, minSize, maxSize);
};

const beatStart = (program, nodeId, fallback) => {
  const match = list(program.beats).find((beat) => list(beat.target_ids).includes(nodeId));
  return match ? Number(match.start_fraction || fallback) : fallback;
};

const revealFor = (program, nodeId, index, frame, fps, durationInFrames) => {
  const fraction = clamp(beatStart(program, nodeId, 0.12 + index * 0.1), 0.04, 0.72);
  const startFrame = Math.round(durationInFrames * fraction);
  return spring({
    frame: frame - startFrame,
    fps,
    durationInFrames: Math.max(8, Math.round(fps * 0.38)),
    config: {damping: 24, stiffness: 130, mass: 0.8, overshootClamping: true},
  });
};

const relationProgressFor = (program, frame, fps, durationInFrames) => {
  const phases = list(program.creative_direction?.choreography?.phases);
  const relationPhase = phases.find((phase) => ['relate', 'connect', 'explain'].includes(text(phase.phase).toLowerCase()));
  const startFraction = clamp(Number(relationPhase?.start ?? 0.32), 0.12, 0.72);
  return spring({
    frame: frame - Math.round(durationInFrames * startFraction),
    fps,
    durationInFrames: Math.max(10, Math.round(fps * 0.46)),
    config: {damping: 26, stiffness: 118, mass: 0.82, overshootClamping: true},
  });
};

const relationLabel = (edge) => {
  const relation = text(edge?.relation).replaceAll('_', ' ').toLowerCase();
  const labels = {precedes: 'then', sequence: 'then', transforms: 'becomes', 'transforms to': 'becomes', causes: 'drives', routes: 'routes to'};
  return labels[relation] || relation || 'connects';
};

const DirectionBackdrop = ({program, palette, base, frame, durationInFrames}) => {
  const direction = program.creative_direction || {};
  const medium = text(direction.medium_family);
  const progress = clamp(frame / Math.max(durationInFrames * 0.68, 1), 0, 1);
  const common = {position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none'};
  if (medium === 'data_sculpture') return <div style={common} aria-hidden="true">
    {[0, 1, 2].map((index) => <div key={index} style={{position: 'absolute', right: 80 + index * 34, top: 104 + index * 38, width: 390 - index * 62, height: 390 - index * 62, border: `1px solid ${index % 2 ? palette.accent : palette.accent2}`, borderRadius: '50%', opacity: 0.14 + index * 0.05, transform: `rotate(${progress * (index % 2 ? -24 : 28)}deg)`}} />)}
    {Array.from({length: 18}).map((_, index) => <i key={index} style={{position: 'absolute', left: `${8 + (index * 29) % 86}%`, top: `${12 + (index * 47) % 76}%`, width: 4 + index % 4, height: 4 + index % 4, borderRadius: '50%', backgroundColor: index % 2 ? palette.accent : palette.accent2, opacity: 0.22}} />)}
  </div>;
  if (medium === 'editorial_collage') return <div style={common} aria-hidden="true">
    <div style={{position: 'absolute', right: -70, top: 90, width: 360, height: 110, backgroundColor: palette.accent, opacity: 0.12, transform: 'rotate(-7deg)'}} />
    <div style={{position: 'absolute', left: -80, bottom: 120, width: 330, height: 86, backgroundColor: palette.accent2, opacity: 0.1, transform: 'rotate(5deg)'}} />
    <div style={{position: 'absolute', left: base.width * 0.54, top: 0, width: 2, height: base.height, backgroundColor: palette.text, opacity: 0.08}} />
  </div>;
  if (medium === 'spatial_metaphor') return <div style={{...common, perspective: 900}} aria-hidden="true">
    <div style={{position: 'absolute', left: -120, right: -120, top: '54%', bottom: -260, borderTop: `2px solid ${palette.accent}`, backgroundImage: `repeating-linear-gradient(90deg, ${palette.accent2}22 0 2px, transparent 2px 88px)`, backgroundPositionX: `${progress * 74}px`, transform: 'rotateX(62deg)', transformOrigin: 'top center', opacity: 0.46}} />
  </div>;
  if (medium === 'kinetic_typography') return <div style={common} aria-hidden="true">
    <div style={{position: 'absolute', right: 34, top: 10, fontSize: 270, lineHeight: 1, fontWeight: 900, color: palette.accent, opacity: 0.055}}>01</div>
    <div style={{position: 'absolute', left: 48, bottom: 72, width: `${32 + progress * 34}%`, height: 12, backgroundColor: palette.accent2, opacity: 0.55}} />
  </div>;
  return null;
};

const Canvas = ({program, children}) => {
  const frame = useCurrentFrame();
  const {width, height, fps, durationInFrames} = useVideoConfig();
  const orientation = program.layout?.orientation || 'landscape';
  const base = dimensionsFor(orientation);
  const scale = Math.min(width / base.width, height / base.height);
  const palette = paletteFor(program);
  const entrance = spring({frame, fps, durationInFrames: Math.max(10, Math.round(fps * 0.45)), config: {damping: 28, stiffness: 110, overshootClamping: true}});
  const finalHold = frame >= durationInFrames * Number(program.quality_contract?.final_hold_start || 0.78);
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg, overflow: 'hidden'}} data-vex-program={program.program_id} data-vex-final-hold={finalHold ? 'true' : 'false'}>
      <div style={{position: 'absolute', inset: 0, opacity: 0.25, backgroundImage: `linear-gradient(${palette.accent2}22 1px, transparent 1px), linear-gradient(90deg, ${palette.accent2}18 1px, transparent 1px)`, backgroundSize: `${Math.max(36, Math.round(58 * scale))}px ${Math.max(36, Math.round(58 * scale))}px`}} />
      <div style={{position: 'absolute', left: '50%', top: '50%', width: base.width, height: base.height, transform: `translate(-50%, -50%) scale(${scale})`, transformOrigin: 'center', fontFamily: 'Arial, sans-serif', color: palette.text, opacity: entrance}}>
        <div style={{position: 'absolute', left: 0, top: 0, width: 12, height: base.height, backgroundColor: palette.accent}} />
        <DirectionBackdrop program={program} palette={palette} base={base} frame={frame} durationInFrames={durationInFrames} />
        {children({palette, base, frame, fps, durationInFrames, orientation})}
        <div style={{position: 'absolute', left: 48, right: 48, bottom: 30, height: 3, display: 'grid', gridTemplateColumns: '2fr 1fr 3fr'}}>
          <div style={{backgroundColor: palette.accent}} />
          <div style={{backgroundColor: palette.surface}} />
          <div style={{backgroundColor: palette.accent2}} />
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Header = ({program, palette, orientation, contentWidth}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const portrait = orientation === 'portrait';
  const square = orientation === 'square';
  const lines = list(program.layout?.title_lines).length ? list(program.layout.title_lines) : [program.title];
  const titleWidth = contentWidth || (portrait ? 620 : square ? 780 : 1050);
  const maxSize = Number(program.layout?.title_max_size || (portrait ? 62 : 78));
  const open = spring({frame: frame - 4, fps, durationInFrames: Math.round(fps * 0.42), config: {damping: 24, stiffness: 120, overshootClamping: true}});
  return (
    <div style={{width: titleWidth, transform: `translateY(${(1 - open) * 18}px)`, opacity: open}}>
      <div style={{display: 'inline-flex', alignItems: 'center', gap: 10, minHeight: 38, padding: '0 12px', outline: `1px solid ${palette.accent2}99`, backgroundColor: palette.surfaceDark, color: palette.text, borderRadius: 5, fontSize: portrait ? 18 : 19, fontWeight: 800, textTransform: 'uppercase'}}>
        <span style={{width: 9, height: 9, backgroundColor: palette.accent}} />
        {program.eyebrow}
      </div>
      <div style={{marginTop: portrait ? 28 : 24}}>
        {lines.map((line, index) => (
          <div key={`${line}-${index}`} style={{fontSize: measuredSize(line, titleWidth, maxSize, portrait ? 38 : 44, 900), lineHeight: 0.98, fontWeight: 900, color: palette.text}}>
            {line}
          </div>
        ))}
      </div>
      {program.takeaway ? (
        <div style={{marginTop: portrait ? 22 : 16, width: portrait ? titleWidth : Math.min(titleWidth, 900), fontSize: measuredSize(program.takeaway, portrait ? titleWidth : Math.min(titleWidth, 900), portrait ? 27 : 25, 18, 600), lineHeight: 1.24, fontWeight: 600, color: palette.muted}}>
          {program.takeaway}
        </div>
      ) : null}
    </div>
  );
};

const AnnotationRail = ({program, palette, orientation, top}) => {
  const annotations = list(program.annotations).slice(0, 3);
  if (!annotations.length) return null;
  const portrait = orientation === 'portrait';
  return <div style={{position: 'absolute', left: portrait ? 50 : 68, right: portrait ? 50 : 68, top, display: 'flex', gap: 12, flexWrap: 'wrap'}}>
    {annotations.map((annotation, index) => <div key={annotation} data-vex-required-label={annotation} style={{display: 'flex', alignItems: 'center', gap: 9, padding: '8px 11px', borderLeft: `4px solid ${index % 2 ? palette.accent2 : palette.accent}`, backgroundColor: `${palette.surfaceDark}CC`, color: palette.text, fontSize: portrait ? 17 : 16, fontWeight: 800}}><span style={{color: palette.accent2, fontSize: 11, textTransform: 'uppercase'}}>Constraint</span>{annotation}</div>)}
  </div>;
};

const NodeCard = ({program, node, index, palette, frame, fps, durationInFrames, width, compact = false, light = false}) => {
  const reveal = revealFor(program, node.node_id, index, frame, fps, durationInFrames);
  const foreground = light ? palette.ink : palette.text;
  const background = light ? palette.surface : palette.surfaceDark;
  const labelSize = measuredSize(node.label, width - 44, compact ? 28 : 34, 19, 850);
  const medium = text(program.creative_direction?.medium_family);
  const editorial = medium === 'editorial_collage';
  const spatial = medium === 'spatial_metaphor';
  return (
    <div data-vex-node-id={node.node_id} data-vex-required-label={node.label} style={{boxSizing: 'border-box', position: 'relative', width, minHeight: compact ? 132 : 174, padding: compact ? 20 : 24, backgroundColor: background, color: foreground, outline: `${editorial ? 3 : 2}px solid ${index % 2 ? palette.accent2 : palette.accent}`, borderRadius: editorial ? 0 : spatial ? 22 : 7, opacity: reveal, transform: `translateY(${(1 - reveal) * 24}px) scale(${0.975 + reveal * 0.025}) rotate(${editorial ? (index % 2 ? 1.2 : -1.2) : 0}deg)`, boxShadow: spatial ? `0 24px 70px ${palette.accent2}22` : `0 18px 50px ${palette.bg}88`, clipPath: editorial ? 'polygon(1% 3%, 98% 0, 100% 96%, 3% 100%)' : undefined, overflow: 'hidden'}}>
      <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12}}>
        <div style={{width: 38, height: 38, flex: '0 0 auto', display: 'grid', placeItems: 'center', backgroundColor: index % 2 ? palette.accent2 : palette.accent, color: '#fff', fontSize: 20, fontWeight: 900}}>{index + 1}</div>
        {node.value ? <div style={{fontSize: measuredSize(node.value, width * 0.48, compact ? 29 : 38, 20, 900), fontWeight: 900, color: light ? palette.accent2 : palette.accent3}}>{node.value}</div> : <div style={{fontSize: 13, fontWeight: 900, color: light ? palette.accent2 : palette.accent3, textTransform: 'uppercase'}}>{text(node.role).replaceAll('_', ' ')}</div>}
      </div>
      <div style={{marginTop: 18, fontSize: labelSize, lineHeight: 1.04, fontWeight: 850}}>{node.label}</div>
      {!compact && node.detail && node.detail !== node.label ? <div style={{marginTop: 12, fontSize: measuredSize(node.detail, width - 44, 19, 15, 600), lineHeight: 1.2, fontWeight: 600, color: light ? '#475569' : palette.muted}}>{node.detail}</div> : null}
      <div style={{position: 'absolute', left: 0, bottom: 0, width: '100%', height: 5, backgroundColor: `${index % 2 ? palette.accent2 : palette.accent}33`}}><div style={{width: `${reveal * 100}%`, height: '100%', backgroundColor: index % 2 ? palette.accent2 : palette.accent, transformOrigin: 'left center'}} /></div>
    </div>
  );
};

const RelationConnector = ({program, edge, palette, frame, fps, durationInFrames, portrait}) => {
  const progress = relationProgressFor(program, frame, fps, durationInFrames);
  const color = palette.accent2;
  if (portrait) return <div data-vex-required-edge={edge?.edge_id} style={{position: 'relative', width: '100%', height: 30, flex: '0 0 30px', display: 'grid', placeItems: 'center'}}>
    <div style={{position: 'absolute', left: '50%', top: 0, width: 4, height: 24, backgroundColor: `${color}35`, transform: 'translateX(-50%)'}}><div style={{width: '100%', height: `${progress * 100}%`, backgroundColor: color, transformOrigin: 'top center'}} /></div>
    <div style={{position: 'absolute', left: '50%', bottom: 2, width: 12, height: 12, borderRight: `4px solid ${color}`, borderBottom: `4px solid ${color}`, opacity: progress, transform: 'translateX(-50%) rotate(45deg)'}} />
    <div style={{position: 'absolute', left: 'calc(50% + 18px)', top: 5, color: palette.muted, fontSize: 12, fontWeight: 900, textTransform: 'uppercase', opacity: progress}}>{relationLabel(edge)}</div>
  </div>;
  return <div data-vex-required-edge={edge?.edge_id} style={{position: 'relative', alignSelf: 'center', width: 28, height: 48, flex: '0 0 28px'}}>
    <div style={{position: 'absolute', left: 0, right: 0, top: 22, height: 4, backgroundColor: `${color}35`}}><div style={{width: `${progress * 100}%`, height: '100%', backgroundColor: color, transformOrigin: 'left center'}} /></div>
    <div style={{position: 'absolute', right: -2, top: 17, width: 14, height: 14, borderRight: `4px solid ${color}`, borderBottom: `4px solid ${color}`, opacity: progress, transform: 'rotate(-45deg)'}} />
    <div style={{position: 'absolute', left: '50%', bottom: 33, width: 82, color: palette.muted, fontSize: 12, fontWeight: 900, textAlign: 'center', textTransform: 'uppercase', opacity: progress, transform: 'translateX(-50%)'}}>{relationLabel(edge)}</div>
  </div>;
};

const MetricScene = ({program}) => (
  <Canvas program={program}>{({palette, frame, fps, durationInFrames, orientation}) => {
    const portrait = orientation === 'portrait';
    const square = orientation === 'square';
    const nodes = list(program.nodes);
    const hero = nodes.find((node) => node.value) || nodes[0];
    const support = nodes.filter((node) => node.node_id !== hero?.node_id).slice(0, 3);
    const heroReveal = revealFor(program, hero?.node_id || 'hero', 0, frame, fps, durationInFrames);
    const percentage = /%/.test(text(hero?.value)) ? clamp(Number.parseFloat(hero?.value) || 0, 0, 100) : null;
    return <>
      <div style={{position: 'absolute', left: portrait || square ? 50 : 68, top: portrait ? 72 : square ? 50 : 64}}><Header program={program} palette={palette} orientation={orientation} contentWidth={portrait ? 620 : square ? 800 : 660} /></div>
      <div data-vex-node-id={hero?.node_id} data-vex-required-label={hero?.label} style={{boxSizing: 'border-box', position: 'absolute', left: portrait ? 50 : square ? 470 : 790, top: portrait ? 430 : square ? 300 : 116, width: portrait ? 620 : square ? 380 : 420, minHeight: portrait ? 360 : square ? 430 : 480, padding: portrait ? 42 : 38, backgroundColor: palette.surface, color: palette.ink, outline: `3px solid ${palette.accent2}`, borderRadius: 8, opacity: heroReveal, transform: `scale(${0.94 + heroReveal * 0.06})`, overflow: 'hidden'}}>
        <div style={{fontSize: 19, fontWeight: 900, color: palette.accent2, textTransform: 'uppercase'}}>Grounded signal</div>
        <div style={{marginTop: 30, fontSize: measuredSize(hero?.value || hero?.label, portrait ? 530 : square ? 300 : 340, portrait ? 112 : 96, 48, 900), lineHeight: 0.92, fontWeight: 900}}>{hero?.value || hero?.label}</div>
        {percentage !== null ? <div style={{position: 'relative', marginTop: 42, width: '100%', height: 18, backgroundColor: `${palette.ink}18`, overflow: 'hidden'}}><div style={{width: `${percentage * heroReveal}%`, height: '100%', backgroundColor: palette.accent2}} /><div style={{position: 'absolute', left: `${percentage}%`, top: -8, width: 3, height: 34, backgroundColor: palette.accent}} /></div> : null}
        {hero?.value && hero?.label !== hero?.value ? <div style={{marginTop: 24, fontSize: measuredSize(hero.label, portrait ? 530 : square ? 300 : 340, 36, 22, 800), lineHeight: 1.08, fontWeight: 800}}>{hero.label}</div> : null}
      </div>
      <div style={{position: 'absolute', left: portrait || square ? 50 : 68, top: portrait ? 830 : square ? 330 : 380, width: portrait ? 620 : square ? 380 : 650, display: 'grid', gridTemplateColumns: portrait || square ? '1fr' : 'repeat(2, 1fr)', gap: 16}}>
        {support.map((node, index) => <NodeCard key={node.node_id} program={program} node={node} index={index + 1} palette={palette} frame={frame} fps={fps} durationInFrames={durationInFrames} width={portrait ? 620 : square ? 380 : 315} compact light={index === 0} />)}
      </div>
    </>;
  }}</Canvas>
);

const KineticTypeScene = ({program}) => (
  <Canvas program={program}>{({palette, frame, fps, durationInFrames, orientation}) => {
    const portrait = orientation === 'portrait';
    const nodes = [...list(program.nodes)].sort((left, right) => Number(right.emphasis || 0) - Number(left.emphasis || 0));
    const hero = nodes[0];
    const support = nodes.slice(1, 4);
    const reveal = revealFor(program, hero?.node_id || 'hero', 0, frame, fps, durationInFrames);
    const heroWidth = portrait ? 620 : 1120;
    const heroText = text(program.title).length > text(hero?.label).length ? program.title : hero?.label;
    return <>
      <div style={{position: 'absolute', left: portrait ? 50 : 72, top: portrait ? 72 : 62, color: palette.accent, fontSize: 18, fontWeight: 900, textTransform: 'uppercase'}}>{program.eyebrow}</div>
      <div data-vex-node-id={hero?.node_id} data-vex-required-label={hero?.label} style={{position: 'absolute', left: portrait ? 50 : 72, right: portrait ? 50 : 72, top: portrait ? 210 : 145, opacity: reveal, transform: `translateY(${(1 - reveal) * 32}px)`}}>
        <div style={{maxWidth: heroWidth, fontSize: measuredSize(heroText, heroWidth, portrait ? 104 : 126, portrait ? 48 : 58, 900), lineHeight: 0.86, fontWeight: 900, textTransform: 'uppercase'}}>{heroText}</div>
        <div style={{marginTop: 34, width: `${30 + reveal * 42}%`, height: portrait ? 10 : 14, backgroundColor: palette.accent}} />
      </div>
      <div style={{position: 'absolute', left: portrait ? 50 : 76, right: portrait ? 50 : 76, bottom: portrait ? 118 : 92, display: 'grid', gridTemplateColumns: portrait ? '1fr' : `repeat(${Math.max(support.length, 1)}, 1fr)`, gap: portrait ? 18 : 32}}>
        {support.map((node, index) => <div key={node.node_id} data-vex-node-id={node.node_id} data-vex-required-label={node.label} style={{display: 'grid', gridTemplateColumns: '40px 1fr', gap: 12, borderTop: `2px solid ${index % 2 ? palette.accent2 : palette.accent}`, paddingTop: 14, color: palette.text}}><span style={{color: palette.accent, fontSize: 15, fontWeight: 900}}>0{index + 2}</span><strong style={{fontSize: measuredSize(node.label, portrait ? 520 : 290, portrait ? 27 : 30, 18, 800), lineHeight: 1, fontWeight: 800}}>{node.label}</strong></div>)}
      </div>
    </>;
  }}</Canvas>
);

const FlowScene = ({program, timeline = false}) => (
  <Canvas program={program}>{({palette, frame, fps, durationInFrames, orientation}) => {
    const portrait = orientation === 'portrait';
    const square = orientation === 'square';
    const nodes = list(program.nodes);
    const edges = list(program.edges);
    const cardWidth = portrait ? 620 : square ? Math.floor((784 - Math.max(nodes.length - 1, 0) * 64) / Math.max(nodes.length, 1)) : Math.floor((1090 - Math.max(nodes.length - 1, 0) * 64) / Math.max(nodes.length, 1));
    return <>
      <div style={{position: 'absolute', left: portrait ? 50 : square ? 50 : 68, top: portrait ? 64 : 58}}><Header program={program} palette={palette} orientation={orientation} contentWidth={portrait ? 620 : square ? 800 : 1070} /></div>
      <div style={{position: 'absolute', left: portrait ? 50 : square ? 58 : 82, right: portrait ? 50 : square ? 58 : 82, top: portrait ? 410 : square ? 350 : 360, bottom: portrait ? 120 : 105, display: 'flex', flexDirection: portrait ? 'column' : 'row', alignItems: 'stretch', justifyContent: 'center', gap: 18}}>
        {nodes.map((node, index) => <React.Fragment key={node.node_id}>
          <NodeCard program={program} node={node} index={index} palette={palette} frame={frame} fps={fps} durationInFrames={durationInFrames} width={cardWidth} compact={nodes.length >= 4 || (square && nodes.length >= 3)} light={timeline && index === nodes.length - 1} />
          {index < nodes.length - 1 ? <RelationConnector program={program} edge={edges.find((edge) => edge.source_id === node.node_id && edge.target_id === nodes[index + 1]?.node_id) || edges[index]} palette={palette} frame={frame} fps={fps} durationInFrames={durationInFrames} portrait={portrait} /> : null}
        </React.Fragment>)}
      </div>
    </>;
  }}</Canvas>
);

const ContrastScene = ({program}) => (
  <Canvas program={program}>{({palette, frame, fps, durationInFrames, orientation}) => {
    const portrait = orientation === 'portrait';
    const square = orientation === 'square';
    const nodes = list(program.nodes).slice(0, 2);
    return <>
      <div style={{position: 'absolute', left: portrait || square ? 50 : 68, top: portrait ? 64 : 58}}><Header program={program} palette={palette} orientation={orientation} contentWidth={portrait ? 620 : square ? 800 : 1080} /></div>
      <AnnotationRail program={program} palette={palette} orientation={orientation} top={portrait ? 380 : square ? 250 : 275} />
      <div style={{position: 'absolute', left: portrait || square ? 50 : 82, right: portrait || square ? 50 : 82, top: portrait ? 450 : square ? 330 : 350, bottom: portrait ? 94 : 100, display: 'grid', gridTemplateColumns: portrait ? '1fr' : '1fr 1fr', gap: square ? 18 : 22}}>
        {nodes.map((node, index) => <NodeCard key={node.node_id} program={program} node={node} index={index} palette={palette} frame={frame} fps={fps} durationInFrames={durationInFrames} width={portrait ? 620 : square ? 391 : 547} light={index === 1} />)}
      </div>
    </>;
  }}</Canvas>
);

const InterfaceScene = ({program}) => (
  <Canvas program={program}>{({palette, frame, fps, durationInFrames, orientation}) => {
    const portrait = orientation === 'portrait';
    const square = orientation === 'square';
    const nodes = list(program.nodes).slice(0, 4);
    const surfaceReveal = revealFor(program, nodes[0]?.node_id || 'surface', 0, frame, fps, durationInFrames);
    return <>
      <div style={{position: 'absolute', left: portrait || square ? 50 : 64, top: portrait ? 64 : square ? 50 : 58}}><Header program={program} palette={palette} orientation={orientation} contentWidth={portrait ? 620 : square ? 800 : 540} /></div>
      <div style={{boxSizing: 'border-box', position: 'absolute', left: portrait || square ? 50 : 655, top: portrait ? 430 : square ? 280 : 74, width: portrait ? 620 : square ? 800 : 555, height: portrait ? 710 : square ? 560 : 560, backgroundColor: palette.surface, color: palette.ink, outline: `2px solid ${palette.accent2}`, borderRadius: 8, overflow: 'hidden', opacity: surfaceReveal, transform: `translateX(${(1 - surfaceReveal) * 24}px)`}}>
        <div style={{height: 52, display: 'flex', alignItems: 'center', gap: 9, padding: '0 18px', backgroundColor: '#DDE5EC'}}>{['#EF4444', '#F59E0B', '#22C55E'].map((color) => <span key={color} style={{width: 13, height: 13, backgroundColor: color}} />)}</div>
        <div style={{padding: square ? 20 : 24, display: 'grid', gap: square ? 12 : 16}}>
          {nodes.map((node, index) => {
            const reveal = revealFor(program, node.node_id, index, frame, fps, durationInFrames);
            return <div key={node.node_id} data-vex-node-id={node.node_id} data-vex-required-label={node.label} style={{minHeight: portrait ? 122 : square ? 86 : 100, display: 'grid', gridTemplateColumns: '64px 1fr', gap: 18, alignItems: 'center', padding: 16, backgroundColor: index === 0 ? `${palette.accent2}22` : '#FFFFFF', outline: `1px solid ${palette.accent2}55`, borderRadius: 6, opacity: reveal, transform: `translateY(${(1 - reveal) * 18}px)`}}><div style={{height: 54, backgroundColor: index % 2 ? palette.accent : palette.accent2}} /><div><div style={{fontSize: measuredSize(node.label, portrait ? 430 : square ? 650 : 380, 27, 18, 850), lineHeight: 1.05, fontWeight: 850}}>{node.label}</div>{node.detail && node.detail !== node.label ? <div style={{marginTop: 9, fontSize: 16, color: '#475569'}}>{node.detail}</div> : null}</div></div>;
          })}
        </div>
      </div>
    </>;
  }}</Canvas>
);

const EmphasisScene = ({program}) => (
  <Canvas program={program}>{({palette, frame, fps, durationInFrames, orientation}) => {
    const portrait = orientation === 'portrait';
    const square = orientation === 'square';
    const nodes = list(program.nodes).slice(0, 4);
    return <>
      <div style={{position: 'absolute', left: portrait || square ? 50 : 82, top: portrait ? 82 : square ? 58 : 76}}><Header program={program} palette={palette} orientation={orientation} contentWidth={portrait ? 620 : square ? 800 : 1100} /></div>
      <div style={{position: 'absolute', left: portrait || square ? 50 : 100, right: portrait || square ? 50 : 100, bottom: portrait ? 100 : 110, display: square ? 'grid' : 'flex', gridTemplateColumns: square ? '1fr 1fr' : undefined, flexDirection: portrait ? 'column' : 'row', gap: 16, justifyContent: 'center'}}>
        {nodes.map((node, index) => <NodeCard key={node.node_id} program={program} node={node} index={index} palette={palette} frame={frame} fps={fps} durationInFrames={durationInFrames} width={portrait ? 620 : square ? 392 : Math.floor((1080 - Math.max(nodes.length - 1, 0) * 16) / Math.max(nodes.length, 1))} compact light={index === 0} />)}
      </div>
    </>;
  }}</Canvas>
);

const openColor = (value, palette) => {
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

const eased = (value, easing) => {
  const p = clamp(value, 0, 1);
  if (easing === 'ease_in') return p * p * p;
  if (easing === 'ease_out' || easing === 'spring_snappy') return 1 - Math.pow(1 - p, 3);
  if (easing === 'ease_in_out' || easing === 'spring_gentle') return p * p * (3 - 2 * p);
  return p;
};

const openTrackValue = (tracks, property, progress, fallback) => {
  const track = tracks.find((item) => text(item.property) === property);
  const keyframes = list(track?.keyframes).filter((item) => Number.isFinite(Number(item?.t)) && Number.isFinite(Number(item?.value))).sort((a, b) => Number(a.t) - Number(b.t));
  if (!keyframes.length) return fallback;
  if (progress <= Number(keyframes[0].t)) return Number(keyframes[0].value);
  if (progress >= Number(keyframes[keyframes.length - 1].t)) return Number(keyframes[keyframes.length - 1].value);
  const rightIndex = keyframes.findIndex((item) => Number(item.t) >= progress);
  const left = keyframes[Math.max(0, rightIndex - 1)];
  const right = keyframes[rightIndex];
  const span = Math.max(Number(right.t) - Number(left.t), 0.0001);
  const local = eased((progress - Number(left.t)) / span, text(right.easing || left.easing || 'linear'));
  return interpolate(local, [0, 1], [Number(left.value), Number(right.value)], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
};

const openElementStyle = (element, tracks, progress, palette, base) => {
  const layout = element.layout || {};
  const style = element.style || {};
  const translateX = openTrackValue(tracks, 'translate_x', progress, 0) * base.width;
  const translateY = openTrackValue(tracks, 'translate_y', progress, 0) * base.height;
  const scale = openTrackValue(tracks, 'scale', progress, 1);
  const rotation = openTrackValue(tracks, 'rotation', progress, 0);
  const opacity = clamp(openTrackValue(tracks, 'opacity', progress, Number(style.opacity ?? 1)), 0, 1);
  const blur = Math.max(0, openTrackValue(tracks, 'blur', progress, Number(style.blur ?? 0)));
  const emphasis = clamp(openTrackValue(tracks, 'emphasis', progress, 0), 0, 1);
  const framed = ['shape', 'token', 'metric', 'group', 'chart', 'image'].includes(text(element.type));
  const elementWidth = Math.max(2, Number(layout.width || 0.1) * base.width);
  const requestedFontSize = clamp(Number(style.font_size || 30), 12, 110);
  const fittedFontSize = text(element.text) ? measuredSize(element.text, Math.max(40, elementWidth - 28), requestedFontSize, 13, Number(style.font_weight || 750)) : requestedFontSize;
  return {
    boxSizing: 'border-box',
    position: 'absolute',
    left: Number(layout.x || 0) * base.width,
    top: Number(layout.y || 0) * base.height,
    width: elementWidth,
    height: Math.max(2, Number(layout.height || 0.1) * base.height),
    color: openColor(style.color || (style.fill === 'text' ? 'text' : 'ink'), palette),
    backgroundColor: ['shape', 'token', 'metric', 'group'].includes(text(element.type)) ? openColor(style.fill || 'surface', palette) : 'transparent',
    border: framed && Number(style.stroke_width || 2) > 0 ? `${Math.max(1, Number(style.stroke_width || 2))}px solid ${openColor(style.stroke || 'accent', palette)}` : undefined,
    borderRadius: Math.max(0, Number(style.radius || (element.type === 'token' ? 8 : 0))),
    display: 'grid',
    placeItems: 'center',
    padding: text(element.text) && framed ? Math.max(8, Math.round(base.width * 0.012)) : 0,
    textAlign: 'center',
    fontSize: fittedFontSize,
    fontWeight: clamp(Number(style.font_weight || 750), 300, 950),
    lineHeight: 1.08,
    overflow: framed ? 'hidden' : 'visible',
    opacity,
    filter: blur > 0 ? `blur(${blur}px)` : undefined,
    transform: `translate(${translateX}px, ${translateY}px) scale(${scale * (1 + emphasis * 0.045)}) rotate(${rotation}deg)`,
    transformOrigin: 'center',
    boxShadow: emphasis > 0 ? `0 0 ${18 + emphasis * 34}px ${openColor(style.stroke || 'accent', palette)}55` : undefined,
  };
};

const OpenVisualElement = ({element, tracks, progress, palette, base}) => {
  const style = openElementStyle(element, tracks, progress, palette, base);
  const repeat = clamp(Math.round(Number(element.repeat || 1)), 1, 24);
  const semanticId = text(element.binding?.id);
  const requiredLabel = text(element.text);
  const type = text(element.type);
  const role = text(element.role);
  const routeProgress = clamp(openTrackValue(tracks, 'progress', progress, 1), 0, 1);
  if (type === 'particle') return <div style={{...style, backgroundColor: 'transparent', border: 0, overflow: 'visible'}} data-vex-node-id={semanticId}>
    {Array.from({length: repeat}).map((_, index) => <i key={index} style={{position: 'absolute', left: `${(index * 37) % 94}%`, top: `${(index * 53) % 88}%`, width: 5 + index % 4, height: 5 + index % 4, borderRadius: '50%', backgroundColor: index % 2 ? palette.accent : palette.accent2, opacity: 0.34 + routeProgress * 0.5, transform: `translateY(${(1 - routeProgress) * (20 + index % 5 * 8)}px)`}} />)}
  </div>;
  if (type === 'chart') return <div style={style} data-vex-node-id={semanticId} data-vex-required-label={requiredLabel || undefined}>
    <div style={{position: 'absolute', inset: '12% 10%', display: 'flex', alignItems: 'end', gap: '6%'}}>{Array.from({length: Math.max(3, repeat)}).map((_, index) => <i key={index} style={{flex: 1, height: `${(30 + ((index * 29) % 66)) * routeProgress}%`, backgroundColor: index % 2 ? palette.accent2 : palette.accent}} />)}</div>
    {requiredLabel ? <strong style={{position: 'absolute', left: 12, top: 10, right: 12}}>{requiredLabel}</strong> : null}
  </div>;
  if (type === 'image' && text(element.asset?.data_uri).startsWith('data:image/')) return <img src={element.asset.data_uri} alt={requiredLabel} style={{...style, objectFit: text(element.asset.fit) || 'cover', padding: 0}} data-vex-node-id={semanticId} />;
  return <div style={{...style, clipPath: role === 'transformation_gate' ? 'polygon(0 0, 100% 12%, 72% 88%, 0 100%)' : undefined}} data-vex-node-id={semanticId || undefined} data-vex-required-label={requiredLabel || undefined}>
    {type === 'path' || type === 'connector' ? <div style={{width: `${routeProgress * 100}%`, height: Math.max(3, Number(element.style?.stroke_width || 3)), backgroundColor: openColor(element.style?.stroke || 'accent', palette)}} /> : null}
    {role === 'source_signal' ? <div style={{position: 'absolute', inset: '18%', display: 'grid', gap: 5}}>{[0, 1, 2].map((index) => <i key={index} style={{height: 4, width: `${72 + index * 10}%`, backgroundColor: index === 1 ? palette.accent : palette.accent2, opacity: 0.72}} />)}</div> : null}
    {role === 'transformation_gate' ? <div style={{position: 'absolute', inset: '14%', display: 'grid', placeItems: 'center'}}><div style={{width: '54%', aspectRatio: 1, borderRadius: '50%', border: `6px solid ${palette.text}`, boxShadow: `0 0 32px ${palette.accent2}88`, transform: `rotate(${routeProgress * 180}deg)`}}><div style={{width: '42%', height: '42%', margin: '29%', backgroundColor: palette.accent2, transform: 'rotate(45deg)'}} /></div></div> : null}
    {role === 'compressed_representation' ? <><div style={{position: 'absolute', inset: '9%', border: `2px solid ${palette.ink}55`, transform: 'translate(-7px, 7px)', zIndex: 0}} /><strong style={{position: 'relative', zIndex: 2}}>{requiredLabel}</strong></> : null}
    {role === 'selection_result' ? <><div style={{position: 'absolute', left: 12, top: `${12 + routeProgress * 64}%`, right: 12, height: 3, backgroundColor: palette.accent, boxShadow: `0 0 18px ${palette.accent}`}} /><strong style={{position: 'relative', zIndex: 2}}>{requiredLabel}</strong></> : null}
    {!['source_signal', 'transformation_gate', 'compressed_representation', 'selection_result'].includes(role) && type !== 'path' && type !== 'connector' ? requiredLabel : null}
    {['shape', 'token', 'metric'].includes(type) ? <div style={{position: 'absolute', left: 0, bottom: 0, width: `${routeProgress * 100}%`, height: 5, backgroundColor: openColor(element.style?.stroke || 'accent', palette)}} /> : null}
  </div>;
};

const OpenVisualRelations = ({program, elements, palette, base, progress}) => {
  const byId = new Map(elements.map((item) => [text(item.element_id), item]));
  return <svg viewBox={`0 0 ${base.width} ${base.height}`} style={{position: 'absolute', inset: 0, width: base.width, height: base.height, overflow: 'visible', pointerEvents: 'none'}}>
    <defs><marker id="vex-open-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill={palette.accent} /></marker></defs>
    {list(program.relations).map((relation, index) => {
      const source = byId.get(text(relation.source_id));
      const target = byId.get(text(relation.target_id));
      if (!source || !target) return null;
      const a = source.layout || {};
      const b = target.layout || {};
      const x1 = (Number(a.x || 0) + Number(a.width || 0) / 2) * base.width;
      const y1 = (Number(a.y || 0) + Number(a.height || 0) / 2) * base.height;
      const x2 = (Number(b.x || 0) + Number(b.width || 0) / 2) * base.width;
      const y2 = (Number(b.y || 0) + Number(b.height || 0) / 2) * base.height;
      const bend = Math.max(42, Math.abs(x2 - x1) * 0.35);
      const path = `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
      const visible = clamp((progress - (0.28 + index * 0.08)) / 0.22, 0, 1);
      return <path key={relation.relation_id || index} d={path} fill="none" stroke={openColor(relation.style?.stroke || 'accent', palette)} strokeWidth={Math.max(2, Number(relation.style?.stroke_width || 4))} pathLength="1" strokeDasharray="1" strokeDashoffset={1 - visible} markerEnd="url(#vex-open-arrow)" data-vex-required-edge={text(relation.binding?.id) || text(relation.relation_id)} />;
    })}
  </svg>;
};

const OpenVisualScene = ({program}) => (
  <Canvas program={program}>{({palette, base, frame, durationInFrames}) => {
    const open = program.open_visual_program || {};
    const elements = list(open.elements);
    const tracks = list(open.tracks);
    const progress = clamp(frame / Math.max(durationInFrames - 1, 1), 0, 1);
    return <div data-vex-open-visual-program={open.program_id} data-vex-open-visual-signature={open.signature} style={{position: 'absolute', inset: 0}}>
      <OpenVisualRelations program={open} elements={elements} palette={palette} base={base} progress={progress} />
      {elements.map((element) => <OpenVisualElement key={element.element_id} element={element} tracks={tracks.filter((track) => text(track.target_id) === text(element.element_id))} progress={progress} palette={palette} base={base} />)}
    </div>;
  }}</Canvas>
);

const VexAutoVisual = ({program}) => {
  if (!program) return <AbsoluteFill style={{backgroundColor: '#101418'}} />;
  if (program.open_visual_program?.elements?.length) return <OpenVisualScene program={program} />;
  if (program.creative_direction?.medium_family === 'kinetic_typography') return <KineticTypeScene program={program} />;
  if (program.scene_family === 'metric') return <MetricScene program={program} />;
  if (program.scene_family === 'contrast') return <ContrastScene program={program} />;
  if (program.scene_family === 'timeline') return <FlowScene program={program} timeline />;
  if (program.scene_family === 'interface') return <InterfaceScene program={program} />;
  if (program.scene_family === 'emphasis') return <EmphasisScene program={program} />;
  return <FlowScene program={program} />;
};

const Root = () => <Composition id="VexAutoVisual" component={VexAutoVisual} durationInFrames={90} fps={30} width={1280} height={720} defaultProps={{program: null}} calculateMetadata={({props}) => {
  const program = props.program || {};
  const fps = clamp(Number(program.fps) || 30, 15, 120);
  const width = Math.max(320, Math.round(Number(program.width) || 1280));
  const height = Math.max(240, Math.round(Number(program.height) || 720));
  const durationSec = clamp(Number(program.duration_sec) || 3, 0.5, 30);
  return {durationInFrames: Math.max(1, Math.round(durationSec * fps)), fps, width, height, defaultCodec: 'h264', defaultPixelFormat: 'yuv420p'};
}} />;

registerRoot(Root);
