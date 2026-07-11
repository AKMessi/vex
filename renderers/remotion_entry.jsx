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
  const theme = program.theme || {};
  return {
    ...base,
    bg: text(theme.background) || base.bg,
    surface: text(theme.panel_fill) || base.surface,
    surfaceDark: text(theme.panel_dark) || base.surfaceDark,
    text: text(theme.text_primary) || base.text,
    muted: text(theme.text_secondary) || base.muted,
    accent: text(theme.accent) || base.accent,
    accent2: text(theme.accent_secondary) || base.accent2,
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

const NodeCard = ({program, node, index, palette, frame, fps, durationInFrames, width, compact = false, light = false}) => {
  const reveal = revealFor(program, node.node_id, index, frame, fps, durationInFrames);
  const foreground = light ? palette.ink : palette.text;
  const background = light ? palette.surface : palette.surfaceDark;
  const labelSize = measuredSize(node.label, width - 44, compact ? 28 : 34, 19, 850);
  return (
    <div data-vex-node-id={node.node_id} data-vex-required-label={node.label} style={{boxSizing: 'border-box', width, minHeight: compact ? 132 : 174, padding: compact ? 20 : 24, backgroundColor: background, color: foreground, outline: `2px solid ${index % 2 ? palette.accent2 : palette.accent}`, borderRadius: 7, opacity: reveal, transform: `translateY(${(1 - reveal) * 24}px)`, boxShadow: `0 18px 50px ${palette.bg}88`, overflow: 'hidden'}}>
      <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12}}>
        <div style={{width: 38, height: 38, flex: '0 0 auto', display: 'grid', placeItems: 'center', backgroundColor: index % 2 ? palette.accent2 : palette.accent, color: '#fff', fontSize: 20, fontWeight: 900}}>{index + 1}</div>
        {node.value ? <div style={{fontSize: measuredSize(node.value, width * 0.48, compact ? 29 : 38, 20, 900), fontWeight: 900, color: light ? palette.accent2 : palette.accent3}}>{node.value}</div> : null}
      </div>
      <div style={{marginTop: 18, fontSize: labelSize, lineHeight: 1.04, fontWeight: 850}}>{node.label}</div>
      {!compact && node.detail && node.detail !== node.label ? <div style={{marginTop: 12, fontSize: measuredSize(node.detail, width - 44, 19, 15, 600), lineHeight: 1.2, fontWeight: 600, color: light ? '#475569' : palette.muted}}>{node.detail}</div> : null}
    </div>
  );
};

const MetricScene = ({program}) => (
  <Canvas program={program}>{({palette, frame, fps, durationInFrames, orientation}) => {
    const portrait = orientation === 'portrait';
    const square = orientation === 'square';
    const nodes = list(program.nodes);
    const hero = nodes.find((node) => node.value) || nodes[0];
    const support = nodes.filter((node) => node.node_id !== hero?.node_id).slice(0, 3);
    const heroReveal = revealFor(program, hero?.node_id || 'hero', 0, frame, fps, durationInFrames);
    return <>
      <div style={{position: 'absolute', left: portrait || square ? 50 : 68, top: portrait ? 72 : square ? 50 : 64}}><Header program={program} palette={palette} orientation={orientation} contentWidth={portrait ? 620 : square ? 800 : 660} /></div>
      <div data-vex-node-id={hero?.node_id} data-vex-required-label={hero?.label} style={{boxSizing: 'border-box', position: 'absolute', left: portrait ? 50 : square ? 470 : 790, top: portrait ? 430 : square ? 300 : 116, width: portrait ? 620 : square ? 380 : 420, minHeight: portrait ? 360 : square ? 430 : 480, padding: portrait ? 42 : 38, backgroundColor: palette.surface, color: palette.ink, outline: `3px solid ${palette.accent2}`, borderRadius: 8, opacity: heroReveal, transform: `scale(${0.94 + heroReveal * 0.06})`, overflow: 'hidden'}}>
        <div style={{fontSize: 19, fontWeight: 900, color: palette.accent2, textTransform: 'uppercase'}}>Grounded signal</div>
        <div style={{marginTop: 30, fontSize: measuredSize(hero?.value || hero?.label, portrait ? 530 : square ? 300 : 340, portrait ? 112 : 96, 48, 900), lineHeight: 0.92, fontWeight: 900}}>{hero?.value || hero?.label}</div>
        {hero?.value && hero?.label !== hero?.value ? <div style={{marginTop: 24, fontSize: measuredSize(hero.label, portrait ? 530 : square ? 300 : 340, 36, 22, 800), lineHeight: 1.08, fontWeight: 800}}>{hero.label}</div> : null}
      </div>
      <div style={{position: 'absolute', left: portrait || square ? 50 : 68, top: portrait ? 830 : square ? 330 : 380, width: portrait ? 620 : square ? 380 : 650, display: 'grid', gridTemplateColumns: portrait || square ? '1fr' : 'repeat(2, 1fr)', gap: 16}}>
        {support.map((node, index) => <NodeCard key={node.node_id} program={program} node={node} index={index + 1} palette={palette} frame={frame} fps={fps} durationInFrames={durationInFrames} width={portrait ? 620 : square ? 380 : 315} compact light={index === 0} />)}
      </div>
    </>;
  }}</Canvas>
);

const FlowScene = ({program, timeline = false}) => (
  <Canvas program={program}>{({palette, frame, fps, durationInFrames, orientation}) => {
    const portrait = orientation === 'portrait';
    const square = orientation === 'square';
    const nodes = list(program.nodes);
    const cardWidth = portrait ? 620 : square ? Math.floor((784 - Math.max(nodes.length - 1, 0) * 64) / Math.max(nodes.length, 1)) : Math.floor((1090 - Math.max(nodes.length - 1, 0) * 64) / Math.max(nodes.length, 1));
    return <>
      <div style={{position: 'absolute', left: portrait ? 50 : square ? 50 : 68, top: portrait ? 64 : 58}}><Header program={program} palette={palette} orientation={orientation} contentWidth={portrait ? 620 : square ? 800 : 1070} /></div>
      <div style={{position: 'absolute', left: portrait ? 50 : square ? 58 : 82, right: portrait ? 50 : square ? 58 : 82, top: portrait ? 410 : square ? 350 : 360, bottom: portrait ? 120 : 105, display: 'flex', flexDirection: portrait ? 'column' : 'row', alignItems: 'stretch', justifyContent: 'center', gap: 18}}>
        {nodes.map((node, index) => <React.Fragment key={node.node_id}>
          <NodeCard program={program} node={node} index={index} palette={palette} frame={frame} fps={fps} durationInFrames={durationInFrames} width={cardWidth} compact={nodes.length >= 4 || (square && nodes.length >= 3)} light={timeline && index === nodes.length - 1} />
          {index < nodes.length - 1 ? <div style={{alignSelf: 'center', width: portrait ? 4 : 28, height: portrait ? 26 : 4, flex: '0 0 auto', backgroundColor: index % 2 ? palette.accent2 : palette.accent, position: 'relative'}}><div style={{position: 'absolute', right: portrait ? -6 : -2, bottom: portrait ? -2 : -6, width: 14, height: 14, borderRight: `4px solid ${palette.accent2}`, borderBottom: `4px solid ${palette.accent2}`, transform: portrait ? 'rotate(45deg)' : 'rotate(-45deg)'}} /></div> : null}
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

const VexAutoVisual = ({program}) => {
  if (!program) return <AbsoluteFill style={{backgroundColor: '#101418'}} />;
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
