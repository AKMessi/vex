// Browser measurements are observed evidence, not authored layout estimates.
export const measureVisualScene = (frame = 0) => {
  const root = document.querySelector('[data-vex-scene-graph],.ovp-stage,#hf-stage') || document.body;
  const rootBounds = root.getBoundingClientRect();
  const nodes = Array.from(root.querySelectorAll('[data-vex-sg-node],[data-vex-ovp-element],[data-element-id]')).map(el => {
    const bounds = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    const label = el.querySelector('strong') || (el.dataset.vexMeasureText === 'true' ? el : null);
    let textBounds = null;
    if (label && label.textContent?.trim()) {
      const range = document.createRange(); range.selectNodeContents(label);
      const box = range.getBoundingClientRect();
      textBounds = {x:box.x,y:box.y,width:box.width,height:box.height};
    }
    const visible = Number(style.opacity) > .1 && style.display !== 'none' && style.visibility !== 'hidden';
    const tolerance = 2;
    const clipped = Boolean(visible && textBounds && (textBounds.x < bounds.x-tolerance || textBounds.y < bounds.y-tolerance || textBounds.x+textBounds.width > bounds.x+bounds.width+tolerance || textBounds.y+textBounds.height > bounds.y+bounds.height+tolerance));
    return {element_id:el.dataset.vexSgNode || el.dataset.vexOvpElement || el.dataset.elementId,visible,text:label?.textContent?.trim() || '',font_size:Number.parseFloat(style.fontSize),font_family:style.fontFamily,bounds:{x:bounds.x,y:bounds.y,width:bounds.width,height:bounds.height},text_bounds:textBounds,clipped};
  });
  return {version:'vex-browser-telemetry-v1',frame,viewport:{width:innerWidth,height:innerHeight},root:{x:rootBounds.x,y:rootBounds.y,width:rootBounds.width,height:rootBounds.height},fonts_ready:document.fonts.status === 'loaded',nodes};
};
