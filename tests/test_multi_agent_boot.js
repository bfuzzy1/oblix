import { readFileSync } from 'fs';
import { JSDOM } from 'jsdom';

function createDom() {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  return new JSDOM(html, {
    url: 'http://localhost/',
    pretendToBeVisual: true,
    runScripts: 'outside-only'
  });
}

function stubCanvas(window) {
  window.HTMLCanvasElement.prototype.getContext = () => ({
    clearRect() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    stroke() {},
    fillRect() {},
    fillText() {},
    font: ''
  });
}

function applyGlobals(window) {
  const assignments = {
    window,
    document: window.document,
    navigator: window.navigator,
    HTMLElement: window.HTMLElement,
    HTMLCanvasElement: window.HTMLCanvasElement,
    CustomEvent: window.CustomEvent,
    Event: window.Event,
    requestAnimationFrame: cb => window.requestAnimationFrame(cb),
    cancelAnimationFrame: handle => window.cancelAnimationFrame(handle),
    performance: window.performance,
    localStorage: window.localStorage,
    Worker: undefined
  };
  const originals = new Map();
  for (const [key, value] of Object.entries(assignments)) {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, key);
    originals.set(key, descriptor);
    Object.defineProperty(globalThis, key, {
      configurable: true,
      writable: true,
      enumerable: descriptor ? descriptor.enumerable : true,
      value
    });
  }
  return () => {
    for (const [key, descriptor] of originals.entries()) {
      if (descriptor === undefined) {
        delete globalThis[key];
      } else {
        Object.defineProperty(globalThis, key, descriptor);
      }
    }
  };
}

export async function run(assert) {
  const dom = createDom();
  const { window } = dom;
  stubCanvas(window);
  window.requestAnimationFrame = cb => setTimeout(cb, 0);
  window.cancelAnimationFrame = handle => clearTimeout(handle);
  window.Worker = undefined;

  const restoreGlobals = applyGlobals(window);
  try {
    await import('../src/ui/index.js');
    const grid = window.document.getElementById('grid');
    assert.ok(grid.children.length > 0, 'grid should render initial cells');
    const scenarioSelect = window.document.getElementById('scenario-select');
    assert.ok(scenarioSelect.options.length > 1, 'scenario options should populate');
  } finally {
    restoreGlobals();
    dom.window.close();
  }
}
