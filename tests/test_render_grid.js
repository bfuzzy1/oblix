import { JSDOM } from 'jsdom';
import { GridWorldEnvironment } from '../src/rl/environment.js';
import { initRenderer, render } from '../src/ui/renderGrid.js';

export async function run(assert) {
  const previousWindow = global.window;
  const previousDocument = global.document;
  const previousLocalStorage = globalThis.localStorage;
  const dom = new JSDOM(`<div id="grid"></div>`, { pretendToBeVisual: true });
  global.window = dom.window;
  global.document = dom.window.document;

  const storageMock = (() => {
    let store = new Map();
    return {
      getItem(key) {
        return store.has(key) ? store.get(key) : null;
      },
      setItem(key, value) {
        store.set(String(key), String(value));
      },
      removeItem(key) {
        store.delete(key);
      },
      clear() {
        store.clear();
      }
    };
  })();

  Object.defineProperty(global.window, 'localStorage', {
    configurable: true,
    value: storageMock
  });
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    writable: true,
    value: storageMock
  });

  try {
    const gridEl = document.getElementById('grid');
    const env = new GridWorldEnvironment(2);
    initRenderer(env, gridEl, env.size);

    const agent = {
      qTable: new Map([
        ['0,0', Float32Array.from([0.25, -0.5, 0.8, 0.1])],
        ['1,0', Float32Array.from([-0.1, 0.3, -0.2, 0.9])]
      ])
    };

    render(env.getState(), agent);

    let cells = gridEl.querySelectorAll('.cell');
    assert.strictEqual(cells.length, 4);

    const overlays = gridEl.querySelectorAll('.q-layer, .q-value');
    assert.strictEqual(overlays.length, 0, 'grid should not render q-value overlays');

    const cell00 = cells[0];
    assert.ok(cell00.classList.contains('agent'), 'starting cell should display agent');

    const goalCell = cells[3];
    assert.ok(goalCell.classList.contains('goal'), 'bottom-right cell should display goal');

    const toggleCell = cells[1];
    toggleCell.dispatchEvent(new dom.window.Event('click'));
    cells = gridEl.querySelectorAll('.cell');
    assert.ok(cells[1].classList.contains('obstacle'), 'clicked cell should toggle obstacle state');
  } finally {
    if (previousWindow === undefined) {
      delete global.window;
    } else {
      global.window = previousWindow;
    }
    if (previousDocument === undefined) {
      delete global.document;
    } else {
      global.document = previousDocument;
    }
    if (previousLocalStorage === undefined) {
      delete globalThis.localStorage;
    } else {
      globalThis.localStorage = previousLocalStorage;
    }
  }
}
