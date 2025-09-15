import { JSDOM } from 'jsdom';
import { GridWorldEnvironment } from '../src/rl/environment.js';
import { initRenderer, render } from '../src/ui/renderGrid.js';

export async function run(assert) {
  const previousWindow = global.window;
  const previousDocument = global.document;
  const dom = new JSDOM(`<div id="grid"></div>`, { pretendToBeVisual: true });
  global.window = dom.window;
  global.document = dom.window.document;

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

    const cells = gridEl.querySelectorAll('.cell');
    assert.strictEqual(cells.length, 4);

    const cell00 = cells[0];
    const left00 = cell00.querySelector('.q-value.q-left');
    assert.ok(left00, 'left q-value should exist for cell (0,0)');
    assert.strictEqual(left00.dataset.action, 'left');
    assert.strictEqual(parseFloat(left00.dataset.value).toFixed(2), '0.80');
    assert.strictEqual(cell00.querySelector('.q-value.best').dataset.action, 'left');

    const down00 = cell00.querySelector('.q-value.q-down');
    assert.ok(down00.textContent.includes('-0.50'));
    if (down00.style.background.startsWith('rgba')) {
      const components = down00.style.background.match(/\d+\.\d+|\d+/g) || [];
      assert.ok(components.length >= 2, 'rgba color should provide at least red and green components');
      const [r, g] = components.map(Number);
      assert.ok(r >= g, 'negative q-value should emphasise red channel');
    } else {
      assert.ok(down00.style.background.startsWith('hsla(0'));
    }

    const cell10 = cells[1];
    const right10 = cell10.querySelector('.q-value.q-right');
    assert.ok(right10.textContent.includes('0.90'));
    assert.strictEqual(parseFloat(right10.dataset.value).toFixed(2), '0.90');
    assert.strictEqual(cell10.querySelector('.q-value.best').dataset.action, 'right');
    if (right10.style.background.startsWith('rgba')) {
      const components = right10.style.background.match(/\d+\.\d+|\d+/g) || [];
      assert.ok(components.length >= 2, 'rgba color should provide at least red and green components');
      const [r, g] = components.map(Number);
      assert.ok(g >= r, 'positive q-value should emphasise green channel');
    } else {
      assert.ok(right10.style.background.startsWith('hsla(140'));
    }

    const up10 = cell10.querySelector('.q-value.q-up');
    assert.ok(up10.textContent.includes('↑'));
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
  }
}
