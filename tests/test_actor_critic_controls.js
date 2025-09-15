import { JSDOM } from 'jsdom';

export async function run(assert) {
  const dom = new JSDOM(`<!doctype html><html><body>
    <select id="agent-select">
      <option value="rl">Q-learning</option>
      <option value="ac">Actor-Critic</option>
    </select>
    <select id="policy-select">
      <option value="epsilon-greedy">Epsilon-Greedy</option>
      <option value="greedy">Greedy</option>
    </select>
    <input id="epsilon-slider" type="range" value="1" />
    <span id="epsilon-value">1.00</span>
    <span id="epsilon">1.00</span>
    <input id="interval-slider" type="range" value="100" />
    <span id="interval-value">100</span>
    <input id="learning-rate-slider" type="range" value="0.1" />
    <span id="learning-rate-value">0.10</span>
    <input id="lambda-slider" type="range" value="0.9" />
    <span id="lambda-value">0.90</span>
    <button id="start"></button>
    <button id="pause"></button>
    <button id="reset"></button>
    <button id="save"></button>
    <button id="load"></button>
  </body></html>`, { url: 'http://localhost' });

  const { window } = dom;
  const previousWindow = global.window;
  const previousDocument = global.document;
  const previousHTMLElement = global.HTMLElement;
  const previousEvent = global.Event;

  global.window = window;
  global.document = window.document;
  global.HTMLElement = window.HTMLElement;
  global.Event = window.Event;

  try {
    const { bindControls } = await import('../src/ui/bindControls.js');

    const initialAgent = {
      epsilon: 0.5,
      policy: 'epsilon-greedy',
      learningRate: 0.1,
      lambda: 0.8
    };

    const trainer = {
      intervalMs: 100,
      metrics: { epsilon: initialAgent.epsilon },
      agent: initialAgent,
      state: {},
      setIntervalMs(ms) {
        this.intervalMs = ms;
      },
      start() {},
      pause() {},
      reset() {
        if (this.agent && typeof this.agent.reset === 'function') {
          this.agent.reset();
        }
      }
    };

    const render = () => {};
    const getEnv = () => ({ size: 5, obstacles: [] });
    const setEnv = () => {};

    bindControls(trainer, initialAgent, render, getEnv, setEnv);

    const agentSelect = window.document.getElementById('agent-select');
    const learningRateSlider = window.document.getElementById('learning-rate-slider');
    const learningRateValue = window.document.getElementById('learning-rate-value');

    agentSelect.value = 'ac';
    agentSelect.dispatchEvent(new window.Event('change'));

    assert.strictEqual(learningRateSlider.disabled, true);
    assert.strictEqual(learningRateValue.textContent, 'N/A');

    assert.doesNotThrow(() => {
      learningRateSlider.value = '0.3';
      learningRateSlider.dispatchEvent(new window.Event('input', { bubbles: true }));
    });

    agentSelect.value = 'rl';
    agentSelect.dispatchEvent(new window.Event('change'));

    assert.strictEqual(learningRateSlider.disabled, false);
    assert.strictEqual(
      learningRateValue.textContent,
      parseFloat(learningRateSlider.value).toFixed(2)
    );
  } finally {
    global.window = previousWindow;
    global.document = previousDocument;
    global.HTMLElement = previousHTMLElement;
    global.Event = previousEvent;
  }
}
