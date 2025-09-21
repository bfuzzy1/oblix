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

  class MockTrainer {
    constructor(agent, env, options = {}) {
      this.agent = agent;
      this.env = env;
      this.intervalMs = options.intervalMs ?? 100;
      this.maxSteps = options.maxSteps ?? 50;
      this.liveChart = options.liveChart ?? null;
      this.metrics = {
        episode: 1,
        steps: 0,
        cumulativeReward: 0,
        epsilon: agent?.epsilon ?? 1
      };
      this.isRunning = false;
      this.onStepHandler = options.onStep || null;
      this.onProgressHandler = options.onProgress || null;
      MockTrainer.instances.push(this);
    }

    start() {
      this.isRunning = true;
    }

    pause() {
      this.isRunning = false;
    }

    reset() {
      this.isRunning = false;
    }

    resetTrainerState() {
      this.isRunning = false;
    }

    setIntervalMs(ms) {
      if (Number.isFinite(ms)) {
        this.intervalMs = ms;
      }
    }

    setMaxSteps(limit) {
      if (Number.isFinite(limit)) {
        this.maxSteps = limit;
      }
    }

    setEnvironment(env) {
      this.env = env;
    }
  }

  MockTrainer.instances = [];

  window.__OBLIX_TEST_HARNESS__ = true;
  window.__OBLIX_TEST_OVERRIDES__ = {
    RLTrainer: MockTrainer
  };

  const restoreGlobals = applyGlobals(window);
  try {
    await import('../src/ui/index.js');
    const grid = window.document.getElementById('grid');
    assert.ok(grid.children.length > 0, 'grid should render initial cells');
    const scenarioSelect = window.document.getElementById('scenario-select');
    assert.ok(scenarioSelect.options.length > 1, 'scenario options should populate');

    const gridSizeInput = window.document.getElementById('grid-size');
    gridSizeInput.value = '12';
    gridSizeInput.dispatchEvent(new window.Event('change', { bubbles: true }));

    const multiAgentControls = window.document.getElementById('multi-agent-controls');
    assert.ok(!multiAgentControls.classList.contains('is-hidden'), 'multi-agent controls become visible on large grid');

    const testApi = window.__oblixTestApi;
    assert.ok(testApi, 'test harness API should be exposed');
    assert.strictEqual(typeof testApi.setAgentCount, 'function', 'setAgentCount should be available');

    const initialTrainerCount = MockTrainer.instances.length;
    assert.ok(initialTrainerCount >= 1, 'base trainer instance should exist');

    const agentCountSelect = window.document.getElementById('agent-count');
    agentCountSelect.value = '3';
    agentCountSelect.dispatchEvent(new window.Event('change', { bubbles: true }));

    assert.strictEqual(MockTrainer.instances.length, initialTrainerCount + 2, 'increasing agent count instantiates additional trainers');
    assert.strictEqual(testApi.getMultiAgentState().agentCount, 3, 'internal agent count should reflect selection');

    const trainer = testApi.getTrainer();
    const liveChart = testApi.getLiveChart();
    assert.ok(liveChart, 'live chart instance should be available');
    assert.strictEqual(trainer.liveChart, null, 'base trainer live chart is disabled during multi-agent runs');

    const pushes = [];
    liveChart.push = (reward, epsilon) => {
      pushes.push({ reward, epsilon });
    };

    const baseMetrics = { episode: 10, steps: 20, cumulativeReward: 30, epsilon: 0.5 };
    const agent2Metrics = { episode: 8, steps: 40, cumulativeReward: 10, epsilon: 0.3 };
    const agent3Metrics = { episode: 12, steps: 50, cumulativeReward: 20, epsilon: 0.1 };

    trainer.metrics = baseMetrics;
    testApi.handleAdditionalProgress(2, { x: 1, y: 1 }, 0, false, agent2Metrics);
    testApi.handleAdditionalProgress(3, { x: 2, y: 2 }, 0, false, agent3Metrics);
    testApi.handleProgress({ x: 0, y: 0 }, 0, false, baseMetrics);

    const aggregated = testApi.computeAggregatedMetrics(baseMetrics);
    assert.strictEqual(aggregated.episode, 12, 'aggregated metrics use the max episode value');
    assert.ok(Math.abs(aggregated.steps - (20 + 40 + 50) / 3) < 1e-6, 'aggregated steps average across agents');
    assert.ok(Math.abs(aggregated.cumulativeReward - (30 + 10 + 20) / 3) < 1e-6, 'aggregated reward averages across agents');
    assert.ok(Math.abs(aggregated.epsilon - (0.5 + 0.3 + 0.1) / 3) < 1e-6, 'aggregated epsilon averages finite values');

    const getText = id => window.document.getElementById(id).textContent;
    assert.strictEqual(getText('episode'), '12', 'episode display reflects aggregated value');
    assert.strictEqual(getText('steps'), '37', 'steps display rounds aggregated average');
    assert.strictEqual(getText('reward'), '20.00', 'reward display reflects aggregated average');
    assert.strictEqual(getText('epsilon'), '0.30', 'epsilon display reflects aggregated average');
    assert.strictEqual(window.document.getElementById('epsilon-value').textContent, '0.30', 'epsilon pill mirrors aggregated epsilon');

    assert.strictEqual(pushes.length, 1, 'live chart receives a single aggregated update');
    assert.ok(Math.abs(pushes[0].reward - 20) < 1e-6, 'live chart reward uses aggregated value');
    assert.ok(Math.abs(pushes[0].epsilon - 0.3) < 1e-6, 'live chart epsilon uses aggregated value');

    agentCountSelect.value = '1';
    agentCountSelect.dispatchEvent(new window.Event('change', { bubbles: true }));
    assert.strictEqual(testApi.getTrainer().liveChart, liveChart, 'live chart reattaches when returning to single agent');
  } finally {
    restoreGlobals();
    dom.window.close();
  }
}
