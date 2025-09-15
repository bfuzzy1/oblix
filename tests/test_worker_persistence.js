import { JSDOM } from 'jsdom';
import { bindControls } from '../src/ui/bindControls.js';
import { RLAgent } from '../src/rl/agent.js';
import { POLICIES } from '../src/rl/policies.js';

export async function run(assert) {
  const dom = new JSDOM(`<!doctype html><body>
    <select id="agent-select"><option value="rl" selected>RL</option></select>
    <select id="policy-select"><option value="epsilon-greedy" selected>Epsilon Greedy</option></select>
    <input id="epsilon-slider" value="0.5" />
    <span id="epsilon-value"></span>
    <span id="epsilon"></span>
    <input id="interval-slider" value="100" />
    <span id="interval-value"></span>
    <input id="learning-rate-slider" value="0.1" />
    <span id="learning-rate-value"></span>
    <input id="lambda-slider" value="0" />
    <span id="lambda-value"></span>
    <button id="start"></button>
    <button id="pause"></button>
    <button id="reset"></button>
    <button id="save"></button>
    <button id="load"></button>
  </body>`);

  global.window = dom.window;
  global.document = dom.window.document;
  global.HTMLElement = dom.window.HTMLElement;

  const storage = {
    data: {},
    setItem(key, value) {
      this.data[key] = value;
    },
    getItem(key) {
      return this.data[key] ?? null;
    }
  };

  global.localStorage = storage;

  const agent = new RLAgent({ epsilon: 0.5, policy: POLICIES.EPSILON_GREEDY });
  agent.toJSON = () => ({ source: 'proxy' });

  const snapshot = { type: 'rl', source: 'worker' };

  const trainer = {
    intervalMs: 100,
    metrics: { epsilon: agent.epsilon },
    agent,
    start() {},
    pause() {},
    reset() {},
    resetTrainerState() {},
    setIntervalMs() {},
    getAgentState: async () => snapshot,
    state: new Float32Array([0, 0])
  };

  const environment = { size: 5, obstacles: [] };
  const getEnv = () => environment;
  const setEnv = (size, obstacles) => {
    environment.size = size;
    environment.obstacles = obstacles;
  };

  bindControls(trainer, agent, () => {}, getEnv, setEnv);

  const saveHandler = document.getElementById('save').onclick;
  await saveHandler();

  assert.ok(storage.data.agent, 'agent should be persisted');
  assert.deepStrictEqual(JSON.parse(storage.data.agent), snapshot);

  dom.window.close();
  delete global.window;
  delete global.document;
  delete global.HTMLElement;
  delete global.localStorage;
}
