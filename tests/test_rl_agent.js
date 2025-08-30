import { RLAgent } from '../src/rl/agent.js';
import { saveAgent, loadAgent } from '../src/rl/storage.js';
import { mapToObject, objectToMap } from '../src/rl/utils/serialization.js';

export async function run(assert) {
  const map = new Map([['a', new Float32Array([1, 2])]]);
  const obj = mapToObject(map);
  assert.deepStrictEqual(obj, { a: [1, 2] });
  const restored = objectToMap(obj, Float32Array);
  assert.deepStrictEqual(Array.from(restored.get('a')), [1, 2]);

  const agent = new RLAgent({
    epsilon: 1,
    epsilonDecay: 0.5,
    minEpsilon: 0.2,
    learningRate: 0.5,
  });
  const state = new Float32Array([0, 0]);
  agent.learn(state, 0, 0, state, false);
  assert.strictEqual(agent.epsilon, 0.5);
  agent.learn(state, 0, 0, state, false);
  assert.strictEqual(agent.epsilon, 0.25);
  agent.learn(state, 0, 0, state, false);
  assert.strictEqual(agent.epsilon, 0.2);

  const greedyAgent = new RLAgent({ epsilon: 0 });
  const key = Array.from(state).join(',');
  greedyAgent.qTable.set(key, new Float32Array([1, 5, 3, 2]));
  const action = greedyAgent.act(state);
  assert.strictEqual(action, 1);

  const data = greedyAgent.toJSON();
  const loadedAgent = RLAgent.fromJSON(data);
  const loadedAction = loadedAgent.act(state);
  assert.strictEqual(loadedAction, action);

  const storage = {
    data: {},
    setItem(key, value) { this.data[key] = value; },
    getItem(key) { return this.data[key] || null; }
  };
  const trainer = {
    agent: greedyAgent,
    resetCalled: false,
    reset() { this.resetCalled = true; }
  };
  saveAgent(greedyAgent, storage);
  trainer.agent = new RLAgent();
  loadAgent(trainer, storage);
  assert.ok(trainer.resetCalled);
  const restoredAction = trainer.agent.act(state);
  assert.strictEqual(restoredAction, action);
}
