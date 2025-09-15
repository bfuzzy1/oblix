import { OptimisticAgent } from '../src/rl/optimisticAgent.js';
import { saveAgent, loadAgent } from '../src/rl/storage.js';

export async function run(assert) {
  const state = new Float32Array([0, 0]);
  const key = Array.from(state).join(',');

  const agent = new OptimisticAgent({ initialValue: 2 });
  agent.act(state);
  const qVals = agent.qTable.get(key);
  assert.deepStrictEqual(Array.from(qVals), [2, 2, 2, 2]);
  agent.learn(state, 0, 1, state, true);
  assert.ok(agent.qTable.get(key)[0] < 2);

  const persistent = new OptimisticAgent({ initialValue: 2 });
  persistent.act(state);
  const storage = {
    data: {},
    setItem(key, value) { this.data[key] = value; },
    getItem(key) { return this.data[key] || null; }
  };
  const trainer = {
    agent: persistent,
    resetCalled: false,
    resetTrainerStateCalled: false,
    reset() { this.resetCalled = true; },
    resetTrainerState() { this.resetTrainerStateCalled = true; }
  };
  saveAgent(persistent, storage);
  trainer.agent = null;
  loadAgent(trainer, storage);
  assert.ok(trainer.resetTrainerStateCalled);
  assert.strictEqual(trainer.resetCalled, false);
  assert.ok(trainer.agent instanceof OptimisticAgent);
  const restored = trainer.agent.qTable.get(key);
  assert.deepStrictEqual(Array.from(restored), [2, 2, 2, 2]);
  assert.strictEqual(trainer.agent.initialValue, 2);
}
