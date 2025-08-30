import { OptimisticAgent } from '../src/rl/optimisticAgent.js';

export async function run(assert) {
  const state = new Float32Array([0, 0]);
  const agent = new OptimisticAgent({ initialValue: 2 });
  agent.act(state);
  const key = Array.from(state).join(',');
  const qVals = agent.qTable.get(key);
  assert.deepStrictEqual(Array.from(qVals), [2, 2, 2, 2]);
  agent.learn(state, 0, 1, state, true);
  assert.ok(agent.qTable.get(key)[0] < 2);
}
