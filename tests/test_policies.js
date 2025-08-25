import { RLAgent } from '../src/rl/agent.js';

export async function run(assert) {
  const state = new Float32Array([0, 0]);
  const key = Array.from(state).join(',');

  const greedyAgent = new RLAgent({ epsilon: 1, policy: 'greedy' });
  greedyAgent.qTable.set(key, new Float32Array([1, 5, 3, 2]));
  assert.strictEqual(greedyAgent.act(state), 1);

  const softAgent = new RLAgent({ policy: 'softmax', temperature: 1 });
  softAgent.qTable.set(key, new Float32Array([1, 5, 1, 1]));
  const origRandom = Math.random;
  Math.random = () => 0.5;
  assert.strictEqual(softAgent.act(state), 1);
  Math.random = () => 0;
  assert.strictEqual(softAgent.act(state), 0);
  Math.random = origRandom;
}
