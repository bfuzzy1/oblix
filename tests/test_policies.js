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
  Math.random = () => { origRandom(); return 0.5; };
  assert.strictEqual(softAgent.act(state), 1);
  Math.random = () => { origRandom(); return 0; };
  assert.strictEqual(softAgent.act(state), 0);
  Math.random = origRandom;

  const ucbAgent = new RLAgent({ policy: 'ucb', ucbC: 1 });
  ucbAgent.qTable.set(key, new Float32Array([1, 5, 3, 2]));
  assert.strictEqual(ucbAgent.act(state, false), 0);
  assert.strictEqual(ucbAgent.act(state), 0);
  assert.strictEqual(ucbAgent.act(state), 1);
  assert.strictEqual(ucbAgent.act(state), 2);
  assert.strictEqual(ucbAgent.act(state), 3);
  assert.strictEqual(ucbAgent.act(state), 1);

  const thompAgent = new RLAgent({ policy: 'thompson' });
  thompAgent.qTable.set(key, new Float32Array([1, 5, 3, 2]));
  thompAgent._gaussian = () => 0;
  assert.strictEqual(thompAgent.act(state), 1);
  assert.strictEqual(thompAgent.countTable.get(key)[1], 1);

  const randAgent = new RLAgent({ policy: 'random' });
  const origRand = Math.random;
  Math.random = () => { origRand(); return 0.95; };
  assert.strictEqual(randAgent.act(state), 3);
  Math.random = () => { origRand(); return 0.1; };
  assert.strictEqual(randAgent.act(state), 0);
  Math.random = origRand;
}
