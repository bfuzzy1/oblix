import { RLAgent } from '../src/rl/agent.js';
import { QLambdaAgent } from '../src/rl/qLambdaAgent.js';

export async function run(assert) {
  const rl = new RLAgent({ epsilon: 0, learningRate: 0.5, gamma: 0.9 });
  const ql = new QLambdaAgent({ epsilon: 0, learningRate: 0.5, gamma: 0.9, lambda: 0.8 });
  const s0 = new Float32Array([0, 0]);
  const s1 = new Float32Array([1, 0]);
  const s2 = new Float32Array([2, 0]);
  rl.learn(s0, 3, -0.01, s1, false);
  ql.learn(s0, 3, -0.01, s1, false);
  rl.learn(s1, 3, 1, s2, true);
  ql.learn(s1, 3, 1, s2, true);
  const key = Array.from(s0).join(',');
  const rlVal = rl.qTable.get(key)[3];
  const qlVal = ql.qTable.get(key)[3];
  assert.ok(qlVal > rlVal);

  ql.qTable.set(key, new Float32Array([1, 2, 3, 4]));
  const json = ql.toJSON();
  const restored = QLambdaAgent.fromJSON(json);
  assert.strictEqual(restored.lambda, ql.lambda);
  assert.deepStrictEqual(Array.from(restored.qTable.get(key)), [1, 2, 3, 4]);
}
