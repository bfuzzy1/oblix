import assert from 'assert';
import { GridWorldEnvironment } from '../src/rl/environment.js';
import { RLAgent } from '../src/rl/agent.js';
import { RLTrainer } from '../src/rl/training.js';

export async function run() {
  const env = new GridWorldEnvironment(3);
  let state = env.reset();
  assert.deepStrictEqual(Array.from(state), [0, 0]);
  let res = env.step(3); // right
  assert.deepStrictEqual(Array.from(res.state), [1, 0]);
  res = env.step(1); // down
  assert.deepStrictEqual(Array.from(res.state), [1, 1]);

  const agent = new RLAgent({ epsilon: 0.2, gamma: 0.9, learningRate: 0.5 });
  await RLTrainer.trainEpisodes(agent, env, 20, 20);
  const key = Array.from(new Float32Array([0, 0])).join(',');
  const q = agent.qTable.get(key);
  assert.ok(q[3] >= q[2], 'Right action should have higher value than left');
}
