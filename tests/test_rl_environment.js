import assert from 'assert';
import { GridWorldEnvironment } from '../src/rl/environment.js';
import { RLAgent } from '../src/rl/agent.js';
import { RLTrainer } from '../src/rl/training.js';
import { Oblix } from '../src/network.js';

export async function run() {
  const env = new GridWorldEnvironment(3);
  let state = env.reset();
  assert.deepStrictEqual(Array.from(state), [0, 0]);
  let res = env.step(3); // right
  assert.deepStrictEqual(Array.from(res.state), [1, 0]);
  res = env.step(1); // down
  assert.deepStrictEqual(Array.from(res.state), [1, 1]);

  const nn = new Oblix(false);
  nn.layer({ type: 'dense', inputSize: 2, outputSize: 4, activation: 'tanh' });
  const agent = new RLAgent(nn, { epsilon: 0.2, gamma: 0.9, learningRate: 0.05 });
  await RLTrainer.trainEpisodes(agent, env, 20, 20);
  const q = nn.predict(new Float32Array([0, 0]));
  assert.ok(q[3] >= q[2], 'Right action should have higher value than left');
}
