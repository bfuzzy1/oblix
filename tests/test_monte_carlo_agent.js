import { MonteCarloAgent } from '../src/rl/monteCarloAgent.js';
import { GridWorldEnvironment } from '../src/rl/environment.js';

export async function run(assert) {
  const env = new GridWorldEnvironment(2);
  const agent = new MonteCarloAgent({ gamma: 1, epsilon: 0 });
  const expectedFirst = -0.01 + 1;
  const expectedSecond = 1;
  for (let i = 0; i < 3; i++) {
    let state = env.reset();
    let res = env.step(3);
    agent.learn(state, 3, res.reward, res.state, res.done);
    state = res.state;
    res = env.step(1);
    agent.learn(state, 1, res.reward, res.state, res.done);
  }
  const keyStart = Array.from(new Float32Array([0, 0])).join(',');
  const keyRight = Array.from(new Float32Array([1, 0])).join(',');
  const qStart = agent.qTable.get(keyStart);
  const qRight = agent.qTable.get(keyRight);
  assert.ok(Math.abs(qStart[3] - expectedFirst) < 1e-6);
  assert.ok(Math.abs(qRight[1] - expectedSecond) < 1e-6);
}
