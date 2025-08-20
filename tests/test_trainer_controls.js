import { GridWorldEnvironment } from '../dist/rl/environment.js';
import { RLTrainer } from '../dist/rl/training.js';

export async function run(assert) {
  const originalSetInterval = global.setInterval;
  const originalClearInterval = global.clearInterval;
  let intervalFn = null;
  let currentMs = null;
  global.setInterval = (fn, ms) => {
    intervalFn = fn;
    currentMs = ms;
    return {};
  };
  global.clearInterval = () => {};

  const env = new GridWorldEnvironment(2);
  class StubAgent {
    constructor() {
      this.epsilon = 0.3;
    }
    act() {
      return 0;
    }
    learn() {}
  }
  const agent = new StubAgent();
  const epsilons = [];
  const trainer = new RLTrainer(agent, env, {
    intervalMs: 100,
    onStep: (state, reward, done, metrics) => {
      epsilons.push(metrics.epsilon);
    }
  });

  trainer.start();
  assert.strictEqual(currentMs, 100);
  await intervalFn();
  agent.epsilon = 0.7;
  trainer.setIntervalMs(50);
  assert.strictEqual(currentMs, 50);
  await intervalFn();
  trainer.pause();
  assert.strictEqual(epsilons[0].toFixed(1), '0.3');
  assert.strictEqual(epsilons[1].toFixed(1), '0.7');

  global.setInterval = originalSetInterval;
  global.clearInterval = originalClearInterval;
}
