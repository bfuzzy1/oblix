import { GridWorldEnvironment } from '../src/rl/environment.js';
import { RLTrainer } from '../src/rl/training.js';

export async function run(assert) {
  const originalSetTimeout = global.setTimeout;
  const originalClearTimeout = global.clearTimeout;
  let timeoutFn = null;
  let currentMs = null;
  global.setTimeout = (fn, ms) => {
    timeoutFn = fn;
    currentMs = ms;
    return {};
  };
  global.clearTimeout = () => {
    timeoutFn = null;
  };

  const env = new GridWorldEnvironment(2);
  class StubAgent {
    constructor() {
      this.epsilon = 0.3;
      this.stateHistory = [];
    }
    act(state) {
      this.stateHistory.push(Array.from(state));
      return 3;
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
  await timeoutFn();
  agent.epsilon = 0.7;
  trainer.setIntervalMs(50);
  assert.strictEqual(currentMs, 50);
  await timeoutFn();
  trainer.pause();

  const pausedState = Array.from(trainer.state);
  assert.deepStrictEqual(pausedState, [1, 0]);
  timeoutFn = null;

  trainer.start();
  assert.deepStrictEqual(Array.from(trainer.state), pausedState);
  assert.strictEqual(currentMs, 50);
  assert.ok(timeoutFn);
  await timeoutFn();

  assert.deepStrictEqual(agent.stateHistory[0], [0, 0]);
  assert.deepStrictEqual(agent.stateHistory[1], pausedState);
  assert.deepStrictEqual(agent.stateHistory[2], pausedState);

  assert.strictEqual(epsilons.length, 4);
  assert.strictEqual(epsilons[0].toFixed(1), '0.3');
  assert.strictEqual(epsilons[1].toFixed(1), '0.3');
  assert.strictEqual(epsilons[2].toFixed(1), '0.7');
  assert.strictEqual(epsilons[3].toFixed(1), '0.7');

  trainer.pause();

  global.setTimeout = originalSetTimeout;
  global.clearTimeout = originalClearTimeout;
}
