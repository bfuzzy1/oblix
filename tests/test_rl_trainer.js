import { GridWorldEnvironment } from '../dist/rl/environment.js';
import { RLTrainer } from '../dist/rl/training.js';

export async function run(assert) {
  const env = new GridWorldEnvironment(2);
  class StubAgent {
    constructor() {
      this.epsilon = 0.1;
      this.actions = [3, 1];
      this.i = 0;
    }
    act() {
      return this.actions[this.i++];
    }
    learn() {}
  }
  const agent = new StubAgent();
  const reports = [];
  const trainer = new RLTrainer(agent, env, {
    onStep: (state, reward, done, metrics) => {
      reports.push({ reward, done, metrics });
    }
  });
  trainer.state = env.reset();
  await trainer.step();
  await trainer.step();
  assert.strictEqual(reports.length, 3);
  assert.deepStrictEqual(reports[0].metrics, {
    episode: 1,
    steps: 1,
    cumulativeReward: -0.01,
    epsilon: 0.1
  });
  assert.strictEqual(reports[1].done, true);
  assert.deepStrictEqual(reports[1].metrics, {
    episode: 1,
    steps: 2,
    cumulativeReward: 0.99,
    epsilon: 0.1
  });
  assert.deepStrictEqual(reports[2].metrics, {
    episode: 2,
    steps: 0,
    cumulativeReward: 0,
    epsilon: 0.1
  });
  assert.deepStrictEqual(trainer.metrics, {
    episode: 2,
    steps: 0,
    cumulativeReward: 0,
    epsilon: 0.1
  });
  assert.strictEqual(trainer.episodeRewards.length, 1);
  assert.strictEqual(trainer.episodeRewards[0], 0.99);
}
