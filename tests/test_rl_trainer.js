import { GridWorldEnvironment } from '../src/rl/environment.js';
import { RLTrainer } from '../src/rl/training.js';

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

  const env2 = new GridWorldEnvironment(2);
  class SlowAgent {
    constructor() {
      this.epsilon = 0.1;
    }
    act() {
      return new Promise(resolve => setTimeout(() => resolve(0), 20));
    }
    learn() {
      return new Promise(resolve => setTimeout(resolve, 20));
    }
  }
  const agent2 = new SlowAgent();
  const trainer2 = new RLTrainer(agent2, env2, { intervalMs: 5 });
  let concurrent = 0;
  let maxConcurrent = 0;
  const originalStep = trainer2.step.bind(trainer2);
  trainer2.step = async () => {
    concurrent++;
    if (concurrent > maxConcurrent) maxConcurrent = concurrent;
    await originalStep();
    concurrent--;
  };
  trainer2.start();
  await new Promise(resolve => setTimeout(resolve, 100));
  trainer2.pause();
  assert.strictEqual(maxConcurrent, 1);
}
