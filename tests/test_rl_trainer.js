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

  const env3 = new GridWorldEnvironment(2);
  class ErrorAgent {
    constructor() {
      this.epsilon = 0.2;
    }
    act() { return 0; }
    learn() {}
  }
  const agent3 = new ErrorAgent();
  const trainer3 = new RLTrainer(agent3, env3, { intervalMs: 5 });
  let calls = 0;
  trainer3.step = async () => {
    calls++;
    if (calls === 1) throw new Error('fail');
  };
  trainer3.start();
  await new Promise(resolve => setTimeout(resolve, 40));
  trainer3.pause();
  assert.ok(calls >= 2);

  const env4 = new GridWorldEnvironment(2);
  class ReplayAgent {
    constructor() {
      this.epsilon = 0.3;
      this.learnCalls = [];
      this.tdErrorCalls = [];
    }
    act() {
      return 0;
    }
    async learn(state, action, reward, nextState, done, weight) {
      this.learnCalls.push({ state, action, reward, nextState, done, weight });
    }
    async computeTdError(state, action, reward, nextState, done) {
      this.tdErrorCalls.push({ state, action, reward, nextState, done });
      return 0.5;
    }
  }
  const agent4 = new ReplayAgent();
  const sampleTransitions = [
    {
      index: 0,
      state: new Float32Array([0, 0]),
      action: 1,
      reward: -0.01,
      nextState: new Float32Array([0, 1]),
      done: false,
      weight: 0.4
    },
    {
      index: 1,
      state: new Float32Array([0, 1]),
      action: 3,
      reward: 1,
      nextState: new Float32Array([1, 1]),
      done: true,
      weight: 0.9
    }
  ];
  class StubReplayBuffer {
    constructor(samples) {
      this.samples = samples;
      this.added = [];
      this.sampleCalls = [];
      this.priorityUpdates = [];
    }
    add(transition, priority) {
      this.added.push({ transition, priority });
    }
    sample(count) {
      this.sampleCalls.push(count);
      return this.samples.map(sample => ({ ...sample }));
    }
    updatePriority(index, priority) {
      this.priorityUpdates.push({ index, priority });
    }
  }
  const replayBuffer = new StubReplayBuffer(sampleTransitions);
  const trainer4 = new RLTrainer(agent4, env4, {
    replaySamples: sampleTransitions.length,
    replayStrategy: 'priority',
    replayBuffer
  });
  trainer4.state = env4.reset();
  await trainer4.step();
  assert.strictEqual(replayBuffer.added.length, 1);
  assert.deepStrictEqual(replayBuffer.sampleCalls, [sampleTransitions.length]);
  assert.strictEqual(agent4.learnCalls.length, 1 + sampleTransitions.length);
  assert.strictEqual(agent4.learnCalls[0].weight, undefined);
  for (let i = 0; i < sampleTransitions.length; i++) {
    const call = agent4.learnCalls[i + 1];
    const sample = sampleTransitions[i];
    assert.strictEqual(call.state, sample.state);
    assert.strictEqual(call.action, sample.action);
    assert.strictEqual(call.reward, sample.reward);
    assert.strictEqual(call.nextState, sample.nextState);
    assert.strictEqual(call.done, sample.done);
    assert.strictEqual(call.weight, sample.weight);
  }
  assert.strictEqual(agent4.tdErrorCalls.length, sampleTransitions.length);
  assert.strictEqual(replayBuffer.priorityUpdates.length, sampleTransitions.length);
  for (const update of replayBuffer.priorityUpdates) {
    assert.strictEqual(update.priority, 0.5);
  }
}
