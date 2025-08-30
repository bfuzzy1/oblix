import { ExperienceReplay } from '../src/rl/experienceReplay.js';
import { GridWorldEnvironment } from '../src/rl/environment.js';
import { RLTrainer } from '../src/rl/training.js';

export async function run(testAssert) {
  // Insertion and capacity
  const buffer = new ExperienceReplay(2);
  buffer.add({ state: 1, action: 0, reward: 0, nextState: 2, done: false }, 1);
  buffer.add({ state: 2, action: 1, reward: 1, nextState: 3, done: false }, 2);
  buffer.add({ state: 3, action: 2, reward: 2, nextState: 4, done: true }, 3);
  testAssert.strictEqual(buffer.buffer.length, 2);
  const states = buffer.buffer.map(t => t.state).sort();
  testAssert.deepStrictEqual(states, [2, 3]);

  // Sampling strategies
  const stratBuffer = new ExperienceReplay(3);
  stratBuffer.add({ state: 'a' }, 0);
  stratBuffer.add({ state: 'b' }, 5);
  stratBuffer.add({ state: 'c' }, 0);
  const [sampled] = stratBuffer.sample(1, 'priority');
  testAssert.strictEqual(sampled.state, 'b');

  const uniformSamples = stratBuffer.sample(2, 'uniform');
  testAssert.strictEqual(uniformSamples.length, 2);

  // Integration with RLTrainer
  class StubAgent {
    constructor() {
      this.epsilon = 0;
      this.calls = [];
    }
    act() {
      return 0;
    }
    async learn(...args) {
      this.calls.push(args);
    }
  }
  const env = new GridWorldEnvironment(2);
  const agent = new StubAgent();
  const replay = new ExperienceReplay(5);
  const trainer = new RLTrainer(agent, env, {
    replayBuffer: replay,
    replaySamples: 1,
    replayStrategy: 'uniform'
  });
  trainer.state = env.reset();
  await trainer.step();
  testAssert.strictEqual(replay.buffer.length, 1);
  testAssert.strictEqual(agent.calls.length, 2);
  testAssert.deepStrictEqual(agent.calls[0], agent.calls[1]);
}
