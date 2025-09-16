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

  // Priority weights
  const weightBuffer = new ExperienceReplay(5, 1, 0.5);
  weightBuffer.add({ state: 'low' }, 1);
  weightBuffer.add({ state: 'high' }, 10);
  const originalRandom = Math.random;
  let weightCalls = 0;
  const sequence = [0.05, 0.95];
  Math.random = () => {
    const value = sequence[weightCalls % sequence.length];
    weightCalls += 1;
    return value;
  };
  const weightedSamples = weightBuffer.sample(2, 'priority');
  Math.random = originalRandom;
  const lowSample = weightedSamples.find(t => t.state === 'low');
  const highSample = weightedSamples.find(t => t.state === 'high');
  testAssert.ok(lowSample);
  testAssert.ok(highSample);
  testAssert.ok(highSample.weight > lowSample.weight);
  testAssert.ok(highSample.weight <= 1);
  testAssert.ok(lowSample.weight >= 0);

  const zeroPriorityBuffer = new ExperienceReplay(3, 1, 0.2, 0.5);
  zeroPriorityBuffer.add({ state: 'first' }, 0);
  zeroPriorityBuffer.add({ state: 'second' }, 0);
  const zeroRandom = Math.random;
  Math.random = () => 0.1;
  const zeroSamples = zeroPriorityBuffer.sample(1, 'priority');
  Math.random = zeroRandom;
  testAssert.strictEqual(zeroSamples.length, 1);
  testAssert.strictEqual(zeroSamples[0].weight, 1);
  testAssert.strictEqual(zeroPriorityBuffer.beta, 0.7);

  // Beta annealing
  const annealBuffer = new ExperienceReplay(5, 1, 0.2, 0.3);
  annealBuffer.add({ state: 'x' }, 1);
  annealBuffer.add({ state: 'y' }, 5);
  const startingBeta = annealBuffer.beta;
  const betaRandom = Math.random;
  Math.random = () => 0.5;
  annealBuffer.sample(1, 'priority');
  Math.random = betaRandom;
  testAssert.strictEqual(annealBuffer.beta, Math.min(1, startingBeta + annealBuffer.betaIncrement));

  const noSampleBuffer = new ExperienceReplay(5, 1, 0.3, 0.4);
  noSampleBuffer.add({ state: 'noop' }, 1);
  const betaBeforeNoSample = noSampleBuffer.beta;
  const emptyResult = noSampleBuffer.sample(0, 'priority');
  testAssert.deepStrictEqual(emptyResult, []);
  testAssert.strictEqual(noSampleBuffer.beta, betaBeforeNoSample);

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
  testAssert.strictEqual(agent.calls[0].length, 5);
  testAssert.strictEqual(agent.calls[1].length, 6);
  testAssert.deepStrictEqual(agent.calls[0], agent.calls[1].slice(0, 5));
  testAssert.strictEqual(agent.calls[1][5], 1);
}
