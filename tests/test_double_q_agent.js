import { RLAgent } from '../src/rl/agent.js';
import { DoubleQAgent } from '../src/rl/doubleQAgent.js';

function createRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

class RandomRewardEnv {
  constructor(rng) {
    this.rng = rng;
    this.state = new Float32Array([0, 0]);
  }
  reset() {
    return this.state;
  }
  step(action) {
    return { state: this.state, reward: this.rng(), done: false };
  }
}

async function train(agent, steps, seed) {
  const rng = createRng(seed);
  const env = new RandomRewardEnv(rng);
  const orig = Math.random;
  Math.random = rng;
  let state = env.reset();
  for (let i = 0; i < steps; i++) {
    const action = agent.act(state);
    const { state: next, reward, done } = env.step(action);
    agent.learn(state, action, reward, next, done);
    state = next;
  }
  Math.random = orig;
  return agent;
}

export async function run(assert) {
  const steps = 5000;
  const qAgent = await train(new RLAgent({ epsilon: 1, epsilonDecay: 1, gamma: 0.9, learningRate: 0.1 }), steps, 42);
  const dqAgent = await train(new DoubleQAgent({ epsilon: 1, epsilonDecay: 1, gamma: 0.9, learningRate: 0.1 }), steps, 42);
  const key = '0,0';
  const qVals = qAgent.qTable.get(key);
  const qMax = Math.max(...qVals);
  const qa = dqAgent.qTableA.get(key);
  const qb = dqAgent.qTableB.get(key);
  const avg = qa.map((v, i) => (v + qb[i]) / 2);
  const dMax = Math.max(...avg);
  assert.ok(qMax > dMax);
}
