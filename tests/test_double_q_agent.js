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
  {
    const agent = new DoubleQAgent();
    const state = new Float32Array([0, 0]);
    const key = Array.from(state).join(',');
    agent.qTableA.set(key, new Float32Array([1, 2, 3, 4]));
    agent.qTableB.set(key, new Float32Array([4, 3, 2, 1]));
    let called = false;
    agent._selectAction = (qVals, s, update) => {
      called = true;
      assert.deepStrictEqual(Array.from(qVals), [2.5, 2.5, 2.5, 2.5]);
      assert.strictEqual(s, state);
      assert.strictEqual(update, false);
      return 3;
    };
    const action = agent.act(state, false);
    assert.ok(called);
    assert.strictEqual(action, 3);
  }

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

  dqAgent.qTableA.set(key, new Float32Array([1, 2, 3, 4]));
  dqAgent.qTableB.set(key, new Float32Array([4, 3, 2, 1]));
  dqAgent.countTable.set(key, new Uint32Array([1, 2, 3, 4]));
  const json = dqAgent.toJSON();
  const restored = DoubleQAgent.fromJSON(json);
  assert.deepStrictEqual(Array.from(restored.qTableA.get(key)), [1, 2, 3, 4]);
  assert.deepStrictEqual(Array.from(restored.qTableB.get(key)), [4, 3, 2, 1]);
  assert.deepStrictEqual(Array.from(restored.countTable.get(key)), [1, 2, 3, 4]);
}
