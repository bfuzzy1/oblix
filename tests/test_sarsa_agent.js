import { SarsaAgent } from '../dist/rl/sarsaAgent.js';

export async function run(assert) {
  const agent = new SarsaAgent({
    epsilon: 0,
    learningRate: 0.5,
    gamma: 0.9
  });
  const state = new Float32Array([0, 0]);
  const nextState = new Float32Array([1, 0]);
  const keyNext = Array.from(nextState).join(',');
  agent.qTable.set(keyNext, new Float32Array([0, 10, 5, 0]));
  agent.learn(state, 0, 1, nextState, false);
  const keyState = Array.from(state).join(',');
  const updated = agent.qTable.get(keyState)[0];
  assert.strictEqual(updated, 5);

  const decayAgent = new SarsaAgent({
    epsilon: 1,
    epsilonDecay: 0.5,
    minEpsilon: 0.2
  });
  const s = new Float32Array([0, 0]);
  decayAgent.learn(s, 0, 0, s, false);
  assert.strictEqual(decayAgent.epsilon, 0.5);
  decayAgent.learn(s, 0, 0, s, false);
  assert.strictEqual(decayAgent.epsilon, 0.25);
  decayAgent.learn(s, 0, 0, s, false);
  assert.strictEqual(decayAgent.epsilon, 0.2);
}
