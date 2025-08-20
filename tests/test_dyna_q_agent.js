import { DynaQAgent } from '../src/rl/dynaQAgent.js';
import { RLAgent } from '../src/rl/agent.js';

export async function run(assert) {
  const state = new Float32Array([0, 0]);
  const nextState = new Float32Array([1, 0]);
  const action = 2;

  const base = new RLAgent({ epsilon: 0, learningRate: 0.5 });
  base.learn(state, action, 1, nextState, false);
  const baseVal = base.qTable.get(Array.from(state).join(','))[action];

  const dyna = new DynaQAgent({ epsilon: 0, learningRate: 0.5, planningSteps: 5 });
  dyna.learn(state, action, 1, nextState, false);
  const dynaVal = dyna.qTable.get(Array.from(state).join(','))[action];

  assert.ok(dynaVal > baseVal);
}
