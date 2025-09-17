import assert from 'assert';
import { GridWorldEnvironment } from '../src/rl/environment.js';
import { RLAgent } from '../src/rl/agent.js';
import { RLTrainer } from '../src/rl/training.js';

export async function run() {
  const env = new GridWorldEnvironment(3);
  let state = env.reset();
  assert.deepStrictEqual(Array.from(state), [0, 0]);
  let res = env.step(3); // right
  assert.deepStrictEqual(Array.from(res.state), [1, 0]);
  res = env.step(1); // down
  assert.deepStrictEqual(Array.from(res.state), [1, 1]);

  const agent = new RLAgent({ epsilon: 0.2, gamma: 0.9, learningRate: 0.5 });
  await RLTrainer.trainEpisodes(agent, env, 20, 20);
  const key = Array.from(new Float32Array([0, 0])).join(',');
  const q = agent.qTable.get(key);
  assert.ok(q[3] >= q[2], 'Right action should have higher value than left');

  const customEnv = new GridWorldEnvironment(2, [], {
    stepPenalty: -0.5,
    obstaclePenalty: -2,
    goalReward: 5
  });
  assert.deepStrictEqual(customEnv.getRewardConfig(), {
    stepPenalty: -0.5,
    obstaclePenalty: -2,
    goalReward: 5
  });
  customEnv.reset();
  let stepResult = customEnv.step(3);
  assert.strictEqual(stepResult.reward, -0.5);
  assert.strictEqual(stepResult.done, false);
  customEnv.reset();
  customEnv.toggleObstacle(1, 0);
  stepResult = customEnv.step(3);
  assert.strictEqual(stepResult.reward, -2);
  assert.strictEqual(stepResult.done, false);
  customEnv.toggleObstacle(1, 0);
  customEnv.reset();
  customEnv.step(3);
  stepResult = customEnv.step(1);
  assert.strictEqual(stepResult.reward, 5);
  assert.strictEqual(stepResult.done, true);

  const cells = env.enumerateCells();
  assert.strictEqual(cells.length, env.size * env.size);
  assert.ok(cells.some(cell => cell.x === 0 && cell.y === 0));
  const states = env.enumerateStates();
  assert.strictEqual(states.length, cells.length);
  const transition = env.getTransition(new Float32Array([0, 0]), 3);
  assert.deepStrictEqual(Array.from(transition.state), [1, 0]);
  assert.strictEqual(transition.reward, env.stepPenalty);
  assert.strictEqual(transition.done, false);
  const goalTransition = env.getTransition(new Float32Array([env.size - 1, env.size - 2]), 1);
  assert.strictEqual(goalTransition.done, true);
  assert.strictEqual(goalTransition.reward, env.goalReward);
}
