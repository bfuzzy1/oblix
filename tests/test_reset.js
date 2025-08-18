import { RLAgent } from '../src/rl/agent.js';
import { GridWorldEnvironment } from '../src/rl/environment.js';
import { RLTrainer } from '../src/rl/training.js';

export async function run(assert) {
  const agent = new RLAgent({ epsilon: 1, epsilonDecay: 0.5 });
  const env = new GridWorldEnvironment(2);
  const trainer = new RLTrainer(agent, env);
  trainer.state = env.reset();
  await trainer.step();
  assert.ok(agent.qTable.size > 0);
  assert.strictEqual(agent.epsilon, 0.5);
  trainer.reset();
  assert.strictEqual(agent.qTable.size, 0);
  assert.strictEqual(agent.epsilon, 1);
  assert.deepStrictEqual(trainer.metrics, {
    episode: 1,
    steps: 0,
    cumulativeReward: 0,
    epsilon: 1
  });
  assert.deepStrictEqual(Array.from(trainer.state), [0, 0]);
}
