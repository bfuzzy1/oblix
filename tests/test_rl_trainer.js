import { RLAgent } from '../src/rl/agent.js';
import { GridWorldEnvironment } from '../src/rl/environment.js';
import { RLTrainer } from '../src/rl/training.js';

export async function run(assert) {
  const env = new GridWorldEnvironment(2);
  const agent = new RLAgent({ epsilon: 0 });
  const steps = [];
  const trainer = new RLTrainer(agent, env, {
    intervalMs: 10,
    onStep: () => steps.push(1)
  });
  trainer.start();
  await new Promise(r => setTimeout(r, 35));
  trainer.pause();
  const count = steps.length;
  assert.ok(count >= 2);
  await new Promise(r => setTimeout(r, 30));
  assert.strictEqual(steps.length, count);
}
