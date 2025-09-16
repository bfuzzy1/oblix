import fs from 'fs';
import { GridWorldEnvironment } from '../src/rl/environment.js';
import { saveEnvironment, loadEnvironment } from '../src/rl/storage.js';

export async function run(assert) {
  const html = fs.readFileSync('index.html', 'utf8');
  assert.ok(html.includes('id="grid-size"'));
  assert.ok(html.includes('id="step-penalty"'));
  assert.ok(html.includes('id="obstacle-penalty"'));
  assert.ok(html.includes('id="goal-reward"'));

  const js = fs.readFileSync('src/ui/renderGrid.js', 'utf8');
  assert.ok(js.includes('saveEnvironment(env);'));

  const storage = {
    data: {},
    setItem(k, v) { this.data[k] = v; },
    getItem(k) { return this.data[k] ?? null; }
  };

  const env = new GridWorldEnvironment(4, [{ x: 1, y: 1 }], {
    stepPenalty: -0.25,
    obstaclePenalty: -1,
    goalReward: 3
  });
  saveEnvironment(env, storage);
  const loaded = loadEnvironment(storage);
  assert.deepStrictEqual(loaded, {
    size: 4,
    obstacles: [{ x: 1, y: 1 }],
    rewards: { stepPenalty: -0.25, obstaclePenalty: -1, goalReward: 3 }
  });
}
