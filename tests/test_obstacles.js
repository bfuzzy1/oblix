import assert from 'assert';
import { GridWorldEnvironment } from '../dist/rl/environment.js';

export async function run() {
  const obstacles = [{ x: 1, y: 0 }];
  const env = new GridWorldEnvironment(3, obstacles);
  env.reset();
  let res = env.step(3);
  assert.deepStrictEqual(Array.from(res.state), [0, 0]);
  env.reset();
  assert.deepStrictEqual(env.obstacles, obstacles);
  res = env.step(3);
  assert.deepStrictEqual(Array.from(res.state), [0, 0]);
}

