import assert from 'assert';
import { oblixUtils } from '../src/utils.js';

/**
 *
 */
export async function run() {
  const targets = [1, 2, 3];
  const predsPerfect = [1, 2, 3];
  const r2Perfect = oblixUtils.calculateRSquared(predsPerfect, targets);
  assert.ok(Math.abs(r2Perfect - 1) < 1e-6);

  const predsOff = [1, 2, 4];
  const r2 = oblixUtils.calculateRSquared(predsOff, targets);
  const expectedMean = targets.reduce((a, b) => a + b, 0) / targets.length;
  const ssTot = targets.reduce((sum, t) => sum + (t - expectedMean) ** 2, 0);
  const ssRes = predsOff.reduce((sum, p, i) => sum + (targets[i] - p) ** 2, 0);
  const expectedR2 = 1 - ssRes / ssTot;
  assert.ok(Math.abs(r2 - expectedR2) < 1e-6);

  const bad = oblixUtils.calculateRSquared([1], [1, 2]);
  assert.ok(Number.isNaN(bad));
}
