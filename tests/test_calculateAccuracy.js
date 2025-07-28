import assert from 'assert';
import { oblixUtils } from '../src/utils.js';

/**
 *
 */
export async function run() {
  const preds = [
    [0.1, 0.9],
    [0.8, 0.2]
  ];
  const oneHotTargets = [
    [0, 1],
    [1, 0]
  ];
  const acc = oblixUtils.calculateAccuracy(preds, oneHotTargets);
  assert.ok(Math.abs(acc - 1) < 1e-6);

  const predsTyped = [
    new Float32Array([0.1, 0.9]),
    new Float32Array([0.8, 0.2])
  ];
  const typedTargets = [
    new Float32Array([0, 1]),
    new Float32Array([1, 0])
  ];
  const accTyped = oblixUtils.calculateAccuracy(predsTyped, typedTargets);
  assert.ok(Math.abs(accTyped - 1) < 1e-6);
}
