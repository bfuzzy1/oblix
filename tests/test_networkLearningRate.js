import assert from 'assert';
import { Oblix } from '../src/network.js';

export async function run() {
  const nn = new Oblix(false);
  let rate = nn.getCurrentLearningRate(0, 0.1, {
    lrSchedule: 'step',
    lrStepDecayFactor: 0.5,
    lrStepDecaySize: 2
  });
  assert.ok(Math.abs(rate - 0.1) < 1e-12);
  rate = nn.getCurrentLearningRate(3, 0.1, {
    lrSchedule: 'step',
    lrStepDecayFactor: 0.5,
    lrStepDecaySize: 2
  });
  assert.ok(Math.abs(rate - 0.05) < 1e-12);

  rate = nn.getCurrentLearningRate(2, 0.1, {
    lrSchedule: 'exponential',
    lrExpDecayRate: 0.9
  });
  assert.ok(Math.abs(rate - 0.1 * Math.pow(0.9, 2)) < 1e-12);
}
