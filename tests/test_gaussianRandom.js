import assert from 'assert';
import { oblixUtils } from '../src/utils.js';

export async function run() {
  oblixUtils._gaussian_spare = null;
  const first = oblixUtils.gaussianRandom();
  assert.strictEqual(typeof first, 'number');
  assert.notStrictEqual(oblixUtils._gaussian_spare, null);
  const spareStored = oblixUtils._gaussian_spare;
  const second = oblixUtils.gaussianRandom();
  assert.strictEqual(typeof second, 'number');
  assert.strictEqual(oblixUtils._gaussian_spare, null);
  assert.strictEqual(second, spareStored);
}
