import assert from 'assert';
import { oblixUtils } from '../src/utils.js';

export async function run() {
  const input = new Float32Array([0, 0]);
  const result = oblixUtils.positionalEncoding(input);
  assert.strictEqual(result.length, 2);
  assert.ok(Math.abs(result[0] - 0) < 1e-6);
  assert.ok(Math.abs(result[1] - Math.cos(1)) < 1e-6);

  const arrInput = [1, 2];
  const out2 = oblixUtils.positionalEncoding(arrInput);
  assert.strictEqual(out2.length, 2);
  assert.ok(out2 instanceof Float32Array);
}
