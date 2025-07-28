import assert from 'assert';
import { oblixUtils } from '../src/utils.js';

/**
 *
 */
export async function run() {
  const data = oblixUtils.generateXORData(5, 0);
  assert.strictEqual(data.length, 5);
  for (const item of data) {
    assert.strictEqual(item.input.length, 2);
    assert.strictEqual(item.output.length, 1);
    const a = Math.round(item.input[0]);
    const b = Math.round(item.input[1]);
    assert.strictEqual((a + b) % 2, item.output[0]);
  }
}
