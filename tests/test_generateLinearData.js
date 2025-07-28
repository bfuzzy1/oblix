import assert from 'assert';
import { oblixUtils } from '../src/utils.js';

/**
 *
 */
export async function run() {
  const data = oblixUtils.generateLinearData(4, 0.2, 2, 3);
  assert.strictEqual(data.length, 4);
  for (const item of data) {
    assert.strictEqual(item.input.length, 2);
    assert.strictEqual(item.output.length, 3);
    for (const v of item.input.concat(item.output)) {
      assert.ok(typeof v === 'number');
    }
  }
}
