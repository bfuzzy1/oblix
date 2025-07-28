import assert from 'assert';
import { Oblix } from '../src/network.js';

/**
 *
 */
export async function run() {
  const nn = new Oblix(false);
  nn.layer({ type: 'dense', inputSize: 2, outputSize: 3 });
  nn.layer({ type: 'layernorm', inputSize: 3 });
  const expected = 2 * 3 + 3 /*bias*/ + 3 /*gamma*/ + 3 /*beta*/;
  const total = nn.getTotalParameters();
  assert.strictEqual(total, expected);
}
