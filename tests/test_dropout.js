import assert from 'assert';
import { oblixLayerOps } from '../src/layers.js';

/**
 *
 */
export async function run() {
  const ctx1 = { isTraining: false, debug: false, masks: [], forwardCache: { activations: [0] } };
  const input = new Float32Array([1, 2, 3]);
  const out1 = oblixLayerOps.dropoutForward(ctx1, input, 0.5);
  assert.deepStrictEqual(Array.from(out1), Array.from(input));
  assert.strictEqual(ctx1.masks[0], null);
  const dOut = new Float32Array([0.1, 0.2, 0.3]);
  const dPass = oblixLayerOps.dropoutBackward(ctx1, dOut, 0);
  assert.deepStrictEqual(Array.from(dPass), Array.from(dOut));

  const ctx2 = { isTraining: true, debug: false, masks: [], forwardCache: { activations: [0] } };
  const seq = [0.1, 0.6, 0.7];
  let idx = 0;
  const orig = crypto.getRandomValues;
  crypto.getRandomValues = arr => {
    for (let i = 0; i < arr.length; i++) {
      arr[i] = Math.floor((seq[idx++] ?? 0) * 4294967296);
    }
  };
  const out2 = oblixLayerOps.dropoutForward(ctx2, input, 0.5);
  crypto.getRandomValues = orig;
  assert.deepStrictEqual(Array.from(out2), [0, 4, 6]);
  assert.deepStrictEqual(Array.from(ctx2.masks[0]), [0, 2, 2]);
  const dIn = oblixLayerOps.dropoutBackward(ctx2, new Float32Array([10, 20, 30]), 0);
  assert.deepStrictEqual(Array.from(dIn), [0, 40, 60]);
}
